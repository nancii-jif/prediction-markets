"""One startup discovery and fixed-cohort lifecycle polling in the local process.

Only blocking public HTTP requests run in workers. They return payloads before
any SQLite access; normalization and exchange mutations run on the event loop.
There is no standalone listener or recurring candidate scan.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from .. import config
from ..integrations import kalshi as kalshi_client
from . import selection as live_markets
from ..exchange import Exchange

log = logging.getLogger(__name__)


def payout_cents(payload: dict) -> int | None:
    """Parse the source YES redemption value without guessing or rounding.

    Kalshi's dollar field is distinct from market prices and from result text.
    Even a supported value is not paid until the status is finalized.
    """
    value = payload.get("settlement_value_dollars")
    if value is None or value == "":
        return None
    try:
        if not isinstance(value, str) or re.fullmatch(r"(?:0|1)(?:\.[0-9]{1,4})?", value) is None:
            raise ValueError("settlement_value_dollars must be a supported decimal string")
        cents = Decimal(value) * 100
        if not cents.is_finite() or not 0 <= cents <= 100 or cents != cents.to_integral_value():
            raise ValueError("payout must be an exact whole number of cents from 0 to 100")
        return int(cents)
    except (InvalidOperation, ValueError):
        log.error("unsupported payout for %s: %r; pending", payload.get("ticker"), value)
        return None


class MarketFeed:
    """Lifecycle wiring shared by the runner and offline tests."""

    def __init__(
        self, exchange: Exchange, settings: config.Settings,
        *, now: Callable[[], int] | None = None,
    ) -> None:
        self.exchange = exchange
        self.settings = settings
        self._now = now if now is not None else lambda: int(time.time())
        self._poll_lock = asyncio.Lock()
        self._initialization_attempted = False

    async def initialize(self) -> list[str]:
        """Select and persist the entire cohort before any participant starts."""
        if self._initialization_attempted or self.exchange.unsettled_tickers() or self.exchange.all_markets_settled():
            raise RuntimeError("market selection may run only once")
        self._initialization_attempted = True
        startup_at = self._now()
        candidates = await asyncio.to_thread(
            kalshi_client.get_market_candidates,
            base_url=self.settings.kalshi_base_url,
        )
        rows = live_markets.select_markets(
            candidates, self.settings, startup_at=startup_at, observed_at=self._now(),
        )
        selected = {row["ticker"] for row in rows}
        raw_markets = {
            payload["ticker"]: payload.get("_raw_market", payload)
            for payload in candidates if isinstance(payload, dict) and payload.get("ticker") in selected
        }
        self.exchange.store_market_cohort(
            rows, raw_markets=raw_markets,
            source_url=self.settings.kalshi_base_url.rstrip("/") + "/events",
        )
        log.info("selected fixed cohort: %s", ", ".join(row["ticker"] for row in rows))
        return self.exchange.unsettled_tickers()

    async def poll_once(self) -> dict:
        """Refresh only unsettled members; missing/failed observations age out."""
        async with self._poll_lock:
            self.exchange.close_due_markets()
            tickers = self.exchange.unsettled_tickers()
            stats = {"updated": [], "settled": [], "failed": False}
            if not tickers:
                return stats
            try:
                payloads = await asyncio.to_thread(
                    kalshi_client.get_markets_by_tickers, tickers,
                    base_url=self.settings.kalshi_base_url,
                )
            except (kalshi_client.KalshiError, OSError, ValueError):
                log.exception("fixed-cohort fetch failed; retaining pending positions")
                stats["failed"] = True
                return stats
            finally:
                self.exchange.close_due_markets()

            observed_at = self._now()
            wanted = set(tickers)
            seen = set()
            for payload in payloads:
                if not isinstance(payload, dict) or not isinstance(payload.get("ticker"), str):
                    log.error("malformed market response; freshness not advanced")
                    continue
                ticker = payload["ticker"]
                if ticker not in wanted or ticker in seen:
                    log.error("unexpected or duplicate market %s in fixed-cohort response", ticker)
                    continue
                seen.add(ticker)
                try:
                    row = live_markets.market_row(payload, observed_at)
                    row["payout_cents"] = payout_cents(payload)
                    result = self.exchange.apply_market_observation(row)
                except ValueError:
                    log.exception("invalid market observation for %s; freshness not advanced", ticker)
                    continue
                stats["updated"].append(ticker)
                if result["settled"]:
                    stats["settled"].append(ticker)
            for ticker in sorted(wanted - seen):
                log.error("selected market %s missing from poll; retaining last observation", ticker)
            return stats

    async def resume(self) -> dict:
        """Refresh the stored cohort; never run discovery or selection filters."""
        if self._initialization_attempted:
            raise RuntimeError("market initialization may run only once")
        self._initialization_attempted = True
        # Even a recent parent's observation is not a resume-time observation.
        # Missing/invalid payloads must not leave a market admitted for trading.
        self.exchange.invalidate_market_observations()
        stats = await self.poll_once()
        if stats["failed"] or (not stats["updated"] and self.exchange.unsettled_tickers()):
            raise ValueError("cannot resume: no successful fresh observations for the original markets")
        log.info("refreshed original cohort for resume: %s", stats)
        return stats

    async def run(self) -> None:
        """Poll serially every interval and enforce cutoffs during slow HTTP.

        A single in-flight poll and the next local cutoff share this timer loop.
        Elapsed waits use the event loop's monotonic clock. Completion signals
        that all selected markets settled; cancellation is the shutdown path.
        """
        if not self.exchange.unsettled_tickers():
            if self.exchange.all_markets_settled():
                return
            raise RuntimeError("select the market cohort before starting polling")
        loop = asyncio.get_running_loop()
        next_poll = loop.time() + self.settings.market_poll_interval_seconds
        pending = None
        try:
            while not self.exchange.all_markets_settled():
                self.exchange.close_due_markets()
                if pending is not None and pending.done():
                    await pending
                    pending = None
                    next_poll = loop.time() + self.settings.market_poll_interval_seconds
                    if self.exchange.all_markets_settled():
                        return
                if pending is None and loop.time() >= next_poll:
                    pending = asyncio.create_task(self.poll_once())
                delay = None if pending is not None else max(0, next_poll - loop.time())
                cutoff = self.exchange.next_close_time()
                if cutoff is not None:
                    until_close = max(0, cutoff - self._now())
                    delay = until_close if delay is None else min(delay, until_close)
                if pending is None:
                    await asyncio.sleep(delay)
                else:
                    await asyncio.wait({pending}, timeout=delay)
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

# Compatibility for historical stream imports; offline callers use integrations.kalshi.
from ..integrations.kalshi import fetch_trade_history, fetch_market_payloads
