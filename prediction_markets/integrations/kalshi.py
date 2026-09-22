"""Sole touchpoint for the Kalshi API.

Public market-data endpoints only; this project never places a real order.
Market selection/updates and offline analysis use this adapter. Agent tools
and the exchange never query human prices through it.
"""

import json
import logging
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

from .. import config

log = logging.getLogger(__name__)

_USER_AGENT = "prediction-markets-mvp (research paper-trading harness)"
_TIMEOUT_S = 30
_MAX_ATTEMPTS = 5
_INITIAL_BACKOFF_S = 1.0
_RATE_LIMIT_BACKOFF_S = 10.0
_PAGE_LIMIT = 1000  # Kalshi's documented maximum
_EVENT_PAGE_LIMIT = 200
# Ticker batches are bounded by encoded URL length, not count: ticker names range
# from ~20 to ~45 characters, and a fixed count large enough to be efficient for
# short names returns HTTP 414 for long ones.
_MAX_TICKER_QUERY_CHARS = 6000
# Guard against runaway pagination; /events excludes multivariate combos.
_MAX_PAGES = 200


class KalshiError(RuntimeError):
    """A request could not be completed, or pagination failed to terminate."""


def _retry_after(exc: urllib.error.HTTPError, fallback: float) -> float:
    header = exc.headers.get("Retry-After") if exc.headers else None
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return fallback


def _get(path: str, params: dict, *, base_url: str | None = None) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{(base_url or config.current().kalshi_base_url).rstrip('/')}{path}"
    if query:
        url = f"{url}?{query}"

    backoff = _INITIAL_BACKOFF_S
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(request, timeout=_TIMEOUT_S) as response:
                return json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                wait = _retry_after(exc, _RATE_LIMIT_BACKOFF_S * attempt)
            elif exc.code >= 500:
                wait = backoff
            else:
                raise KalshiError(f"GET {url} -> HTTP {exc.code} {exc.reason}") from exc
            last_exc = exc
        except (urllib.error.URLError, TimeoutError) as exc:
            wait = backoff
            last_exc = exc

        if attempt == _MAX_ATTEMPTS:
            break
        log.warning(
            "GET %s failed (%s), attempt %d/%d, retrying in %.1fs",
            path, last_exc, attempt, _MAX_ATTEMPTS, wait,
        )
        time.sleep(wait)
        backoff *= 2

    raise KalshiError(f"GET {url} failed after {_MAX_ATTEMPTS} attempts") from last_exc


def _paginate(
    path: str, key: str, params: dict, *, base_url: str | None = None,
) -> list[dict]:
    markets: list[dict] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None

    for page in range(1, _MAX_PAGES + 1):
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor

        payload = _get(path, page_params, base_url=base_url)
        if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
            raise KalshiError(f"GET {path} returned a malformed {key} page")
        markets.extend(payload[key])
        log.debug("page %d: %d %s (running total %d)", page, len(payload[key]), key, len(markets))

        cursor = payload.get("cursor") or None
        if cursor is None:
            return markets
        if not isinstance(cursor, str):
            raise KalshiError(f"GET {path} returned a malformed cursor")
        # A repeated cursor means the server stopped advancing; stop rather than spin.
        if cursor in seen_cursors:
            raise KalshiError(f"pagination cursor repeated after {page} pages")
        seen_cursors.add(cursor)

    raise KalshiError(
        f"pagination did not terminate within {_MAX_PAGES} pages "
        f"({len(markets)} {key} from {path})"
    )


def get_market_candidates(*, base_url: str | None = None) -> list[dict]:
    """Discover open events with nested markets and current series categories.

    Event categories can be stale; join by the explicit series_ticker instead
    of guessing from market titles or ticker prefixes. /events excludes
    multivariate combos. Category metadata is fetched only at startup.
    """
    series_categories = {}
    for series in _paginate("/series", "series", {}, base_url=base_url):
        if not isinstance(series, dict) or not isinstance(series.get("ticker"), str):
            raise KalshiError("GET /series returned a malformed series")
        ticker = series["ticker"]
        if ticker in series_categories:
            raise KalshiError(f"GET /series returned duplicate series {ticker}")
        categories = series.get("categories")
        if categories is None:
            categories = [series.get("category")]
        elif isinstance(categories, list):
            # Keep Kalshi's primary category, first: a secondary allowed
            # category must never admit a series whose primary category is
            # Sports or Crypto, and selection buckets on this leading entry.
            categories = [series.get("category"), *categories]
        else:
            categories = []  # Malformed metadata must fail the allowlist.
        series_categories[ticker] = categories

    events = _paginate(
        "/events", "events",
        {"status": "open", "limit": _EVENT_PAGE_LIMIT, "with_nested_markets": "true"},
        base_url=base_url,
    )
    candidates = []
    for event in events:
        if (
            not isinstance(event, dict)
            or not isinstance(event.get("event_ticker"), str)
            or not isinstance(event.get("series_ticker"), str)
            or not isinstance(event.get("markets"), list)
        ):
            raise KalshiError("GET /events returned a malformed nested event")
        categories = series_categories.get(event["series_ticker"], [])
        for market in event["markets"]:
            if not isinstance(market, dict) or market.get("event_ticker") != event["event_ticker"]:
                raise KalshiError("GET /events returned a malformed or mismatched nested market")
            # Keep the exact market object separately from selection-only
            # category enrichment so research exports preserve API fields.
            candidates.append({**market, "categories": categories, "_raw_market": market})
    return candidates


def _ticker_batches(tickers):
    """Group tickers so each batch's encoded query stays under the URI limit."""
    batch: list[str] = []
    length = 0
    for ticker in tickers:
        encoded = len(urllib.parse.quote(ticker, safe="")) + len("%2C")
        if batch and length + encoded > _MAX_TICKER_QUERY_CHARS:
            yield batch
            batch, length = [], 0
        batch.append(ticker)
        length += encoded
    if batch:
        yield batch


def get_markets_by_tickers(tickers, *, base_url: str | None = None) -> list[dict]:
    """Fetch specific markets by ticker, batched by encoded query length."""
    markets: list[dict] = []
    for batch in _ticker_batches(tickers):
        markets.extend(
            _paginate(
                "/markets", "markets", {"tickers": ",".join(batch), "limit": _PAGE_LIMIT},
                base_url=base_url,
            )
        )
    return markets


def get_trade_history(ticker: str, start_ts: int, end_ts: int, *, base_url: str) -> list[dict]:
    """Public YES-price history for offline analysis; inclusive time bounds.

    Route by trade creation cutoff, not market settlement cutoff. Overlap the
    API's second-resolution boundaries and let the caller perform precise as-of
    matching. Never assume the API's page order is chronological.
    """
    cutoff_payload = _get("/historical/cutoff", {}, base_url=base_url)
    try:
        value = cutoff_payload["trades_created_ts"]
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError("missing timezone")
        cutoff = date.timestamp()
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise KalshiError("Kalshi returned a malformed trade-history cutoff") from exc
    ranges = []
    if start_ts < cutoff:
        ranges.append(("/historical/trades", start_ts, min(end_ts, math.ceil(cutoff))))
    if end_ts >= cutoff:
        ranges.append(("/markets/trades", max(start_ts, math.floor(cutoff)), end_ts))
    trades = {}
    for path, lower, upper in ranges:
        for trade in _paginate(path, "trades", {
            "ticker": ticker, "min_ts": lower - 1, "max_ts": upper + 1, "limit": _PAGE_LIMIT,
        }, base_url=base_url):
            if not isinstance(trade, dict) or trade.get("ticker") != ticker or not trade.get("trade_id"):
                raise KalshiError(f"GET {path} returned a malformed or mismatched trade")
            trades[trade["trade_id"]] = trade
    return list(trades.values())


def fetch_trade_history(ticker: str, start_ts: int, end_ts: int, *, base_url: str) -> list[dict]:
    """Offline plot data only; never written to the exchange or participant state."""
    try:
        return get_trade_history(ticker, start_ts, end_ts, base_url=base_url)
    except (KalshiError, OSError) as exc:
        raise ValueError(f"could not fetch Kalshi trade history for {ticker}: {exc}") from exc


def fetch_market_payloads(tickers: list[str], *, base_url: str) -> list[dict]:
    """Explicit export-time backfill for runs lacking original API objects.

    Return unmodified market objects; callers must label them as current data,
    never as the run's original observation. No exchange or DB is mutated.
    """
    try:
        payloads = get_markets_by_tickers(tickers, base_url=base_url)
    except (KalshiError, OSError) as exc:
        raise ValueError(f"could not fetch missing Kalshi market payloads: {exc}") from exc
    wanted, seen = set(tickers), set()
    for payload in payloads:
        if not isinstance(payload, dict) or not isinstance(payload.get("ticker"), str):
            raise ValueError("Kalshi returned a malformed market payload")
        ticker = payload["ticker"]
        if ticker not in wanted or ticker in seen:
            raise ValueError(f"Kalshi returned an unexpected or duplicate market: {ticker}")
        seen.add(ticker)
    return payloads
