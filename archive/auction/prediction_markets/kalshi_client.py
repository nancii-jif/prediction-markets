"""Sole touchpoint for the Kalshi API.

Public market-data endpoints only — this project never authenticates and never
places a real order. Per the cross-cutting rules, `stream.py` is the SOLE
importer of this module: if data seems to be missing downstream, the fix
belongs in stream, never a direct call from another module.
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from . import config

log = logging.getLogger(__name__)

_USER_AGENT = "prediction-markets-v0 (research paper-trading harness)"
_TIMEOUT_S = 30
_MAX_ATTEMPTS = 5
_INITIAL_BACKOFF_S = 1.0
_RATE_LIMIT_BACKOFF_S = 10.0
_PAGE_LIMIT = 1000  # Kalshi's documented maximum
# Ticker batches are bounded by encoded URL length, not count: ticker names range
# from ~20 to ~45 characters, and a fixed count large enough to be efficient for
# short names returns HTTP 414 for long ones.
_MAX_TICKER_QUERY_CHARS = 6000
# ~30 pages covers the real market universe; anything near this means a runaway
# pagination (typically a universe-wide query missing mve_filter=exclude).
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


def _get(path: str, params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    url = f"{config.KALSHI_BASE_URL}{path}"
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


def _paginate(path: str, key: str, params: dict) -> list[dict]:
    markets: list[dict] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None

    for page in range(1, _MAX_PAGES + 1):
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor

        payload = _get(path, page_params)
        markets.extend(payload.get(key) or [])
        log.debug("page %d: %d %s (running total %d)", page, len(payload.get(key) or []), key, len(markets))

        cursor = payload.get("cursor") or None
        if cursor is None:
            return markets
        # A repeated cursor means the server stopped advancing; stop rather than spin.
        if cursor in seen_cursors:
            log.warning("pagination cursor repeated after %d pages; stopping", page)
            return markets
        seen_cursors.add(cursor)

    raise KalshiError(
        f"pagination did not terminate within {_MAX_PAGES} pages "
        f"({len(markets)} markets); check that mve_filter=exclude was passed"
    )


def get_markets(**params) -> list[dict]:
    """GET /markets, following cursor pagination to exhaustion.

    Defaults to mve_filter=exclude: auto-generated multivariate (parlay) combo
    markets outnumber real markets roughly 6:1 and make universe-wide
    pagination effectively non-terminating.
    """
    params.setdefault("limit", _PAGE_LIMIT)
    params.setdefault("mve_filter", "exclude")
    return _paginate("/markets", "markets", params)


def get_events(**params) -> list[dict]:
    """GET /events, following cursor pagination to exhaustion.

    Markets carry no category of their own — the sector a market belongs to is
    a property of its event — so sector stratification has to join through
    here on event_ticker.
    """
    params.setdefault("limit", 200)  # /events caps lower than /markets
    return _paginate("/events", "events", params)


def get_series(**params) -> list[dict]:
    """GET /series. Supports a server-side `category` filter and returns the
    whole category in one unpaginated response (~2.3k series for Politics).

    This is how sectors are resolved in the routine path: three requests, one
    per pinned category, rather than paginating ~12k open events.
    """
    return _paginate("/series", "series", params)


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


def get_markets_by_tickers(tickers) -> list[dict]:
    """Fetch specific markets by ticker, batched by encoded query length."""
    markets: list[dict] = []
    for batch in _ticker_batches(tickers):
        markets.extend(
            _paginate("/markets", "markets", {"tickers": ",".join(batch), "limit": _PAGE_LIMIT})
        )
    return markets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="connectivity smoke test")
    parser.add_argument("--config", required=True, help="run config JSON")
    args = parser.parse_args()
    config.load(args.config)

    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    now = int(time.time())
    started = time.monotonic()
    markets = get_markets(
        status="open",
        min_close_ts=now,
        max_close_ts=now + config.EXPIRY_MAX_H * 3600,
    )
    elapsed = time.monotonic() - started

    tickers = {m["ticker"] for m in markets}
    mve = {m["ticker"] for m in markets if m["event_ticker"].startswith("KXMVE")}
    statuses = sorted({m["status"] for m in markets})
    print(f"\nopen markets closing within {config.HORIZON_MAX_DAYS}d: {len(markets)}")
    print(f"unique tickers: {len(tickers)}")
    print(f"leaked multivariate markets: {len(mve)}")
    print(f"status values returned: {statuses}")
    print(f"pagination terminated in {elapsed:.1f}s")

    if markets:
        sample = markets[0]
        print(
            "\nsample: "
            + json.dumps(
                {
                    k: sample.get(k)
                    for k in (
                        "ticker", "status", "close_time",
                        "yes_bid_dollars", "yes_ask_dollars",
                        "yes_bid_size_fp", "yes_ask_size_fp",
                        "last_price_dollars", "volume_24h_fp",
                    )
                },
                indent=2,
            )
        )
