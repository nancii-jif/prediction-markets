"""The SOLE module that talks to Kalshi. Everything else is downstream of the
database this writes.

    python -m prediction_markets.stream                    # refresh_live() (cron)
    python -m prediction_markets.stream --sweep            # force a full sweep first
    python -m prediction_markets.stream --discover-sectors # print the sector vocabulary

Kalshi is queried on demand, not on a poll loop. The routine load is a
targeted fetch of the active live markets — one or two batched requests. Full
candidate sweeps happen only at seed and when a resolution demands a refill.

No price data written here is ever rendered into an agent-visible surface;
`live_markets.py` reads these prices for selection, agents never do.
"""

import argparse
import collections
import logging
import time

from . import config, database, kalshi_client, live_markets

log = logging.getLogger(__name__)

# Fields compared to decide whether a market moved since its last snapshot.
TRACKED_FIELDS = (
    "yes_bid", "yes_ask", "bid_size", "ask_size",
    "last", "volume_24h", "status", "result", "settlement_value",
)


def _to_float(value):
    if value is None or value == "":
        return None
    return float(value)


def _to_ts(value):
    """Kalshi timestamps are RFC3339 strings; store Unix seconds UTC."""
    if not value:
        return None
    from datetime import datetime

    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def snapshot_from_market(market: dict, captured_at: int) -> dict:
    """Map a Kalshi market payload onto a snapshots row.

    Kalshi returns prices and sizes as fixed-point strings ("0.0100"), so
    parsing is deterministic and equality between polls is exact.
    """
    return {
        "ticker": market["ticker"],
        "captured_at": captured_at,
        "yes_bid": _to_float(market.get("yes_bid_dollars")),
        "yes_ask": _to_float(market.get("yes_ask_dollars")),
        "bid_size": _to_float(market.get("yes_bid_size_fp")),
        "ask_size": _to_float(market.get("yes_ask_size_fp")),
        "last": _to_float(market.get("last_price_dollars")),
        "volume_24h": _to_float(market.get("volume_24h_fp")),
        "status": market.get("status"),
        "result": market.get("result") or "",
        # Binary markets can still settle to a partial value (result "scalar"),
        # in which case this carries the per-contract payout.
        "settlement_value": _to_float(market.get("settlement_value_dollars")),
    } 
    # in any case we can retroactively pull historical market data


def market_row(market: dict, sector: str, first_seen: int) -> dict:
    """Map a Kalshi market payload onto a markets row.

    This is the only entry point for text agents will read, so title and rules
    are stored verbatim — nothing is summarised, trimmed or reworded.
    """
    return {
        "ticker": market["ticker"],
        # Stored verbatim rather than derived from the ticker: selection
        # strata by event, so this is load-bearing, and the EVENT-STRIKE
        # ticker convention is only a convention.
        "event_ticker": market.get("event_ticker"),
        "title": market.get("title"),
        "rules": market.get("rules_primary"),
        "sector": sector,
        "open_ts": _to_ts(market.get("open_time")),
        "close_ts": _to_ts(market.get("close_time")),
        "first_seen": first_seen,
    }


def _changed(new: dict, old) -> bool:
    return any(new[field] != old[field] for field in TRACKED_FIELDS)


# --- sectors ---------------------------------------------------------------


def series_ticker(market: dict) -> str:
    """The series a market belongs to.

    Kalshi's ticker convention is SERIES-EVENTSUFFIX-STRIKE, so the series is
    the segment of the event ticker before the first dash. Series tickers
    themselves contain no dash.
    """
    return (market.get("event_ticker") or "").split("-", 1)[0]


def sector_map() -> dict[str, str]:
    """series_ticker -> sector, restricted to the pinned allowlist.

    Markets carry no category field of their own (verified against the live
    API: /markets returns no `category` key), so a market's sector has to be
    resolved through its series. /series takes a server-side category filter
    and answers unpaginated, which makes this three requests rather than the
    ~63 pages that walking open events would cost on every sweep.
    """
    mapping: dict[str, str] = {}
    for sector in config.SECTORS:
        for series in kalshi_client.get_series(category=sector):
            mapping[series["ticker"]] = sector
    return mapping


def discover_sectors() -> collections.Counter:
    """The live sector vocabulary, counted by open event. A build-time tool:
    run once, pin the result into config.SECTORS so membership never silently
    follows an upstream rename. Not used on any routine path."""
    events = kalshi_client.get_events(status="open")
    return collections.Counter((e.get("category") or "") for e in events)


# --- the two entry points --------------------------------------------------


def sweep_candidates() -> tuple[int, int]:
    """Full horizon query: the candidate-pool path. Returns (markets, written).

    Invoked at seed, on refill demand from refresh_live(), and optionally on
    ARCHIVE_SWEEP_INTERVAL_S. Roughly 92s over the full ~25k-market horizon.

    Sector narrowing is client-side, off the event category: there is no
    server-side category filter on /markets, so the horizon is swept whole and
    filtered here. Only markets inside the pinned allowlist are stored — an
    out-of-allowlist market can never be admitted, so its history is of no use
    to either selection or the benchmark.
    """
    now = int(time.time())
    started = time.monotonic()

    sectors = sector_map()
    markets = kalshi_client.get_markets(
        status="open",
        min_close_ts=now + config.EXPIRY_MIN_H * 3600,
        max_close_ts=now + config.EXPIRY_MAX_H * 3600,
    )

    market_rows, snapshot_rows = [], []
    for market in markets:
        sector = sectors.get(series_ticker(market))
        if sector is None:
            continue
        market_rows.append(market_row(market, sector, now))
        snapshot_rows.append(snapshot_from_market(market, now))

    database.upsert_markets(market_rows)
    written = database.write_snapshots(snapshot_rows)

    per_sector = collections.Counter(r["sector"] for r in market_rows)
    log.info(
        "sweep: %d markets in horizon, %d in %d pinned sectors %s, %d snapshots "
        "written in %.1fs",
        len(markets), len(market_rows), len(config.SECTORS), dict(per_sector),
        written, time.monotonic() - started,
    )
    if not market_rows and markets:
        log.error(
            "sweep matched 0 markets to a pinned sector — check that "
            "config.SECTORS still matches the live vocabulary "
            "(python -m prediction_markets.stream --discover-sectors)"
        )
    return len(market_rows), written


def refresh_live() -> dict:
    """The routine path: targeted fetch of the active live markets, then
    membership maintenance. Idempotent.

    Only live members are fetched. Retired markets are never re-fetched: a
    result observed while the market was still live is already durable in
    `snapshots`, so settlement reads it from there however many rounds later
    it runs, and the retirement grace period is what guarantees the result
    had time to arrive before the seat was freed.

    Retirement and refill are driven as two independent processes, in order:

      1. refresh the live members from Kalshi;
      2. retire whatever now qualifies (result, or past close + grace);
      3. if any sector is short a seat, sweep for fresh candidates;
      4. refill from that sweep.

    The sweep is triggered by the deficit itself rather than by predicting
    what step 2 is about to retire, so the retirement criteria live in exactly
    one place. It also means a seat left open by an earlier failure — or a
    sector that has simply never been full — pulls a fresh sweep on the next
    run instead of being refilled from stale candidate data, and seeding needs
    no special case: a cold start is just a deficit of every seat.

    Failure handling is asymmetric on purpose. A targeted-fetch exception must
    not stop retirement running on whatever data is already fresh. A sweep
    exception skips the refill entirely: the seat stays open, and the deficit
    that is still there next run triggers another sweep.
    """
    stats = {"fetched": 0, "written": 0, "retired": [], "added": [],
             "swept": False, "sweep_failed": False}

    tickers = database.active_live_tickers()
    if tickers:
        previous = database.latest_snapshots(tickers)
        try:
            markets = kalshi_client.get_markets_by_tickers(tickers)
        except Exception:
            log.exception("targeted fetch failed; retiring on existing data")
            markets = None

        if markets is not None:
            captured_at = int(time.time())
            rows = [
                row for row in (
                    snapshot_from_market(m, captured_at) for m in markets
                )
                if previous.get(row["ticker"]) is None
                or _changed(row, previous[row["ticker"]])
            ]
            stats["fetched"] = len(markets)
            stats["written"] = database.write_snapshots(rows)

            # Keep close/open times current: clock-based retirement reads
            # close_ts, and Kalshi amends close times (early closes,
            # extensions), so the stored value must track the payload rather
            # than freeze at admission.
            database.update_market_times(
                (_to_ts(m.get("close_time")), _to_ts(m.get("open_time")),
                 m["ticker"])
                for m in markets
            )

    stats["retired"] = live_markets.retire()

    short_by = live_markets.deficit()
    if short_by:
        log.info("%d seat(s) open %s; sweeping for candidates",
                 sum(short_by.values()), dict(short_by))
        try:
            sweep_candidates()
            stats["swept"] = True
        except Exception:
            stats["sweep_failed"] = True
            log.exception(
                "SWEEP FAILED with %d seat(s) open — leaving them open; the "
                "deficit will pull another sweep on the next run",
                sum(short_by.values()),
            )
        else:
            stats["added"] = live_markets.refill()["added"]

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="path to the run config JSON in configs/")
    parser.add_argument("--sweep", action="store_true",
                        help="force a full candidate sweep before refreshing")
    parser.add_argument("--discover-sectors", action="store_true",
                        help="print the live sector vocabulary and exit")
    args = parser.parse_args()

    config.load(args.config)
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(config.RUN_DIR / "stream.log"),
                  logging.StreamHandler()],
    )

    if args.discover_sectors:
        counts = discover_sectors()
        print(f"\n{sum(counts.values())} open events across {len(counts)} sectors\n")
        for sector, n in counts.most_common():
            marker = "  <- pinned" if sector in config.SECTORS else ""
            print(f"{n:6d}  {sector or '(none)'}{marker}")
        raise SystemExit(0)

    database.init()
    if args.sweep:
        sweep_candidates()
    result = refresh_live()
    log.info("refresh_live: %s", result)
