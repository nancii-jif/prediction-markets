"""The round loop (cadence B).

    python -m prediction_markets.runner --rounds 3   # three rounds, then exit
    python -m prediction_markets.runner              # run on the round cadence
    python -m prediction_markets.runner --dry-run    # budget check only

The runner never touches Kalshi itself. It calls stream.refresh_live() at the
top of each round so resolution state is fresh at the round boundary whatever
the cron stream's phase happens to be — that call is still stream's code.
"""

import argparse
import asyncio
import logging
import time
import uuid

from . import agents, auction, config, database, display, ledger, stream, tracking

log = logging.getLogger(__name__)


def next_round_id() -> int:
    """The round to run next.

    A round is complete when every active market has an auction row for it. A
    round that crashed part-way is resumed under its own id rather than
    abandoned, so the markets that did clear are not re-cleared and the ones
    that did not still get their auction.
    """
    last = database.last_round_id()
    if last is None:
        return 1
    active = set(database.active_live_tickers())
    if active - database.auction_tickers_in_round(last):
        log.info("round %d is incomplete; resuming it", last)
        return last
    return last + 1


def check_budget(rounds: int | None = None) -> int:
    """Refuse to start a run whose *daily* call rate exceeds the ceiling.

    MAX_DAILY_CALLS is a rate, so the quantity to compare against it is peak
    consumption in any 24h window — not the run total. For a run shorter than a
    day the total is the peak (it cannot spend more than it spends); for a
    longer run the peak is a day's worth of rounds. Comparing a multi-day total
    against a per-day ceiling would refuse runs that never exceed the rate.
    """
    live = len(database.active_live_tickers()) or len(config.SECTORS) * config.MARKETS_PER_SECTOR
    rounds = rounds if rounds is not None else config.ROUNDS
    per_call = len(config.AGENTS) * live
    rounds_per_day = 86_400 // config.ROUND_INTERVAL_S
    daily = per_call * rounds_per_day

    if rounds is None:
        peak, total = daily, None
        log.info(
            "budget: %d agents x %d markets x %d rounds/day = %d calls/day "
            "(open-ended; ceiling %d)",
            len(config.AGENTS), live, rounds_per_day, daily, config.MAX_DAILY_CALLS,
        )
    else:
        total = per_call * rounds
        peak = min(total, daily)
        log.info(
            "budget: %d agents x %d markets x %d rounds = %d calls total over "
            "%.1f days; peak %d calls/day (ceiling %d)",
            len(config.AGENTS), live, rounds, total,
            rounds * config.ROUND_INTERVAL_S / 86_400, peak, config.MAX_DAILY_CALLS,
        )

    if peak > config.MAX_DAILY_CALLS:
        raise SystemExit(
            f"refusing to start: {peak} calls/day exceeds "
            f"MAX_DAILY_CALLS={config.MAX_DAILY_CALLS}"
        )
    return total if total is not None else peak


def _validate_and_record(quotes, round_id: int):
    """Margin-check each agent's whole quote set and write the quote rows.

    Rejection is per agent and total: every market in a rejected set gets a
    row recording the rejection, so the reason survives in the database.
    """
    by_agent: dict[str, list[dict]] = {}
    for quote in quotes:
        by_agent.setdefault(quote["agent"], []).append(quote)

    accepted, rows = [], []
    for agent_id, agent_quotes in sorted(by_agent.items()):
        ok, reason = ledger.validate_quote_set(agent_id, agent_quotes)
        if not ok:
            log.warning("quote set from %s rejected: %s", agent_id, reason)
        for quote in agent_quotes:
            rows.append({
                "round_id": round_id, "agent": agent_id, "ticker": quote["ticker"],
                "bid_cents": quote["bid_cents"], "bid_size": quote["bid_size"],
                "ask_cents": quote["ask_cents"], "ask_size": quote["ask_size"],
                "status": "accepted" if ok else "rejected",
                "reject_reason": None if ok else reason,
            })
        if ok:
            accepted.extend(agent_quotes)

    database.insert_quotes(rows)
    # Rows, not just the accepted quotes: the display shows rejections and the
    # reason alongside the quotes that stood.
    return accepted, rows


def run_round(round_id: int) -> dict:
    """One auction round, end to end."""
    started = time.monotonic()
    print(display.round_header(round_id), flush=True)

    # 1. Fresh resolution state (and any sweep/refill it demands).
    stream.refresh_live()

    # 2. Settle anything that resolved, and close out any ghost market that
    #    was retired on the clock without ever producing a result.
    settled = ledger.settle(round_id)
    voided = ledger.void(round_id)

    # 3. Only markets that are still live and unresolved get quoted.
    tickers = database.active_live_tickers()
    snapshots = database.latest_snapshots(tickers)
    tickers = [
        t for t in tickers
        if not (snapshots.get(t) and snapshots[t]["result"])
    ]
    already = database.auction_tickers_in_round(round_id)
    pending = [t for t in tickers if t not in already]
    print(display.live_markets(pending), flush=True)
    print(display.settlements(settled + voided), flush=True)
    if not pending:
        log.info("round %d: nothing to auction", round_id)
        return {"round_id": round_id, "settled": len(settled),
                "voided": len(voided), "auctions": 0}

    roster = agents.build_roster()
    ledger.register_agents(a.agent_id for a in roster)

    # 4. Every agent quotes every pending market, concurrently.
    quotes = asyncio.run(agents.gather_quotes(roster, pending, round_id))
    log.info("round %d: %d quotes from %d agents on %d markets",
             round_id, len(quotes), len(roster), len(pending))

    # 5. Margin validation, then quote rows.
    accepted, all_rows = _validate_and_record(quotes, round_id)

    # 6. Clear each market and commit the result with its ledger effects.
    by_ticker: dict[str, list[dict]] = {}
    for quote in accepted:
        by_ticker.setdefault(quote["ticker"], []).append(quote)
    shown: dict[str, list[dict]] = {}
    for row in all_rows:
        shown.setdefault(row["ticker"], []).append(row)

    now = int(time.time())
    printed = 0
    cleared: list[tuple] = []
    for ticker in pending:
        market_quotes = sorted(by_ticker.get(ticker, []), key=lambda q: q["agent"])
        result = auction.clear(market_quotes, database.last_clear_cents(ticker))

        fills = [
            (f.agent, ticker, f.signed_qty, f.price_cents) for f in result.fills
        ]
        cash_rows, position_rows = ledger.project_fills(fills)

        written = database.record_auction(
            {"ticker": ticker, "round_id": round_id, "ts": now,
             "clear_cents": result.clear_cents, "cleared_qty": result.cleared_qty,
             "tie_lo": result.tie_lo, "tie_hi": result.tie_hi,
             "indicative_cents": result.indicative_cents},
            [{"round_id": round_id, "agent": f.agent, "ticker": ticker,
              "signed_qty": f.signed_qty, "price_cents": f.price_cents}
             for f in result.fills],
            cash_rows, position_rows,
        )
        if not written:
            log.info("%s already cleared in round %d; skipping", ticker, round_id)
            continue

        printed += 1
        cleared.append((ticker, result, market_quotes))
        print(
            display.market_round(ticker, shown.get(ticker, []), result),
            flush=True,
        )

    print(display.pnl_table([a.agent_id for a in roster], round_id), flush=True)
    log.info("round %d done in %.1fs", round_id, time.monotonic() - started)
    stats = {
        "round_id": round_id, "settled": len(settled), "voided": len(voided),
        "auctions": printed, "quotes": len(quotes), "accepted": len(accepted),
        "seconds": round(time.monotonic() - started, 1),
    }
    # After the print, so a tracking hiccup can never cost the operator the
    # round they just watched happen.
    tracking.log_round(round_id, cleared, [a.agent_id for a in roster], stats)
    return stats


def run(rounds: int | None = None) -> None:
    rounds = rounds if rounds is not None else config.ROUNDS
    database.init()
    check_budget(rounds)
    config.freeze()
    database.log_run_config(str(uuid.uuid4()), int(time.time()))
    tracking.start()

    completed = 0
    next_t = time.monotonic()
    try:
        while rounds is None or completed < rounds:
            try:
                run_round(next_round_id())
            except Exception:
                log.exception("round failed; continuing to the next one")
            completed += 1

            if rounds is not None and completed >= rounds:
                break
            next_t += config.ROUND_INTERVAL_S
            sleep_for = next_t - time.monotonic()
            if sleep_for < 0:
                log.warning("round overran the %ds interval by %.1fs; realigning",
                            config.ROUND_INTERVAL_S, -sleep_for)
                next_t = time.monotonic()
                sleep_for = 0
            time.sleep(sleep_for)
    finally:
        # Also on Ctrl-C: an unfinished W&B run leaves the dashboard stuck
        # mid-stream with no indication the process is gone.
        tracking.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True,
                        help="path to the run config JSON in configs/")
    parser.add_argument("--rounds", type=int, default=None,
                        help="override the config's ROUNDS")
    parser.add_argument("--dry-run", action="store_true",
                        help="log the call budget and exit without calling anything")
    args = parser.parse_args()

    config.load(args.config)
    config.RUN_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(config.RUN_DIR / "runner.log"),
                  logging.StreamHandler()],
    )
    log.info("run %s — %s", config.RUN_NAME, config.DESCRIPTION)

    database.init()
    if args.dry_run:
        check_budget(args.rounds)
        raise SystemExit(0)
    run(rounds=args.rounds)
