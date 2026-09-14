"""Live run metrics, to Weights & Biases when configured and nowhere otherwise.

Operator-side, like `display`. This is the one surface where the internal price
and the Kalshi price appear side by side — which is the comparison the whole run
exists to make, and exactly why nothing here may ever be read back into a brief.

Everything is logged in YES space regardless of `QUOTE_SIDE`, so a yes-run and a
no-run on matched markets are directly comparable; the frame is recorded in the
W&B config instead of being baked into the numbers.

Tracking never breaks a run. A missing package, a bad key, a network failure —
all of it degrades to logging nothing and letting the round proceed.
"""

import logging

from . import auction, config, database, display

log = logging.getLogger(__name__)

_run = None
_failed = False


def enabled() -> bool:
    return _run is not None


def start() -> bool:
    """Open a W&B run. Returns False when tracking is off or unavailable."""
    global _run, _failed

    if _run is not None or _failed:
        return _run is not None
    if not config.WANDB_PROJECT:
        return False

    try:
        import wandb

        _run = wandb.init(
            project=config.WANDB_PROJECT,
            name=config.RUN_NAME,
            config=config.snapshot(),
            dir=str(config.RUN_DIR),
        )
        # Plot everything against the round, not W&B's implicit global step, so
        # panels are labelled "round" and a market seated at round 12 starts its
        # line at 12 rather than at the first step it happened to be logged in.
        _run.define_metric("round")
        _run.define_metric("*", step_metric="round")
    except Exception as exc:
        # A dashboard is not worth a run. Warn once and carry on blind.
        _failed = True
        log.warning("W&B tracking unavailable (%s); continuing without it", exc)
        return False

    log.info("W&B tracking live: %s", getattr(_run, "url", config.WANDB_PROJECT))
    return True


def finish() -> None:
    global _run
    if _run is None:
        return
    try:
        _run.finish()
    except Exception:
        log.warning("W&B finish failed", exc_info=True)
    _run = None


def _mid_cents(snapshot) -> int | None:
    """A Kalshi snapshot's mid, in cents.

    Snapshot prices are dollars in [0, 1]; the internal grid is whole cents, so
    the two are only comparable after this conversion. Returns None when either
    side of the Kalshi book is missing, rather than inventing a one-sided mid.
    """
    if snapshot is None:
        return None
    bid, ask = snapshot["yes_bid"], snapshot["yes_ask"]
    if bid is None or ask is None:
        return None
    return round((bid + ask) / 2 * 100)


def kalshi_mid_cents(ticker: str) -> int | None:
    """The Kalshi mid right now."""
    return _mid_cents(database.latest_snapshot(ticker))


def _depth(result, quotes) -> tuple[int | None, int | None]:
    """The size standing on each side of the price, in contracts.

    Two different questions, because a round that traded and a round that did
    not are asking different things:

      traded   — everything that crossed: all size bid at or above the clearing
                 price, and all size offered at or below it. The gap between
                 the two is the side that went unfilled.
      no trade — the size at the best bid and the best ask, since nothing
                 crossed and those two levels are the whole story.

    Summed across agents either way, so a point never identifies a quote.
    """
    if result.clear_cents is not None:
        return (auction.demand(quotes, result.clear_cents),
                auction.supply(quotes, result.clear_cents))

    top = auction.top_of_book(quotes)
    if top is None:
        return None, None
    return top["bid_size"], top["ask_size"]


def round_metrics(round_id: int, markets, agent_ids, stats=None) -> dict:
    """Build the flat metric dict for one round.

    `markets` is a sequence of (ticker, auction.Result, accepted_quotes).
    Separated from the logging call so the payload can be tested without wandb
    installed and without a network.

    Three groups, matching the three questions the dashboard answers:

      market/<ticker>/{price,kalshi,bid_size,ask_size}
      agent/<id>/*       who is winning
      round/*            whether the run itself is healthy

    Four series per market and no more. W&B draws one panel per metric key, so
    every key added here is another panel to scroll past on a run with fifteen
    live markets.
    """
    # The x-axis for every other series (see define_metric in start()).
    metrics: dict[str, float | int] = {"round": round_id}

    for ticker, result, quotes in markets:
        prefix = f"market/{ticker}"
        clear = result.clear_cents

        # One continuous price series per market: the clearing price where the
        # round traded, the indicative microprice where it did not, so the
        # market draws an unbroken line across rounds that never printed.
        price = clear if clear is not None else result.indicative_cents
        if price is not None:
            metrics[f"{prefix}/price"] = price

        bid_size, ask_size = _depth(result, quotes)
        if bid_size is not None:
            metrics[f"{prefix}/bid_size"] = bid_size
            metrics[f"{prefix}/ask_size"] = ask_size

        kalshi = kalshi_mid_cents(ticker)
        if kalshi is not None:
            metrics[f"{prefix}/kalshi"] = kalshi

    for agent_id in agent_ids:
        account = display.ledger_pnl(agent_id)
        metrics[f"agent/{agent_id}/equity"] = account["equity_cents"]
        metrics[f"agent/{agent_id}/cash"] = account["cash_cents"]
        metrics[f"agent/{agent_id}/realised"] = account["realized_cents"]
        metrics[f"agent/{agent_id}/unrealised"] = account["unrealized_cents"]
        metrics[f"agent/{agent_id}/open"] = account["open_positions"]

    for key, value in (stats or {}).items():
        if key != "round_id" and isinstance(value, (int, float)):
            metrics[f"round/{key}"] = value

    return metrics


def log_round(round_id: int, markets, agent_ids, stats=None) -> dict:
    """Send one round's metrics. Returns the payload, logged or not."""
    metrics = round_metrics(round_id, markets, agent_ids, stats)
    if _run is not None:
        try:
            _run.log(metrics, step=round_id)
        except Exception:
            log.warning("W&B log failed for round %d", round_id, exc_info=True)
    return metrics
