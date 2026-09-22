"""Cash, signed YES positions, settlement.

Everything here is in integer cents and integer contracts, always in YES
space: a fill on an agent's own bid is +qty, a fill on its own ask is -qty,
and selling without inventory is a short YES position, which is the same
thing as being long NO.

Shorts are uncollateralised: an agent may sell what it does not own with no
cash backing the obligation, and both cash and equity may go negative. The
exchange is a price-discovery instrument here, not a risk system, and a margin
rule silences exactly the agents whose views are strongest.

This is bookkeeping for the brief, not a scoring pipeline. Nothing in here
computes a benchmark or compares agents.
"""

import logging

from . import config, database

log = logging.getLogger(__name__)

CONTRACT_VALUE_CENTS = 100


def register_agents(agent_ids) -> None:
    for agent_id in agent_ids:
        if database.register_agent(agent_id, config.INITIAL_CASH_CENTS):
            log.info("registered %s with %d cents", agent_id, config.INITIAL_CASH_CENTS)


def validate_quote_set(agent: str, quotes) -> tuple[bool, str | None]:
    """Admit or reject an agent's whole set of quotes for a round.

    Two checks only: the quote is well formed on the price grid, and it does
    not breach the position cap. There is no solvency test — see the module
    docstring on uncollateralised shorts.

    A violation rejects the entire set — no partial repair, no dropping the
    offending market. Repairing part of a set would silently change the quote
    the agent actually chose to make.
    """
    for quote in quotes:
        ticker = quote["ticker"]
        bid, ask = quote["bid_cents"], quote["ask_cents"]
        bid_size, ask_size = quote["bid_size"], quote["ask_size"]

        if not (config.PRICE_MIN <= bid < ask <= config.PRICE_MAX):
            return False, (
                f"{ticker}: prices must satisfy {config.PRICE_MIN} <= bid < ask "
                f"<= {config.PRICE_MAX}, got bid={bid} ask={ask}"
            )
        for label, size in (("bid_size", bid_size), ("ask_size", ask_size)):
            if not (1 <= size <= config.MAX_LOT):
                return False, (
                    f"{ticker}: {label} must be 1..{config.MAX_LOT}, got {size}"
                )

        if config.MAX_POSITION is None:
            continue  # no cap configured; any resulting position is admissible

        held, _ = database.position(agent, ticker)
        # Pro-rata allocation can only ever fill less than a quote, so checking
        # the full-fill extremes here is enough to make the cap unbreachable.
        for label, resulting in (("bid", held + bid_size), ("ask", held - ask_size)):
            if abs(resulting) > config.MAX_POSITION:
                return False, (
                    f"{ticker}: a full {label} fill would take the position to "
                    f"{resulting}, past the +/-{config.MAX_POSITION} cap"
                )

    return True, None


def _position_after(held: int, basis: int, signed_qty: int, price_cents: int):
    after = held + signed_qty
    if held == 0 or (held > 0) == (signed_qty > 0):
        basis_after = basis + signed_qty * price_cents  # opening or adding
    elif (after > 0) == (held > 0) and after != 0:
        basis_after = round(basis * after / held)  # reducing, same side
    else:
        basis_after = after * price_cents  # closed out, or flipped through zero
    return after, basis_after


def project_fills(fills) -> tuple[dict, dict]:
    """Work out the cash and position rows a set of fills implies, without
    writing anything.

    Separating the arithmetic from the write lets the runner commit an
    auction, its fills and their ledger effects in one transaction, so a
    process killed mid-round can never leave a printed auction whose money was
    never moved.

    `fills` is a sequence of (agent, ticker, signed_qty, price_cents).
    """
    cash: dict[str, int] = {}
    positions: dict[tuple[str, str], tuple[int, int]] = {}

    for agent, ticker, signed_qty, price_cents in fills:
        if signed_qty == 0:
            continue
        balance = cash.get(agent, database.cash_cents(agent))
        cash[agent] = balance - signed_qty * price_cents

        key = (agent, ticker)
        held, basis = positions.get(key, database.position(agent, ticker))
        positions[key] = _position_after(held, basis, signed_qty, price_cents)

    return cash, positions


def apply_fill(agent: str, ticker: str, signed_qty: int, price_cents: int) -> None:
    """Book one fill: cash moves against the trade, the position moves with it."""
    if signed_qty == 0:
        return
    cash, positions = project_fills([(agent, ticker, signed_qty, price_cents)])
    for who, balance in cash.items():
        database.set_cash_cents(who, balance)
    for (who, what), (qty, basis) in positions.items():
        database.set_position(who, what, qty, basis)


def settle(round_id: int) -> list[dict]:
    """Realise every open position in a market Kalshi has resolved.

    Idempotent: the settlements primary key (agent, ticker, round_id) means a
    rerun of a crashed round re-books nothing.
    """
    settled = []
    tickers = database.held_tickers()
    if not tickers:
        return settled

    snapshots = database.latest_snapshots(tickers)
    for ticker in tickers:
        snapshot = snapshots.get(ticker)
        if snapshot is None or not snapshot["result"]:
            continue
        result = snapshot["result"]

        if result == "yes":
            per_contract, is_scalar = CONTRACT_VALUE_CENTS, 0
        elif result == "no":
            per_contract, is_scalar = 0, 0
        else:
            # "scalar": a binary market that settled to a partial value, about
            # 2.6% of resolutions. Never pay zero on a missing value — leave
            # the position open and settle it once the value arrives.
            if snapshot["settlement_value"] is None:
                log.warning(
                    "%s resolved '%s' with no settlement_value; leaving open",
                    ticker, result,
                )
                continue
            per_contract = round(snapshot["settlement_value"] * 100)
            is_scalar = 1

        for row in database.positions_in(ticker):
            agent, qty = row["agent"], row["signed_qty"]
            payout = per_contract * qty
            if not database.insert_settlement(
                agent, ticker, round_id, qty, payout, is_scalar
            ):
                continue  # already settled in this round
            database.set_cash_cents(agent, database.cash_cents(agent) + payout)
            database.set_position(agent, ticker, 0, 0)
            settled.append(
                {"agent": agent, "ticker": ticker, "qty": qty,
                 "payout_cents": payout, "result": result}
            )
            log.info(
                "SETTLE %-40s %-12s qty=%+d result=%s payout=%+dc",
                ticker, agent, qty, result, payout,
            )
    return settled


def void(round_id: int) -> list[dict]:
    """Close out positions in ghost markets at the last internal clear.

    A market retired on the clock without ever producing a result is not
    going to produce one — it was deactivated, or otherwise abandoned. Its
    positions cannot be settled, and because retired markets are no longer
    fetched, waiting would strand them forever.

    They are closed at the last price the internal exchange itself printed —
    never a Kalshi price, and the same mark agents were already shown as
    their unrealised PnL, so voiding just makes that mark final. Positions in
    a market always net to zero across agents, so paying everyone at one
    uniform price moves no net cash: `sum(price * qty) = price * 0`. Zeroing
    the positions and moving no cash at all would instead be arithmetically
    identical to resolving every ghost market NO, which would systematically
    pay the short side.
    """
    voided = []
    tickers = database.held_tickers()
    if not tickers:
        return voided

    live = set(database.active_live_tickers())
    snapshots = database.latest_snapshots(tickers)

    for ticker in tickers:
        if ticker in live:
            continue  # still trading
        snapshot = snapshots.get(ticker)
        if snapshot is not None and snapshot["result"]:
            continue  # has a result; settle() handles it

        price = database.last_clear_cents(ticker)
        if price is None:
            # Unreachable in practice: a position requires a fill, and a fill
            # requires a print. Never invent a price to close on.
            log.error("%s has positions but never printed; leaving open", ticker)
            continue

        for row in database.positions_in(ticker):
            agent, qty = row["agent"], row["signed_qty"]
            payout = price * qty
            if not database.insert_settlement(
                agent, ticker, round_id, qty, payout, scalar=0, void=1
            ):
                continue  # already voided in this round
            database.set_cash_cents(agent, database.cash_cents(agent) + payout)
            database.set_position(agent, ticker, 0, 0)
            voided.append(
                {"agent": agent, "ticker": ticker, "qty": qty,
                 "payout_cents": payout, "price_cents": price}
            )
            log.warning(
                "VOID   %-40s %-12s qty=%+d closed at last clear %dc (no result)",
                ticker, agent, qty, price,
            )
    return voided


def mark_cents(ticker: str) -> int | None:
    """The mark for an open position: the last internal clear, never Kalshi."""
    return database.last_clear_cents(ticker)


def pnl(agent: str) -> dict:
    """Realised and unrealised PnL, both in cents.

    Unrealised marks open positions at the last price the internal exchange
    actually printed. A market that has never printed has no mark, so its
    position is carried at cost and contributes nothing unrealised.
    """
    cash = database.cash_cents(agent)
    positions = database.open_positions(agent)

    basis_total = 0
    unrealized = 0
    marks = {}
    for row in positions:
        basis_total += row["cost_basis_cents"]
        mark = mark_cents(row["ticker"])
        marks[row["ticker"]] = mark
        if mark is not None:
            unrealized += mark * row["signed_qty"] - row["cost_basis_cents"]

    realized = cash + basis_total - config.INITIAL_CASH_CENTS
    return {
        "cash_cents": cash,
        "realized_cents": realized,
        "unrealized_cents": unrealized,
        "total_cents": realized + unrealized,
        "marks": marks,
    }
