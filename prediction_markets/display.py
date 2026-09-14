"""Live console rendering of a run: markets, quotes, clears, PnL.

Presentation only. Every function takes data and returns lines; nothing here
reads Kalshi, and nothing it prints is visible to an agent — this is the
operator's view, which is exactly why it may show things a brief never would.

A market block is a price view and nothing else: the quotes that were posted,
where the book stood, and what it cleared at. Per-agent fills are in the `fills`
table for analysis and deliberately not printed here.
"""

import time

from . import auction, config, database

WIDTH = 78
LABEL = 10  # width of the leading label column inside a market block


def _rule(char: str = "─") -> str:
    return char * WIDTH


def _row(label: str, text: str) -> str:
    return f"    {label:<{LABEL}}{text}"


def _frame(price_cents):
    """Present a price in the run's quote frame (YES cents, or 100-x for NO)."""
    if price_cents is None:
        return None
    return price_cents if config.QUOTE_SIDE == "yes" else 100 - price_cents


def _duration(seconds: int) -> str:
    if seconds <= 0:
        return "closed"
    hours, minutes = divmod(int(seconds) // 60, 60)
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h" if days else f"{hours}h{minutes:02d}m"


def round_header(round_id: int) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    title = f"  Round {round_id}"
    return "\n".join([
        "", _rule(), f"{title}{stamp.rjust(WIDTH - len(title))}", _rule(),
    ])


def live_markets(tickers) -> str:
    """The roster being auctioned this round."""
    now = int(time.time())
    lines = ["", f"  Live markets ({len(tickers)})"]
    for ticker in tickers:
        market = database.market(ticker)
        if market is None:
            lines.append(f"    {ticker}")
            continue
        closes = _duration(market["close_ts"] - now) if market["close_ts"] else "?"
        title = (market["title"] or "")[:44]
        lines.append(
            f"    {(market['sector'] or '')[:11]:<11} {ticker[:30]:<30} "
            f"{title:<44} {closes:>7}"
        )
    return "\n".join(lines)


def settlements(rows) -> str:
    if not rows:
        return ""
    lines = ["", f"  Settlements ({len(rows)})"]
    for row in rows:
        tag = "  void" if row.get("void") else ""
        lines.append(
            f"    {row['agent']:<10} {row['ticker'][:34]:<34} "
            f"qty {row['qty']:+4d}  payout {row['payout_cents']:>7,}c{tag}"
        )
    return "\n".join(lines)


def _quote_cell(quote) -> str:
    bid, ask = _frame(quote["bid_cents"]), _frame(quote["ask_cents"])
    bid_size, ask_size = quote["bid_size"], quote["ask_size"]
    if config.QUOTE_SIDE == "no":
        bid, ask = ask, bid
        bid_size, ask_size = ask_size, bid_size
    return f"{quote['agent']:<9}{bid:>3}c x{bid_size:<3}/{ask:>3}c x{ask_size:<3}"


def _book_line(quotes) -> str | None:
    """Where the book stood before clearing: the two best prices and their size."""
    top = auction.top_of_book(quotes)
    if top is None:
        return None
    bid, bid_size = _frame(top["bid_cents"]), top["bid_size"]
    ask, ask_size = _frame(top["ask_cents"]), top["ask_size"]
    if config.QUOTE_SIDE == "no":
        bid, ask = ask, bid
        bid_size, ask_size = ask_size, bid_size
    parts = [
        f"best bid {bid}c x{bid_size}" if bid is not None else "no bid",
        f"best ask {ask}c x{ask_size}" if ask is not None else "no ask",
    ]
    return "   ".join(parts)


def _unfilled(quotes, clear_cents: int) -> str:
    excess = auction.imbalance(quotes, clear_cents)
    if excess == 0:
        return "unfilled 0"
    side = "bid" if (excess > 0) == (config.QUOTE_SIDE == "yes") else "ask"
    return f"unfilled {side} x{abs(excess)}"


def market_round(ticker, quotes, result) -> str:
    """One market: the quotes posted, the book they made, and the clear."""
    market = database.market(ticker)
    title = (market["title"] if market else ticker) or ticker

    lines = ["", f"  {ticker}", f"  {title[:WIDTH - 2]}", ""]

    accepted = [q for q in quotes if q.get("status", "accepted") == "accepted"]
    rejected = [q for q in quotes if q.get("status") == "rejected"]

    cells = [_quote_cell(q) for q in sorted(accepted, key=lambda q: q["agent"])]
    body = ["   ".join(cells[i:i + 2]).rstrip() for i in range(0, len(cells), 2)]
    body += [f"{q['agent']:<12}rejected — {q['reject_reason']}"
             for q in sorted(rejected, key=lambda q: q["agent"])]
    if not body:
        body = ["(none)"]
    lines.append(_row("Quotes", body[0]))
    lines += [_row("", line) for line in body[1:]]

    book = _book_line(accepted)
    if book:
        lines.append(_row("Book", book))

    if result is None or result.clear_cents is None:
        lines.append(_row("Clear", "no trade"))
        return "\n".join(lines)

    lines.append(_row("Clear", (
        f"{_frame(result.clear_cents)}c   volume x{result.cleared_qty}   "
        f"{_unfilled(accepted, result.clear_cents)}"
    )))
    return "\n".join(lines)


def pnl_table(agent_ids, round_id: int) -> str:
    """Standings after a round. Equity is cash plus open positions at last clear."""
    lines = [
        "", _rule(), f"  Standings after round {round_id}", _rule(),
        f"  {'agent':<10} {'cash':>10} {'open':>6} {'realised':>10} "
        f"{'unreal':>9} {'equity':>11}",
    ]
    rows = []
    for agent_id in agent_ids:
        account = ledger_pnl(agent_id)
        rows.append((account["equity_cents"], agent_id, account))
    for _, agent_id, account in sorted(rows, reverse=True):
        lines.append(
            f"  {agent_id:<10} {account['cash_cents']:>10,} "
            f"{account['open_positions']:>6} {account['realized_cents']:>+10,} "
            f"{account['unrealized_cents']:>+9,} {account['equity_cents']:>11,}"
        )
    return "\n".join(lines)


def ledger_pnl(agent_id: str) -> dict:
    """PnL plus the two figures the ledger does not report: open-position count
    and equity.

    Equity is the starting bankroll plus total PnL. That identity holds because
    realised is defined as cash + cost basis - initial, so
    initial + realised + unrealised collapses to cash + positions at mark.
    """
    from . import ledger

    account = dict(ledger.pnl(agent_id))
    account["open_positions"] = len(database.open_positions(agent_id))
    account["equity_cents"] = config.INITIAL_CASH_CENTS + account["total_cents"]
    return account
