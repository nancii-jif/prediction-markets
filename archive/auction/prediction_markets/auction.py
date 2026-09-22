"""Call-auction clearing. Pure functions — no database, no clock, no config
beyond the price grid. Replaying the same quotes reproduces the same fills.

Standard end-of-day auction rule: pick the price that maximises executable
volume, break ties toward the last internal clear, allocate by price priority
with pro-rata at the marginal level. There is no external reference price
anywhere in here — the internal exchange never sees a Kalshi number.
"""

from dataclasses import dataclass, field

from . import config


@dataclass(frozen=True)
class Fill:
    agent: str
    signed_qty: int  # + on a filled bid (long YES), - on a filled ask
    price_cents: int


@dataclass(frozen=True)
class Result:
    clear_cents: int | None  # None = no print
    cleared_qty: int
    tie_lo: int | None
    tie_hi: int | None
    fills: list[Fill] = field(default_factory=list)
    # Where the book suggests it would have cleared, when nothing crossed.
    # Never a trade and never a clearing price — see _indicative.
    indicative_cents: int | None = None


def demand(quotes, price: int) -> int:
    """Contracts bid at or above `price`."""
    return sum(q["bid_size"] for q in quotes if q["bid_cents"] >= price)


def supply(quotes, price: int) -> int:
    """Contracts offered at or below `price`."""
    return sum(q["ask_size"] for q in quotes if q["ask_cents"] <= price)


def top_of_book(quotes) -> dict | None:
    """Best bid and best ask, with size summed across every agent at each.

    Summing is what makes this publishable: a level reports how much size stood
    there, never whose it was. Returns None when neither side has a quote.
    """
    bids = [q for q in quotes if q["bid_cents"] is not None]
    asks = [q for q in quotes if q["ask_cents"] is not None]
    if not bids and not asks:
        return None

    best_bid = max((q["bid_cents"] for q in bids), default=None)
    best_ask = min((q["ask_cents"] for q in asks), default=None)
    return {
        "bid_cents": best_bid,
        "bid_size": sum(q["bid_size"] for q in bids if q["bid_cents"] == best_bid),
        "ask_cents": best_ask,
        "ask_size": sum(q["ask_size"] for q in asks if q["ask_cents"] == best_ask),
    }


def imbalance(quotes, price: int) -> int:
    """Contracts left unfilled at `price`, signed: + bids, - asks.

    At the clearing price the executed volume is min(demand, supply); whatever
    the larger side brought beyond that could not trade. Publishing it says how
    much of the demand at the print went unsatisfied, and on which side.

    Evaluated at the price actually chosen, not across the tie interval: the
    two sides move in opposite directions across that interval, so only the
    chosen price gives a defined answer.
    """
    return demand(quotes, price) - supply(quotes, price)


def _indicative(quotes) -> int | None:
    """Size-weighted mid of an uncrossed book — the microprice.

    Each side is weighted by the OTHER side's size: heavy resting size at the
    best ask means sellers outnumber buyers there, which pulls fair value down
    toward the bid.

        (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)

    Sizes are summed across every agent quoting at that best level, so one
    agent showing size cannot dominate the estimate on its own. This is a
    read on the book, not a trade: no fill happens at it, and it is never fed
    back as a clearing price.
    """
    top = top_of_book(quotes)
    if top is None or top["bid_cents"] is None or top["ask_cents"] is None:
        return None

    bid_size, ask_size = top["bid_size"], top["ask_size"]
    if bid_size + ask_size == 0:
        return None

    weighted = (
        top["bid_cents"] * ask_size + top["ask_cents"] * bid_size
    ) / (bid_size + ask_size)
    return max(config.PRICE_MIN, min(config.PRICE_MAX, round(weighted)))


def _round_toward_50(value: float) -> int:
    """Round a half-integer price toward 50, the neutral point of the grid."""
    low, high = int(value), int(value) + (1 if value % 1 else 0)
    if low == high:
        return low
    return high if value < 50 else low


def _pick_price(candidates: list[int], last_clear_cents: int | None) -> int:
    """Choose within the tie interval.

    First auction in a market: the midpoint, rounded toward 50. Afterwards:
    the candidate closest to the last internal clear, an exact tie resolved
    toward 50 (and, if that is still tied, toward the lower price, so the
    result stays deterministic).
    """
    if last_clear_cents is None:
        return _round_toward_50((candidates[0] + candidates[-1]) / 2)
    return min(
        candidates,
        key=lambda p: (abs(p - last_clear_cents), abs(p - 50), p),
    )


def _allocate(orders, total: int) -> dict[str, int]:
    """Fill `total` contracts across `orders` by price priority.

    `orders` is a list of (rank_key, agent, price, size) already sorted so the
    most aggressive comes first. Whole price levels fill until the volume runs
    out; the level it runs out on is shared pro-rata by size, rounded down,
    with the remainder handed out one contract each in lexicographic agent
    order.

    Price priority rather than "everything strictly inside fills fully":
    when the tie interval is wide, strictly-inside volume can itself exceed the
    cleared volume, and only price priority stays well defined there.
    """
    allocation: dict[str, int] = {}
    remaining = total
    index = 0
    while remaining > 0 and index < len(orders):
        level_price = orders[index][0]
        level = []
        while index < len(orders) and orders[index][0] == level_price:
            level.append(orders[index])
            index += 1

        level_total = sum(size for _, _, size in ((k, a, s) for k, a, s in level))
        if level_total <= remaining:
            for _, agent, size in level:
                allocation[agent] = allocation.get(agent, 0) + size
            remaining -= level_total
            continue

        # Marginal level: pro-rata by size, rounded down.
        level.sort(key=lambda o: o[1])  # lexicographic agent id
        shares = [(agent, size * remaining // level_total) for _, agent, size in level]
        handed_out = sum(share for _, share in shares)
        leftover = remaining - handed_out
        for position, (agent, share) in enumerate(shares):
            extra = 1 if position < leftover else 0
            if share + extra:
                allocation[agent] = allocation.get(agent, 0) + share + extra
        remaining = 0

    return allocation


def clear(quotes, last_clear_cents: int | None = None) -> Result:
    """Clear one market for one round.

    `quotes` is a list of dicts with agent, bid_cents, bid_size, ask_cents,
    ask_size — each two-sided quote contributing one buy and one sell order.
    Upstream validation guarantees bid < ask, so no single uniform price can
    put an agent on both sides of its own trade.
    """
    quotes = list(quotes)
    if not quotes:
        return Result(None, 0, None, None, [], None)

    grid = range(config.PRICE_MIN, config.PRICE_MAX + 1)
    volumes = {p: min(demand(quotes, p), supply(quotes, p)) for p in grid}
    best = max(volumes.values())

    if best == 0:
        return Result(None, 0, None, None, [], _indicative(quotes))

    candidates = [p for p in grid if volumes[p] == best]
    tie_lo, tie_hi = candidates[0], candidates[-1]
    price = _pick_price(candidates, last_clear_cents)

    buys = sorted(
        ((-q["bid_cents"], q["agent"], q["bid_size"])
         for q in quotes if q["bid_cents"] >= price),
    )
    sells = sorted(
        ((q["ask_cents"], q["agent"], q["ask_size"])
         for q in quotes if q["ask_cents"] <= price),
    )

    bought = _allocate(buys, best)
    sold = _allocate(sells, best)

    fills = [Fill(agent, qty, price) for agent, qty in sorted(bought.items()) if qty]
    fills += [Fill(agent, -qty, price) for agent, qty in sorted(sold.items()) if qty]

    return Result(price, best, tie_lo, tie_hi, fills)
