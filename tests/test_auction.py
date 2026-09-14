"""Auction clearing. Pure functions, so these use synthetic quotes only."""

import random

import pytest

from prediction_markets import auction, config


def q(agent, bid, bid_size, ask, ask_size):
    return {
        "agent": agent, "bid_cents": bid, "bid_size": bid_size,
        "ask_cents": ask, "ask_size": ask_size,
    }


def brute_force_volumes(quotes):
    """V(p) computed the slow, obvious way, for the property test to trust."""
    volumes = {}
    for price in range(config.PRICE_MIN, config.PRICE_MAX + 1):
        demand = sum(x["bid_size"] for x in quotes if x["bid_cents"] >= price)
        supply = sum(x["ask_size"] for x in quotes if x["ask_cents"] <= price)
        volumes[price] = min(demand, supply)
    return volumes


# --- volume maximisation ---------------------------------------------------


def test_clear_price_maximises_volume_over_random_books():
    rng = random.Random(20260901)
    for _ in range(400):
        quotes = []
        for i in range(rng.randint(2, 9)):
            bid = rng.randint(config.PRICE_MIN, config.PRICE_MAX - 1)
            ask = rng.randint(bid + 1, config.PRICE_MAX)
            quotes.append(q(f"a{i}", bid, rng.randint(1, 50), ask, rng.randint(1, 50)))

        volumes = brute_force_volumes(quotes)
        best = max(volumes.values())
        result = auction.clear(quotes, None)

        assert result.cleared_qty == best
        if best == 0:
            assert result.clear_cents is None
            assert result.fills == []
            continue

        assert volumes[result.clear_cents] == best
        assert result.tie_lo == min(p for p, v in volumes.items() if v == best)
        assert result.tie_hi == max(p for p, v in volumes.items() if v == best)

        bought = sum(f.signed_qty for f in result.fills if f.signed_qty > 0)
        sold = -sum(f.signed_qty for f in result.fills if f.signed_qty < 0)
        assert bought == sold == best
        assert all(f.price_cents == result.clear_cents for f in result.fills)


def test_no_agent_ever_fills_both_sides_of_its_own_quote():
    """bid < ask upstream means no uniform price can cross an agent with itself."""
    rng = random.Random(7)
    for _ in range(400):
        quotes = []
        for i in range(rng.randint(2, 8)):
            bid = rng.randint(config.PRICE_MIN, config.PRICE_MAX - 1)
            ask = rng.randint(bid + 1, config.PRICE_MAX)
            quotes.append(q(f"a{i}", bid, rng.randint(1, 50), ask, rng.randint(1, 50)))

        result = auction.clear(quotes, rng.choice([None, 20, 50, 80]))
        signs = {}
        for fill in result.fills:
            sign = 1 if fill.signed_qty > 0 else -1
            assert signs.setdefault(fill.agent, sign) == sign


# --- tie breaking ----------------------------------------------------------


def test_first_auction_takes_the_midpoint_of_the_tie_interval():
    quotes = [q("a", 60, 10, 99, 1), q("b", 1, 1, 40, 10)]
    result = auction.clear(quotes, None)
    assert (result.tie_lo, result.tie_hi) == (40, 60)
    assert result.clear_cents == 50


@pytest.mark.parametrize(
    "bid, ask, expected",
    [
        (60, 41, 50),  # midpoint 50.5 rounds down toward 50
        (21, 20, 21),  # midpoint 20.5 rounds up toward 50
        (81, 80, 80),  # midpoint 80.5 rounds down toward 50
    ],
)
def test_midpoint_half_values_round_toward_50(bid, ask, expected):
    result = auction.clear([q("a", bid, 10, 99, 1), q("b", 1, 1, ask, 10)], None)
    assert result.clear_cents == expected


@pytest.mark.parametrize(
    "last_clear, expected",
    [(45, 45), (70, 60), (10, 40), (50, 50)],
)
def test_later_auctions_clear_closest_to_the_last_internal_clear(last_clear, expected):
    quotes = [q("a", 60, 10, 99, 1), q("b", 1, 1, 40, 10)]
    result = auction.clear(quotes, last_clear)
    assert (result.tie_lo, result.tie_hi) == (40, 60)
    assert result.clear_cents == expected


def test_exact_distance_tie_resolves_toward_50():
    # Candidates 40..60, last clear 30 is equidistant from nothing, but a last
    # clear of exactly 50 sits on a candidate; use a symmetric interval instead.
    quotes = [q("a", 55, 10, 99, 1), q("b", 1, 1, 45, 10)]
    result = auction.clear(quotes, 50)
    assert result.clear_cents == 50


# --- allocation ------------------------------------------------------------


def test_marginal_level_is_pro_rata_and_order_independent():
    quotes = [
        q("alpha", 60, 10, 99, 1),
        q("bravo", 60, 30, 99, 1),
        q("charlie", 1, 1, 50, 10),
    ]
    result = auction.clear(quotes, None)
    assert result.cleared_qty == 10

    filled = {f.agent: f.signed_qty for f in result.fills}
    # 10 contracts across 40 quoted at the same price: 10*10//40=2 and
    # 30*10//40=7, with the single remainder going to the first agent id.
    assert filled == {"alpha": 3, "bravo": 7, "charlie": -10}

    shuffled = auction.clear(list(reversed(quotes)), None)
    assert {f.agent: f.signed_qty for f in shuffled.fills} == filled


def test_price_priority_before_pro_rata():
    quotes = [
        q("alpha", 70, 10, 99, 1),  # more aggressive, fills first
        q("bravo", 55, 30, 99, 1),
        q("charlie", 1, 1, 50, 15),
    ]
    result = auction.clear(quotes, None)
    filled = {f.agent: f.signed_qty for f in result.fills}
    assert result.cleared_qty == 15
    assert filled["alpha"] == 10
    assert filled["bravo"] == 5


# --- degenerate cases ------------------------------------------------------


def test_no_overlap_produces_no_print():
    result = auction.clear([q("a", 20, 10, 80, 10), q("b", 21, 10, 79, 10)], None)
    assert result.clear_cents is None
    assert result.cleared_qty == 0
    assert result.fills == []
    assert (result.tie_lo, result.tie_hi) == (None, None)


def test_empty_book_produces_no_print():
    result = auction.clear([], None)
    assert result.clear_cents is None and result.fills == []


def test_replay_is_deterministic():
    rng = random.Random(99)
    quotes = [
        q(f"agent{i}", b := rng.randint(10, 70), rng.randint(1, 50),
          rng.randint(b + 1, 99), rng.randint(1, 50))
        for i in range(6)
    ]
    first = auction.clear(quotes, 44)
    for _ in range(5):
        assert auction.clear(quotes, 44) == first
