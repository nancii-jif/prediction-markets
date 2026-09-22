"""Step-two contracts for accounting, funding, matching, and exchange reads.

Settlement and lifecycle mutations intentionally remain outside this module;
they belong to step three of the MVP migration.
"""

from __future__ import annotations

import sqlite3

import pytest

from prediction_markets import config
from prediction_markets.storage import database as database
from prediction_markets.exchange import Exchange


NOW = 2_000_000_000
AGENTS = tuple(f"agent-{number:02d}" for number in range(1, 9))


def _market(index: int) -> dict:
    return {
        "ticker": f"MKT-{index:02d}",
        "event_ticker": f"EVENT-{index:02d}",
        "title": f"Market {index}",
        "rules": f"Market {index} resolves according to its source.",
        "market_type": "binary",
        "status": "active",
        "close_time": NOW + 100_000 + index,
        "expected_expiration_time": NOW + 7_000_000 + index,
        "latest_expiration_time": NOW + 8_000_000 + index,
        "last_successful_poll": NOW,
        "payout_cents": None,
    }


@pytest.fixture
def venue(tmp_path):
    settings = config.load(config.CONFIG_DIR / "mvp.yaml")
    conn = database.init(tmp_path / "run.db", run_settings=settings)
    database.store_market_cohort([_market(index) for index in range(1, 11)])
    exchange = Exchange(conn, now=lambda: NOW)
    try:
        yield exchange, conn, settings
    finally:
        database.close()


def _submit(
    exchange: Exchange,
    agent_id: str,
    ticker: str,
    action: str,
    order_type: str,
    quantity: int,
    price_cents: int | None = None,
) -> dict:
    result = exchange.submit_order(
        agent_id,
        ticker,
        action,
        order_type,
        quantity,
        price_cents,
    )
    assert result["ok"] is True, result
    return result


def _account(exchange: Exchange, agent_id: str) -> dict:
    result = exchange.get_account(agent_id)
    assert result["ok"] is True, result
    return result


def _positions(account: dict) -> dict[str, int]:
    return {
        position["ticker"]: position["signed_qty"]
        for position in account["positions"]
    }


def _own_order(exchange: Exchange, agent_id: str, order_id: int) -> dict:
    result = exchange.get_own_orders(agent_id)
    assert result["ok"] is True, result
    return next(
        order for order in result["orders"] if order["order_id"] == order_id
    )


def _assert_error(result: dict) -> None:
    assert result["ok"] is False
    assert set(result) == {"ok", "error"}
    assert isinstance(result["error"]["code"], str)
    assert result["error"]["code"]
    assert isinstance(result["error"]["message"], str)
    assert result["error"]["message"]


def test_fresh_short_and_cover_use_purchase_style_cash(venue):
    exchange, conn, _ = venue
    conn.execute(
        "UPDATE accounts SET balance_cents = 10000 WHERE agent_id = ?",
        (AGENTS[0],),
    )
    conn.commit()

    _submit(exchange, AGENTS[0], "MKT-01", "sell", "limit", 10, 40)
    _submit(exchange, AGENTS[1], "MKT-01", "buy", "market", 10)

    short = _account(exchange, AGENTS[0])
    assert short["cash_cents"] == 9_400
    assert _positions(short) == {"MKT-01": -10}
    assert short["reserved_cents"] == 0

    _submit(exchange, AGENTS[2], "MKT-01", "sell", "limit", 10, 30)
    _submit(exchange, AGENTS[0], "MKT-01", "buy", "market", 10)

    covered = _account(exchange, AGENTS[0])
    assert covered["cash_cents"] == 10_100
    assert covered["positions"] == []
    assert covered["reserved_cents"] == 0
    assert covered["available_cents"] == 10_100


def test_selling_can_flip_a_long_position_to_short(venue):
    exchange, _, _ = venue
    _submit(exchange, AGENTS[1], "MKT-01", "sell", "limit", 10, 40)
    _submit(exchange, AGENTS[0], "MKT-01", "buy", "market", 10)

    _submit(exchange, AGENTS[2], "MKT-01", "buy", "limit", 15, 60)
    _submit(exchange, AGENTS[0], "MKT-01", "sell", "market", 15)

    account = _account(exchange, AGENTS[0])
    assert account["cash_cents"] == 100_000
    assert _positions(account) == {"MKT-01": -5}


def test_buying_can_flip_a_short_position_to_long(venue):
    exchange, _, _ = venue
    _submit(exchange, AGENTS[0], "MKT-01", "sell", "limit", 10, 40)
    _submit(exchange, AGENTS[1], "MKT-01", "buy", "market", 10)

    _submit(exchange, AGENTS[2], "MKT-01", "sell", "limit", 15, 30)
    _submit(exchange, AGENTS[0], "MKT-01", "buy", "market", 15)

    account = _account(exchange, AGENTS[0])
    assert account["cash_cents"] == 99_950
    assert _positions(account) == {"MKT-01": 5}


def test_reservations_use_worst_side_per_market_and_sum_across_markets(venue):
    exchange, conn, _ = venue

    buy = _submit(
        exchange, AGENTS[0], "MKT-01", "buy", "limit", 1_000, 60
    )["order"]
    assert _account(exchange, AGENTS[0])["reserved_cents"] == 60_000

    sell = _submit(
        exchange, AGENTS[0], "MKT-01", "sell", "limit", 1_000, 70
    )["order"]
    account = _account(exchange, AGENTS[0])
    assert account["reserved_cents"] == 60_000
    assert account["available_cents"] == 40_000

    _submit(exchange, AGENTS[0], "MKT-02", "buy", "limit", 800, 50)
    account = _account(exchange, AGENTS[0])
    assert account["reserved_cents"] == 100_000
    assert account["available_cents"] == 0

    rejected = exchange.submit_order(
        AGENTS[0], "MKT-03", "buy", "limit", 1, 1
    )
    _assert_error(rejected)
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3

    assert exchange.cancel_order(AGENTS[0], buy["order_id"])["ok"] is True
    assert _account(exchange, AGENTS[0])["reserved_cents"] == 70_000
    assert exchange.cancel_order(AGENTS[0], sell["order_id"])["ok"] is True
    account = _account(exchange, AGENTS[0])
    assert account["reserved_cents"] == 40_000
    assert account["available_cents"] == 60_000


def test_orders_that_only_close_holdings_need_no_unreserved_cash(venue):
    exchange, conn, _ = venue
    conn.executemany(
        "UPDATE accounts SET balance_cents = ? WHERE agent_id = ?",
        ((400, AGENTS[0]), (600, AGENTS[2])),
    )
    conn.commit()

    # Spend all cash acquiring YES, then offer only the owned contracts.
    _submit(exchange, AGENTS[1], "MKT-01", "sell", "limit", 10, 40)
    _submit(exchange, AGENTS[0], "MKT-01", "buy", "limit", 10, 40)
    assert _account(exchange, AGENTS[0])["available_cents"] == 0
    closing_long = _submit(
        exchange, AGENTS[0], "MKT-01", "sell", "limit", 10, 50
    )["order"]
    long_account = _account(exchange, AGENTS[0])
    assert long_account["cash_cents"] == 0
    assert long_account["reserved_cents"] == 0
    assert closing_long["status"] == "open"

    # Spend all cash acquiring the NO-equivalent, then bid only to cover it.
    _submit(exchange, AGENTS[2], "MKT-02", "sell", "limit", 10, 40)
    _submit(exchange, AGENTS[3], "MKT-02", "buy", "market", 10)
    assert _account(exchange, AGENTS[2])["available_cents"] == 0
    closing_short = _submit(
        exchange, AGENTS[2], "MKT-02", "buy", "limit", 10, 20
    )["order"]
    short_account = _account(exchange, AGENTS[2])
    assert short_account["cash_cents"] == 0
    assert short_account["reserved_cents"] == 0
    assert closing_short["status"] == "open"


def test_partial_fill_and_cancellation_recalculate_reservations(venue):
    exchange, _, _ = venue
    bid = _submit(
        exchange, AGENTS[0], "MKT-01", "buy", "limit", 10, 60
    )["order"]
    assert _account(exchange, AGENTS[0])["reserved_cents"] == 600

    _submit(exchange, AGENTS[1], "MKT-01", "sell", "limit", 4, 50)
    partially_filled = _own_order(
        exchange, AGENTS[0], bid["order_id"]
    )
    assert partially_filled["status"] == "open"
    assert partially_filled["filled_quantity"] == 4
    assert partially_filled["remaining_quantity"] == 6

    account = _account(exchange, AGENTS[0])
    assert account["cash_cents"] == 99_760
    assert account["reserved_cents"] == 360
    assert account["available_cents"] == 99_400

    canceled = exchange.cancel_order(AGENTS[0], bid["order_id"])
    assert canceled["ok"] is True
    assert canceled["order"]["status"] == "canceled"
    assert canceled["order"]["filled_quantity"] == 4
    assert canceled["order"]["remaining_quantity"] == 0
    account = _account(exchange, AGENTS[0])
    assert account["reserved_cents"] == 0
    assert account["available_cents"] == 99_760


def test_matching_is_price_time_priority_and_executes_at_maker_prices(venue):
    exchange, _, _ = venue
    first_40 = _submit(
        exchange, AGENTS[0], "MKT-01", "sell", "limit", 5, 40
    )["order"]
    second_40 = _submit(
        exchange, AGENTS[1], "MKT-01", "sell", "limit", 5, 40
    )["order"]
    best_35 = _submit(
        exchange, AGENTS[2], "MKT-01", "sell", "limit", 5, 35
    )["order"]

    incoming = _submit(
        exchange, AGENTS[3], "MKT-01", "buy", "limit", 12, 50
    )
    assert [
        (trade["price_cents"], trade["quantity"])
        for trade in incoming["trades"]
    ] == [(35, 5), (40, 5), (40, 2)]
    assert incoming["order"]["status"] == "filled"

    assert _own_order(exchange, AGENTS[2], best_35["order_id"])["status"] == (
        "filled"
    )
    assert _own_order(
        exchange, AGENTS[0], first_40["order_id"]
    )["status"] == "filled"
    remaining = _own_order(exchange, AGENTS[1], second_40["order_id"])
    assert remaining["status"] == "open"
    assert remaining["filled_quantity"] == 2
    assert remaining["remaining_quantity"] == 3


def test_bid_matching_uses_highest_price_then_earliest_order(venue):
    exchange, _, _ = venue
    first_60 = _submit(
        exchange, AGENTS[0], "MKT-01", "buy", "limit", 5, 60
    )["order"]
    second_60 = _submit(
        exchange, AGENTS[1], "MKT-01", "buy", "limit", 5, 60
    )["order"]
    best_65 = _submit(
        exchange, AGENTS[2], "MKT-01", "buy", "limit", 5, 65
    )["order"]

    incoming = _submit(
        exchange, AGENTS[3], "MKT-01", "sell", "limit", 12, 50
    )
    assert [
        (trade["price_cents"], trade["quantity"])
        for trade in incoming["trades"]
    ] == [(65, 5), (60, 5), (60, 2)]
    assert _own_order(exchange, AGENTS[2], best_65["order_id"])["status"] == (
        "filled"
    )
    assert _own_order(
        exchange, AGENTS[0], first_60["order_id"]
    )["status"] == "filled"
    remaining = _own_order(exchange, AGENTS[1], second_60["order_id"])
    assert remaining["filled_quantity"] == 2
    assert remaining["remaining_quantity"] == 3


def test_limit_remainder_rests_but_market_remainder_is_canceled(venue):
    exchange, _, _ = venue
    _submit(exchange, AGENTS[0], "MKT-01", "sell", "limit", 5, 40)
    limit = _submit(
        exchange, AGENTS[1], "MKT-01", "buy", "limit", 10, 50
    )["order"]
    assert limit["status"] == "open"
    assert limit["filled_quantity"] == 5
    assert limit["remaining_quantity"] == 5

    _submit(exchange, AGENTS[2], "MKT-02", "sell", "limit", 5, 40)
    market = _submit(
        exchange, AGENTS[3], "MKT-02", "buy", "market", 10
    )["order"]
    assert market["status"] == "canceled"
    assert market["filled_quantity"] == 5
    assert market["remaining_quantity"] == 0
    assert market["cancel_reason"] == "market_remainder"


def test_self_match_fills_both_legs_at_price_time_priority(venue):
    exchange, _, _ = venue
    own_ask = _submit(
        exchange, AGENTS[0], "MKT-01", "sell", "limit", 5, 40
    )["order"]
    _submit(exchange, AGENTS[1], "MKT-01", "sell", "limit", 2, 35)

    incoming = _submit(
        exchange, AGENTS[0], "MKT-01", "buy", "market", 5
    )
    # The better-priced stranger still matches first; the remainder then
    # crosses this agent's own resting ask instead of being canceled.
    assert [
        (trade["price_cents"], trade["quantity"])
        for trade in incoming["trades"]
    ] == [(35, 2), (40, 3)]
    assert incoming["order"]["status"] == "filled"
    assert incoming["order"]["filled_quantity"] == 5
    assert incoming["order"]["cancel_reason"] is None

    original = _own_order(exchange, AGENTS[0], own_ask["order_id"])
    assert original["status"] == "open"
    assert original["filled_quantity"] == 3
    assert original["remaining_quantity"] == 2


def test_self_match_leaves_cash_and_position_unchanged(venue):
    exchange, _, _ = venue
    before = _account(exchange, AGENTS[0])
    _submit(exchange, AGENTS[0], "MKT-01", "sell", "limit", 10, 60)
    _submit(exchange, AGENTS[0], "MKT-01", "buy", "limit", 10, 60)

    after = _account(exchange, AGENTS[0])
    assert after["cash_cents"] == before["cash_cents"]
    assert _positions(after) == _positions(before)
    assert after["reserved_cents"] == 0
    # Both legs still print, so the wash sets the mark other agents are valued at.
    trades = exchange.get_recent_trades("MKT-01")["trades"]
    assert [(trade["price_cents"], trade["quantity"]) for trade in trades] == [(60, 10)]


def test_only_owner_can_cancel_and_filled_orders_cannot_be_undone(venue):
    exchange, _, _ = venue
    resting = _submit(
        exchange, AGENTS[0], "MKT-01", "sell", "limit", 3, 40
    )["order"]

    denied = exchange.cancel_order(AGENTS[1], resting["order_id"])
    _assert_error(denied)
    assert _own_order(
        exchange, AGENTS[0], resting["order_id"]
    )["status"] == "open"

    canceled = exchange.cancel_order(AGENTS[0], resting["order_id"])
    assert canceled["ok"] is True
    assert canceled["order"]["status"] == "canceled"
    _assert_error(exchange.cancel_order(AGENTS[0], resting["order_id"]))

    filled = _submit(
        exchange, AGENTS[0], "MKT-02", "sell", "limit", 3, 40
    )["order"]
    _submit(exchange, AGENTS[1], "MKT-02", "buy", "market", 3)
    _assert_error(exchange.cancel_order(AGENTS[0], filled["order_id"]))
    assert _own_order(
        exchange, AGENTS[0], filled["order_id"]
    )["status"] == "filled"


@pytest.mark.parametrize(
    "arguments",
    [
        ("MKT-01", "hold", "limit", 1, 50),
        ("MKT-01", ["buy"], "limit", 1, 50),
        ("MKT-01", "buy", "iceberg", 1, 50),
        ("MKT-01", "buy", {"kind": "limit"}, 1, 50),
        ("MKT-01", "buy", "limit", 0, 50),
        ("MKT-01", "buy", "limit", -1, 50),
        ("MKT-01", "buy", "limit", True, 50),
        ("MKT-01", "buy", "limit", 1.5, 50),
        ("MKT-01", "buy", "limit", 1, None),
        ("MKT-01", "buy", "limit", 1, 0),
        ("MKT-01", "buy", "limit", 1, 100),
        ("MKT-01", "buy", "limit", 1, True),
        ("MKT-01", "buy", "limit", 1, 40.5),
        ("MKT-01", "buy", "market", 1, 50),
        ("NOT-IN-COHORT", "buy", "limit", 1, 50),
    ],
)
def test_invalid_order_inputs_are_structured_and_create_no_order(
    venue, arguments
):
    exchange, conn, _ = venue
    result = exchange.submit_order(AGENTS[0], *arguments)
    _assert_error(result)
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_unknown_agent_and_invalid_cancel_inputs_are_structured(venue):
    exchange, _, _ = venue
    _assert_error(
        exchange.submit_order(
            "not-an-agent", "MKT-01", "buy", "limit", 1, 50
        )
    )
    _assert_error(exchange.cancel_order("not-an-agent", 1))
    _assert_error(exchange.cancel_order(AGENTS[0], True))
    _assert_error(exchange.cancel_order(AGENTS[0], 0))
    _assert_error(exchange.cancel_order(AGENTS[0], 99_999))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("status", "inactive"),
        ("close_time", NOW),
        ("last_successful_poll", NOW - 601),
    ],
)
def test_inactive_closed_or_stale_market_rejects_new_orders(
    venue, column, value
):
    exchange, conn, _ = venue
    conn.execute(f"UPDATE markets SET {column} = ? WHERE ticker = 'MKT-01'", (value,))
    conn.commit()

    result = exchange.submit_order(
        AGENTS[0], "MKT-01", "buy", "limit", 1, 50
    )
    _assert_error(result)
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_market_order_funding_uses_full_protection_bound(venue):
    exchange, conn, _ = venue
    conn.execute(
        "UPDATE accounts SET balance_cents = 500 WHERE agent_id = ?",
        (AGENTS[0],),
    )
    conn.commit()
    _submit(exchange, AGENTS[1], "MKT-01", "sell", "limit", 10, 1)

    # The visible liquidity would cost only 10 cents, but admission reserves
    # 99 cents per requested contract and must not silently resize the order.
    result = exchange.submit_order(
        AGENTS[0], "MKT-01", "buy", "market", 10
    )
    _assert_error(result)
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM orders WHERE agent_id = ?", (AGENTS[0],)
    ).fetchone()[0] == 0

    conn.execute(
        "UPDATE accounts SET balance_cents = 500 WHERE agent_id = ?",
        (AGENTS[2],),
    )
    conn.commit()
    _submit(exchange, AGENTS[3], "MKT-02", "buy", "limit", 10, 99)

    # Symmetrically, an unpriced sell is funded as though it executes at 1.
    result = exchange.submit_order(
        AGENTS[2], "MKT-02", "sell", "market", 10
    )
    _assert_error(result)
    assert conn.execute(
        "SELECT COUNT(*) FROM trades WHERE ticker = 'MKT-02'"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM orders WHERE agent_id = ?", (AGENTS[2],)
    ).fetchone()[0] == 0


def test_public_book_aggregates_five_levels_without_identity(venue):
    exchange, _, _ = venue
    for price in range(10, 16):
        _submit(exchange, AGENTS[0], "MKT-01", "buy", "limit", 1, price)
    _submit(exchange, AGENTS[2], "MKT-01", "buy", "limit", 4, 15)
    for price in range(70, 76):
        _submit(exchange, AGENTS[1], "MKT-01", "sell", "limit", 2, price)

    book = exchange.get_orderbook("MKT-01")
    assert book["ok"] is True
    assert book["ticker"] == "MKT-01"
    assert book["bids"] == [
        {"price_cents": 15, "quantity": 5},
        {"price_cents": 14, "quantity": 1},
        {"price_cents": 13, "quantity": 1},
        {"price_cents": 12, "quantity": 1},
        {"price_cents": 11, "quantity": 1},
    ]
    assert book["asks"] == [
        {"price_cents": 70, "quantity": 2},
        {"price_cents": 71, "quantity": 2},
        {"price_cents": 72, "quantity": 2},
        {"price_cents": 73, "quantity": 2},
        {"price_cents": 74, "quantity": 2},
    ]
    for level in book["bids"] + book["asks"]:
        assert set(level) == {"price_cents", "quantity"}
    _assert_error(exchange.get_orderbook("NOT-IN-COHORT"))


def test_recent_trades_returns_newest_five_without_private_identifiers(venue):
    exchange, _, _ = venue
    for price in range(30, 36):
        _submit(
            exchange, AGENTS[0], "MKT-01", "sell", "limit", 1, price
        )
        _submit(exchange, AGENTS[1], "MKT-01", "buy", "market", 1)

    result = exchange.get_recent_trades("MKT-01")
    assert result["ok"] is True
    assert result["ticker"] == "MKT-01"
    assert [trade["price_cents"] for trade in result["trades"]] == [
        35,
        34,
        33,
        32,
        31,
    ]
    assert all(
        set(trade) == {"ticker", "price_cents", "quantity", "executed_at"}
        for trade in result["trades"]
    )
    _assert_error(exchange.get_recent_trades("NOT-IN-COHORT"))


def test_own_orders_are_private_complete_and_newest_first(venue):
    exchange, _, _ = venue
    created_ids = []
    for _ in range(25):
        created_ids.append(
            _submit(
                exchange, AGENTS[0], "MKT-03", "buy", "limit", 1, 1
            )["order"]["order_id"]
        )
    for _ in range(2):
        _submit(exchange, AGENTS[0], "MKT-04", "buy", "limit", 1, 1)
    _submit(exchange, AGENTS[1], "MKT-03", "buy", "limit", 1, 1)

    # No page cap: all 25 come back in one call, newest first.
    history = exchange.get_own_orders(AGENTS[0], ticker="MKT-03")
    assert history["ok"] is True
    assert [order["order_id"] for order in history["orders"]] == list(
        reversed(created_ids)
    )
    assert all(order["agent_id"] == AGENTS[0] for order in history["orders"])

    every_market = exchange.get_own_orders(AGENTS[0])
    assert len(every_market["orders"]) == 27

    other_market = exchange.get_own_orders(AGENTS[0], ticker="MKT-04")
    assert len(other_market["orders"]) == 2
    assert all(order["ticker"] == "MKT-04" for order in other_market["orders"])
    _assert_error(exchange.get_own_orders("not-an-agent"))


def test_own_orders_include_filled_and_canceled_history(venue):
    exchange, _, _ = venue
    resting = _submit(exchange, AGENTS[0], "MKT-01", "sell", "limit", 5, 40)["order"]
    _submit(exchange, AGENTS[1], "MKT-01", "buy", "limit", 5, 40)
    canceled = _submit(exchange, AGENTS[0], "MKT-01", "sell", "limit", 3, 90)["order"]
    exchange.cancel_order(AGENTS[0], canceled["order_id"])

    orders = exchange.get_own_orders(AGENTS[0], ticker="MKT-01")["orders"]
    assert {order["order_id"]: order["status"] for order in orders} == {
        resting["order_id"]: "filled", canceled["order_id"]: "canceled",
    }
    filled = next(o for o in orders if o["order_id"] == resting["order_id"])
    assert (filled["filled_quantity"], filled["remaining_quantity"]) == (5, 0)


def test_account_marks_positions_at_last_internal_trade_and_reports_pnl(venue):
    exchange, _, _ = venue
    _submit(exchange, AGENTS[0], "MKT-01", "sell", "limit", 10, 40)
    _submit(exchange, AGENTS[1], "MKT-01", "buy", "market", 10)

    buyer = _account(exchange, AGENTS[1])
    seller = _account(exchange, AGENTS[0])
    assert buyer["equity_cents"] == 100_000
    assert buyer["total_pnl_cents"] == 0
    assert buyer["positions"] == [
        {
            "ticker": "MKT-01",
            "signed_qty": 10,
            "mark_cents": 40,
            "value_cents": 400,
        }
    ]
    assert seller["equity_cents"] == 100_000
    assert seller["total_pnl_cents"] == 0

    _submit(exchange, AGENTS[2], "MKT-01", "sell", "limit", 1, 60)
    _submit(exchange, AGENTS[3], "MKT-01", "buy", "market", 1)

    buyer = _account(exchange, AGENTS[1])
    seller = _account(exchange, AGENTS[0])
    assert buyer["positions"][0]["mark_cents"] == 60
    assert buyer["positions"][0]["value_cents"] == 600
    assert buyer["equity_cents"] == 100_200
    assert buyer["total_pnl_cents"] == 200
    assert seller["positions"][0]["mark_cents"] == 60
    assert seller["positions"][0]["value_cents"] == 400
    assert seller["equity_cents"] == 99_800
    assert seller["total_pnl_cents"] == -200


def test_trades_conserve_signed_quantity_and_cash_plus_settlement_pool(venue):
    exchange, conn, settings = venue
    _submit(exchange, AGENTS[0], "MKT-01", "sell", "limit", 10, 40)
    _submit(exchange, AGENTS[1], "MKT-01", "buy", "market", 10)
    _submit(exchange, AGENTS[2], "MKT-01", "buy", "limit", 4, 70)
    _submit(exchange, AGENTS[1], "MKT-01", "sell", "market", 4)
    _submit(exchange, AGENTS[3], "MKT-02", "sell", "limit", 20, 20)
    _submit(exchange, AGENTS[4], "MKT-02", "buy", "market", 20)
    _submit(exchange, AGENTS[5], "MKT-03", "buy", "limit", 7, 55)
    _submit(exchange, AGENTS[6], "MKT-03", "sell", "market", 7)

    per_market = conn.execute(
        "SELECT ticker, SUM(signed_qty) AS total_qty "
        "FROM positions GROUP BY ticker ORDER BY ticker"
    ).fetchall()
    assert [(row["ticker"], row["total_qty"]) for row in per_market] == [
        ("MKT-01", 0),
        ("MKT-02", 0),
        ("MKT-03", 0),
    ]

    cash = conn.execute("SELECT SUM(balance_cents) FROM accounts").fetchone()[0]
    settlement_pool = 100 * sum(
        max(-row["signed_qty"], 0)
        for row in conn.execute("SELECT signed_qty FROM positions")
    )
    assert cash + settlement_pool == (
        settings.participant_count * settings.initial_cash_cents
    )


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE accounts SET balance_cents = 100.5 WHERE agent_id = 'agent-01'",
        "UPDATE positions SET signed_qty = 0.5 WHERE agent_id = 'agent-01'",
        "UPDATE orders SET price_cents = 40.5 WHERE order_id = 1",
        "UPDATE orders SET original_quantity = 1.5 WHERE order_id = 1",
        "UPDATE trades SET price_cents = 40.5 WHERE trade_id = 1",
        "UPDATE trades SET quantity = 0.5 WHERE trade_id = 1",
        "UPDATE markets SET payout_cents = 0.5 WHERE ticker = 'MKT-01'",
    ],
)
def test_schema_rejects_fractional_money_and_contract_values(venue, statement):
    exchange, conn, _ = venue
    _submit(exchange, AGENTS[0], "MKT-01", "sell", "limit", 1, 40)
    _submit(exchange, AGENTS[1], "MKT-01", "buy", "market", 1)

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(statement)
    conn.rollback()


def test_matching_rolls_back_every_mutation_if_trade_insert_fails(venue):
    exchange, conn, _ = venue
    maker = _submit(
        exchange, AGENTS[0], "MKT-01", "sell", "limit", 5, 40
    )["order"]
    balances_before = conn.execute(
        "SELECT agent_id, balance_cents FROM accounts ORDER BY agent_id"
    ).fetchall()
    conn.executescript(
        """
        CREATE TRIGGER abort_trade_insert
        BEFORE INSERT ON trades
        BEGIN
            SELECT RAISE(ABORT, 'forced trade failure');
        END;
        """
    )

    try:
        result = exchange.submit_order(
            AGENTS[1], "MKT-01", "buy", "market", 5
        )
    except sqlite3.DatabaseError:
        pass
    else:
        _assert_error(result)

    assert conn.in_transaction is False
    assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 0
    maker_after = conn.execute(
        "SELECT status, filled_quantity, remaining_quantity FROM orders "
        "WHERE order_id = ?",
        (maker["order_id"],),
    ).fetchone()
    assert tuple(maker_after) == ("open", 0, 5)
    balances_after = conn.execute(
        "SELECT agent_id, balance_cents FROM accounts ORDER BY agent_id"
    ).fetchall()
    assert [tuple(row) for row in balances_after] == [
        tuple(row) for row in balances_before
    ]


def test_account_money_is_spelled_out_in_dollars(venue):
    exchange, _, _ = venue
    account = _account(exchange, AGENTS[0])
    # 100000 cents was repeatedly read as $100,000 rather than $1,000.
    assert account["cash_cents"] == 100_000
    assert account["in_dollars"]["cash"] == "$1,000.00"
    assert account["in_dollars"]["total_pnl"] == "$0.00"


def test_rejected_quote_cancels_resting_side_but_keeps_what_already_traded(venue):
    exchange, conn, _ = venue
    _submit(exchange, AGENTS[1], "MKT-01", "sell", "limit", 2, 10)
    conn.execute("UPDATE accounts SET balance_cents = 2000 WHERE agent_id = ?", (AGENTS[0],))
    conn.commit()

    # The bid lifts the 2 available and rests 98; the ask is then unfundable.
    result = exchange.submit_quote(AGENTS[0], "MKT-01", 10, 20, 100)
    assert result["ok"] is False
    assert result["error"]["code"] == "insufficient_funds"

    assert exchange.get_orderbook("MKT-01")["bids"] == []
    order = exchange.get_own_orders(AGENTS[0])["orders"][0]
    assert order["status"] == "canceled"
    assert order["cancel_reason"] == "quote_rollback"
    # A filled order cannot be undone, so the two contracts remain held.
    assert order["filled_quantity"] == 2
    assert _positions(_account(exchange, AGENTS[0])) == {"MKT-01": 2}
