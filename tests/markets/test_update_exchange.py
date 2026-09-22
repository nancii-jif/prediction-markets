"""Offline contracts for fixed-cohort closure, observations, and settlement."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

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
    clock = {"now": NOW}
    exchange = Exchange(conn, now=lambda: clock["now"])
    try:
        yield exchange, conn, clock, settings
    finally:
        database.close()


def _observation(conn, ticker="MKT-01", **changes):
    row = dict(conn.execute(
        "SELECT * FROM markets WHERE ticker = ?", (ticker,)
    ).fetchone())
    row.pop("cohort_index")
    return {**row, **changes}


def _submit(exchange, agent, action, quantity, price, ticker="MKT-01"):
    result = exchange.submit_order(agent, ticker, action, "limit", quantity, price)
    assert result["ok"] is True, result
    return result["order"]


def _matched_pair(exchange, ticker="MKT-01", quantity=10, price=40):
    _submit(exchange, AGENTS[0], "sell", quantity, price, ticker)
    _submit(exchange, AGENTS[1], "buy", quantity, price, ticker)


def _assert_conservation(conn, initial_total):
    cash = conn.execute("SELECT SUM(balance_cents) FROM accounts").fetchone()[0]
    pool = conn.execute(
        "SELECT COALESCE(SUM(100 * MAX(-signed_qty, 0)), 0) FROM positions"
    ).fetchone()[0]
    assert cash + pool == initial_total
    for row in conn.execute(
        "SELECT ticker, SUM(signed_qty) FROM positions GROUP BY ticker"
    ):
        assert row[1] == 0, row[0]


def _state(conn):
    return {
        table: [tuple(row) for row in conn.execute(f"SELECT * FROM {table}")]
        for table in (
            "markets", "accounts", "positions", "orders", "trades",
            "market_settlements",
        )
    }


@pytest.mark.parametrize("payout", [0, 37, 100])
def test_final_settlement_pays_each_side_and_preserves_history(venue, payout):
    exchange, conn, clock, settings = venue
    _matched_pair(exchange)
    _matched_pair(exchange, "MKT-02", quantity=3, price=60)
    resting = _submit(exchange, AGENTS[2], "buy", 5, 20)
    before = {
        agent: exchange.get_account(agent)["cash_cents"]
        for agent in AGENTS[:2]
    }
    other_positions = list(conn.execute(
        "SELECT * FROM positions WHERE ticker = 'MKT-02'"
    ))
    history = exchange.get_recent_trades("MKT-01")
    clock["now"] += 10

    result = exchange.apply_market_observation(_observation(
        conn, status="finalized", payout_cents=payout,
        last_successful_poll=clock["now"],
    ))

    assert result["ok"] is True
    assert result["settled"] is True
    assert exchange.get_account(AGENTS[0])["cash_cents"] == (
        before[AGENTS[0]] + 10 * (100 - payout)
    )
    assert exchange.get_account(AGENTS[1])["cash_cents"] == (
        before[AGENTS[1]] + 10 * payout
    )
    assert conn.execute(
        "SELECT COUNT(*) FROM positions WHERE ticker = 'MKT-01'"
    ).fetchone()[0] == 0
    assert list(conn.execute(
        "SELECT * FROM positions WHERE ticker = 'MKT-02'"
    )) == other_positions
    canceled = conn.execute(
        "SELECT * FROM orders WHERE order_id = ?", (resting["order_id"],)
    ).fetchone()
    assert canceled["status"] == "canceled"
    assert canceled["cancel_reason"] == "market_settled"
    assert canceled["remaining_quantity"] == 0
    assert exchange.get_account(AGENTS[2])["reserved_cents"] == 0
    assert exchange.get_recent_trades("MKT-01") == history
    assert exchange.get_orderbook("MKT-01")["bids"] == []
    assert exchange.get_orderbook("MKT-01")["asks"] == []
    assert dict(conn.execute(
        "SELECT * FROM market_settlements WHERE ticker = 'MKT-01'"
    ).fetchone()) == {
        "ticker": "MKT-01", "payout_cents": payout,
        "settled_at": clock["now"],
    }
    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 10
    assert "MKT-01" not in exchange.unsettled_tickers()
    assert len(exchange.unsettled_tickers()) == 9
    assert exchange.all_markets_settled() is False
    assert exchange.submit_order(
        AGENTS[3], "MKT-01", "buy", "limit", 1, 40
    )["ok"] is False
    _assert_conservation(conn, 8 * settings.initial_cash_cents)


@pytest.mark.parametrize("status", ["determined", "disputed", "amended"])
def test_nonfinal_result_never_pays_early(venue, status):
    exchange, conn, _, settings = venue
    _matched_pair(exchange)
    resting = _submit(exchange, AGENTS[2], "buy", 5, 20)
    before = _state(conn)

    result = exchange.apply_market_observation(_observation(
        conn, status=status, payout_cents=100,
    ))

    assert result["ok"] is True
    assert result["settled"] is False
    assert _state(conn)["accounts"] == before["accounts"]
    assert _state(conn)["positions"] == before["positions"]
    assert conn.execute("SELECT COUNT(*) FROM market_settlements").fetchone()[0] == 0
    assert conn.execute(
        "SELECT status FROM orders WHERE order_id = ?", (resting["order_id"],)
    ).fetchone()[0] == "canceled"
    assert "MKT-01" in exchange.unsettled_tickers()
    _assert_conservation(conn, 8 * settings.initial_cash_cents)


def test_final_without_supported_payout_stays_pending_even_after_dates_pass(venue):
    exchange, conn, clock, _ = venue
    _matched_pair(exchange)
    _submit(exchange, AGENTS[2], "buy", 5, 20)
    before = _state(conn)
    clock["now"] += 100_000_000

    result = exchange.apply_market_observation(_observation(
        conn, status="finalized", payout_cents=None,
        last_successful_poll=clock["now"],
    ))
    exchange.close_due_markets()

    assert result["settled"] is False
    assert _state(conn)["accounts"] == before["accounts"]
    assert _state(conn)["positions"] == before["positions"]
    assert conn.execute("SELECT COUNT(*) FROM market_settlements").fetchone()[0] == 0
    assert "MKT-01" in exchange.unsettled_tickers()
    assert exchange.all_markets_settled() is False
    assert exchange.get_account(AGENTS[2])["reserved_cents"] == 0


def test_repeated_final_observation_cannot_pay_twice_or_reopen_market(venue):
    exchange, conn, _, settings = venue
    _matched_pair(exchange)
    final = _observation(conn, status="finalized", payout_cents=100)
    assert exchange.apply_market_observation(final)["settled"] is True
    before = _state(conn)

    exchange.apply_market_observation(final)
    exchange.apply_market_observation({**final, "payout_cents": 0})
    exchange.apply_market_observation({**final, "status": "active"})

    assert _state(conn) == before
    market = exchange.assigned_markets(AGENTS[0])[0]
    assert market["settled"] is True
    assert market["tradable"] is False
    assert market["payout_cents"] == 100
    _assert_conservation(conn, 8 * settings.initial_cash_cents)


def test_settlement_failure_rolls_back_market_cash_positions_and_orders(venue):
    exchange, conn, _, settings = venue
    _matched_pair(exchange)
    _submit(exchange, AGENTS[2], "buy", 5, 20)
    before = _state(conn)
    conn.execute(
        "CREATE TRIGGER fail_settlement BEFORE INSERT ON market_settlements "
        "BEGIN SELECT RAISE(ABORT, 'injected settlement failure'); END"
    )
    conn.commit()
    final = _observation(conn, status="finalized", payout_cents=37)

    with pytest.raises(sqlite3.IntegrityError, match="injected settlement failure"):
        exchange.apply_market_observation(final)

    assert not conn.in_transaction
    assert _state(conn) == before
    conn.execute("DROP TRIGGER fail_settlement")
    conn.commit()
    assert exchange.apply_market_observation(final)["settled"] is True
    _assert_conservation(conn, 8 * settings.initial_cash_cents)


def test_clock_cutoff_cancels_remainders_without_liquidating_holdings(venue):
    exchange, conn, clock, _ = venue
    _submit(exchange, AGENTS[0], "sell", 15, 40)
    _submit(exchange, AGENTS[1], "buy", 10, 40)
    _submit(exchange, AGENTS[2], "buy", 4, 20)
    other = _submit(exchange, AGENTS[3], "buy", 1, 20, "MKT-02")
    before = _state(conn)
    clock["now"] = _market(1)["close_time"]

    assert exchange.submit_order(
        AGENTS[4], "MKT-01", "buy", "limit", 5, 40
    )["ok"] is False
    assert exchange.close_due_markets() == 2
    assert exchange.close_due_markets() == 0
    assert _state(conn)["positions"] == before["positions"]
    assert _state(conn)["accounts"] == before["accounts"]
    assert _state(conn)["trades"] == before["trades"]
    closed = list(conn.execute(
        "SELECT status, cancel_reason, remaining_quantity, filled_quantity "
        "FROM orders WHERE ticker = 'MKT-01' ORDER BY order_id"
    ))
    assert [tuple(row) for row in closed] == [
        ("canceled", "market_closed", 0, 10),
        ("filled", None, 0, 10),
        ("canceled", "market_closed", 0, 0),
    ]
    assert exchange.get_account(AGENTS[0])["reserved_cents"] == 0
    assert exchange.get_account(AGENTS[2])["reserved_cents"] == 0
    assert conn.execute(
        "SELECT status FROM orders WHERE order_id = ?", (other["order_id"],)
    ).fetchone()[0] == "open"
    assert "MKT-01" in exchange.unsettled_tickers()


def test_source_close_cancels_orders_before_local_cutoff(venue):
    exchange, conn, _, _ = venue
    _matched_pair(exchange)
    _submit(exchange, AGENTS[2], "buy", 4, 20)
    before = _state(conn)

    result = exchange.apply_market_observation(_observation(conn, status="closed"))

    assert result["settled"] is False
    assert _state(conn)["positions"] == before["positions"]
    assert _state(conn)["accounts"] == before["accounts"]
    assert exchange.get_account(AGENTS[2])["reserved_cents"] == 0
    assert exchange.submit_order(
        AGENTS[3], "MKT-01", "sell", "limit", 4, 20
    )["ok"] is False


@pytest.mark.parametrize("blocked_by", ["pause", "stale"])
def test_pause_or_stale_keeps_resting_orders_but_allows_owner_cancellation(
    venue, blocked_by,
):
    exchange, conn, clock, settings = venue
    order = _submit(exchange, AGENTS[0], "sell", 10, 40)
    if blocked_by == "pause":
        exchange.apply_market_observation(_observation(conn, status="inactive"))
    else:
        clock["now"] += settings.market_stale_after_seconds + 1

    assert exchange.close_due_markets() == 0
    assert exchange.submit_order(
        AGENTS[1], "MKT-01", "buy", "limit", 10, 40
    )["ok"] is False
    assert conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0
    assert exchange.get_account(AGENTS[0])["reserved_cents"] == 600
    assert exchange.cancel_order(AGENTS[0], order["order_id"])["ok"] is True
    assert exchange.get_account(AGENTS[0])["reserved_cents"] == 0


def test_unchanged_successful_poll_refreshes_freshness(venue):
    exchange, conn, clock, settings = venue
    clock["now"] += settings.market_stale_after_seconds
    assert exchange.assigned_markets(AGENTS[0])[0]["tradable"] is True
    clock["now"] += 1
    assert exchange.assigned_markets(AGENTS[0])[0]["tradable"] is False

    exchange.apply_market_observation(_observation(
        conn, last_successful_poll=clock["now"],
    ))

    assert exchange.assigned_markets(AGENTS[0])[0]["tradable"] is True
    _submit(exchange, AGENTS[0], "buy", 1, 20)


def test_agents_share_fixed_cohort_with_labeled_utc_times_and_no_external_prices(venue):
    exchange, conn, _, _ = venue
    expected_fields = {
        "ticker", "title", "rules", "status", "tradable", "settled",
        "payout_cents", "close_time", "expected_expiration_time",
        "latest_expiration_time",
    }
    markets = exchange.assigned_markets(AGENTS[0])
    assert len(markets) == 10
    assert [row["ticker"] for row in markets] == [
        f"MKT-{index:02d}" for index in range(1, 11)
    ]
    for agent in AGENTS:
        assert exchange.assigned_markets(agent) == markets
    for market in markets:
        assert set(market) == expected_fields
        assert market["tradable"] is True
        assert market["settled"] is False
    for name in (
        "close_time", "expected_expiration_time", "latest_expiration_time",
    ):
        value = markets[0][name]
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        assert parsed.tzinfo == timezone.utc
        assert int(parsed.timestamp()) == _market(1)[name]

    exchange.apply_market_observation(_observation(
        conn, title="Revised title", rules="Revised source rules.",
        expected_expiration_time=None, latest_expiration_time=None,
    ))
    revised = exchange.assigned_markets(AGENTS[0])
    assert revised[0]["title"] == "Revised title"
    assert revised[0]["rules"] == "Revised source rules."
    assert revised[0]["expected_expiration_time"] is None
    assert revised[0]["latest_expiration_time"] is None
    assert [row["ticker"] for row in revised] == [row["ticker"] for row in markets]


def test_all_settled_means_all_original_markets_and_leaves_ten_history_rows(venue):
    exchange, conn, _, settings = venue
    original_tickers = exchange.unsettled_tickers()
    assert len(original_tickers) == 10
    assert exchange.all_markets_settled() is False
    for index, ticker in enumerate(original_tickers):
        result = exchange.apply_market_observation(_observation(
            conn, ticker, status="finalized", payout_cents=100 * (index % 2),
        ))
        assert result["settled"] is True
        assert len(exchange.unsettled_tickers()) == 9 - index
        assert exchange.all_markets_settled() is (index == 9)

    assert exchange.unsettled_tickers() == []
    history = exchange.assigned_markets(AGENTS[0])
    assert [row["ticker"] for row in history] == original_tickers
    assert all(row["settled"] and not row["tradable"] for row in history)
    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM market_settlements").fetchone()[0] == 10
    _assert_conservation(conn, 8 * settings.initial_cash_cents)
