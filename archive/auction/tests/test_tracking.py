"""Live metrics: the payload shape, and that tracking never breaks a run.

No wandb, no network. `round_metrics` is deliberately separable from the
logging call so the whole dashboard contract is testable offline.
"""

import time

import pytest

from prediction_markets import auction, config, database, tracking

NOW = int(time.time())


@pytest.fixture
def db(tmp_path):
    database.init(tmp_path / "test.db")
    database.upsert_markets([{
        "ticker": "T1", "event_ticker": "EV-T1", "title": "Example market",
        "rules": "Example rules.", "sector": "Politics", "open_ts": NOW - 3600,
        "close_ts": NOW + 48 * 3600, "first_seen": NOW,
    }])
    for agent in ("a1", "a2"):
        database.register_agent(agent, config.INITIAL_CASH_CENTS)
    yield
    database.close()


def snapshot(yes_bid=0.40, yes_ask=0.44):
    database.write_snapshots([{
        "ticker": "T1", "captured_at": NOW, "yes_bid": yes_bid,
        "yes_ask": yes_ask, "bid_size": 10.0, "ask_size": 10.0, "last": 0.42,
        "volume_24h": 500.0, "status": "active", "result": "",
        "settlement_value": None,
    }])


CROSSED = [
    {"agent": "a1", "bid_cents": 50, "bid_size": 10, "ask_cents": 60, "ask_size": 10},
    {"agent": "a2", "bid_cents": 30, "bid_size": 2, "ask_cents": 45, "ask_size": 4},
]
UNCROSSED = [
    {"agent": "a1", "bid_cents": 40, "bid_size": 8, "ask_cents": 55, "ask_size": 9},
    {"agent": "a2", "bid_cents": 30, "bid_size": 2, "ask_cents": 60, "ask_size": 4},
]


def test_a_cleared_round_logs_price_volume_and_imbalance(db):
    result = auction.clear(CROSSED, None)
    metrics = tracking.round_metrics(1, [("T1", result, CROSSED)], [])

    assert metrics["market/T1/price"] == result.clear_cents
    # Everything that crossed: 10 bid at 50 >= 48, 4 offered at 45 <= 48.
    assert metrics["market/T1/bid_size"] == auction.demand(CROSSED, result.clear_cents)
    assert metrics["market/T1/ask_size"] == auction.supply(CROSSED, result.clear_cents)


def test_a_no_trade_round_logs_the_projected_price(db):
    """The continuous `price` series falls back to the indicative, so a market
    still draws a line through rounds where nothing traded."""
    result = auction.clear(UNCROSSED, None)
    metrics = tracking.round_metrics(2, [("T1", result, UNCROSSED)], [])

    assert result.clear_cents is None
    assert metrics["market/T1/price"] == result.indicative_cents
    # Nothing crossed, so the two best levels are the whole story.
    assert metrics["market/T1/bid_size"] == 8
    assert metrics["market/T1/ask_size"] == 9


def test_kalshi_mid_is_logged_in_cents_beside_the_internal_price(db):
    snapshot(yes_bid=0.40, yes_ask=0.44)
    result = auction.clear(CROSSED, None)
    metrics = tracking.round_metrics(1, [("T1", result, CROSSED)], [])

    assert metrics["market/T1/kalshi"] == 42


def test_a_market_with_no_snapshot_logs_no_kalshi_comparison(db):
    result = auction.clear(CROSSED, None)
    metrics = tracking.round_metrics(1, [("T1", result, CROSSED)], [])
    assert "market/T1/kalshi" not in metrics


def test_a_one_sided_kalshi_book_is_not_given_an_invented_mid(db):
    snapshot(yes_bid=0.40, yes_ask=None)
    result = auction.clear(CROSSED, None)
    metrics = tracking.round_metrics(1, [("T1", result, CROSSED)], [])
    assert "market/T1/kalshi" not in metrics


def test_agent_pnl_is_logged_per_agent(db):
    database.set_cash_cents("a1", 99_000)
    database.set_position("a1", "T1", 20, 1_000)
    metrics = tracking.round_metrics(1, [], ["a1", "a2"])

    assert metrics["agent/a1/cash"] == 99_000
    assert metrics["agent/a1/open"] == 1
    assert metrics["agent/a2/equity"] == config.INITIAL_CASH_CENTS
    assert set(metrics) == {"round"} | {
        f"agent/{a}/{m}" for a in ("a1", "a2")
        for m in ("equity", "cash", "realised", "unrealised", "open")
    }


def test_round_stats_are_logged_but_the_round_id_is_not_a_metric(db):
    metrics = tracking.round_metrics(
        3, [], [], {"round_id": 3, "quotes": 12, "auctions": 2, "seconds": 1.4}
    )
    assert metrics == {
        "round": 3,
        "round/quotes": 12, "round/auctions": 2, "round/seconds": 1.4,
    }


def test_metrics_are_yes_space_regardless_of_the_quote_frame(db, monkeypatch):
    """A no-run and a yes-run must be comparable on the same axes, so the
    frame is recorded in the W&B config rather than applied to the numbers."""
    result = auction.clear(CROSSED, None)
    yes = tracking.round_metrics(1, [("T1", result, CROSSED)], [])
    monkeypatch.setattr(config, "QUOTE_SIDE", "no")
    assert tracking.round_metrics(1, [("T1", result, CROSSED)], []) == yes


def test_logging_without_a_wandb_run_is_a_no_op_that_still_returns_the_payload(db):
    result = auction.clear(CROSSED, None)
    assert not tracking.enabled()
    metrics = tracking.log_round(1, [("T1", result, CROSSED)], ["a1"])
    assert metrics["market/T1/price"] == result.clear_cents


def test_tracking_stays_off_when_no_project_is_configured(monkeypatch):
    monkeypatch.setattr(config, "WANDB_PROJECT", None)
    assert tracking.start() is False
    assert tracking.enabled() is False


def test_the_round_is_logged_as_the_x_axis_for_every_series(db):
    """`round` is a metric in its own right so W&B plots against it rather than
    an implicit step counter — which is also what makes a market seated at
    round 12 start its line at 12."""
    result = auction.clear(CROSSED, None)
    metrics = tracking.round_metrics(12, [("T1", result, CROSSED)], ["a1"])
    assert metrics["round"] == 12


def test_exactly_four_series_are_logged_per_market(db):
    """W&B draws one panel per key, so the per-market payload is capped at the
    four that are actually read: price, Kalshi, and the size on each side."""
    snapshot()
    result = auction.clear(CROSSED, None)
    metrics = tracking.round_metrics(1, [("T1", result, CROSSED)], [])

    assert {k for k in metrics if k.startswith("market/")} == {
        "market/T1/price", "market/T1/kalshi",
        "market/T1/bid_size", "market/T1/ask_size",
    }


def test_depth_on_a_traded_round_is_the_whole_crossing_book_not_the_top(db):
    """The best level would report 10 x 4; what crossed at 48c is the same here
    only because one quote sits on each side — the metric is cumulative."""
    quotes = CROSSED + [
        {"agent": "a3", "bid_cents": 49, "bid_size": 7,
         "ask_cents": 60, "ask_size": 3},
    ]
    result = auction.clear(quotes, None)
    metrics = tracking.round_metrics(1, [("T1", result, quotes)], [])

    top = auction.top_of_book(quotes)
    assert metrics["market/T1/bid_size"] > top["bid_size"]
    assert metrics["market/T1/bid_size"] == auction.demand(quotes, result.clear_cents)


def test_a_market_with_no_quotes_logs_no_depth(db):
    result = auction.clear([], None)
    metrics = tracking.round_metrics(1, [("T1", result, [])], [])
    assert "market/T1/bid_size" not in metrics
    assert "market/T1/ask_size" not in metrics
