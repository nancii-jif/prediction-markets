"""Ledger: quote validation, netting, settlement. Synthetic rows, no network."""

import pytest

from prediction_markets import config, database, ledger


@pytest.fixture
def db(tmp_path):
    database.init(tmp_path / "test.db")
    yield
    database.close()


def snapshot(ticker, result="", settlement_value=None, captured_at=1000):
    return {
        "ticker": ticker, "captured_at": captured_at, "yes_bid": 0.4,
        "yes_ask": 0.6, "bid_size": 10.0, "ask_size": 10.0, "last": 0.5,
        "volume_24h": 500.0, "status": "active", "result": result,
        "settlement_value": settlement_value,
    }


def quote(ticker, bid=40, bid_size=10, ask=60, ask_size=10):
    return {
        "ticker": ticker, "bid_cents": bid, "bid_size": bid_size,
        "ask_cents": ask, "ask_size": ask_size,
    }


# --- validation ------------------------------------------------------------


def test_crossed_quote_is_rejected(db):
    ledger.register_agents(["a"])
    ok, reason = ledger.validate_quote_set("a", [quote("T", bid=60, ask=40)])
    assert not ok and "bid < ask" in reason


def test_equal_bid_and_ask_is_rejected(db):
    ledger.register_agents(["a"])
    ok, reason = ledger.validate_quote_set("a", [quote("T", bid=50, ask=50)])
    assert not ok and "bid < ask" in reason


@pytest.mark.parametrize("size", [0, -1, config.MAX_LOT + 1])
def test_sizes_outside_the_lot_limit_are_rejected(db, size):
    ledger.register_agents(["a"])
    ok, reason = ledger.validate_quote_set("a", [quote("T", bid_size=size)])
    assert not ok and "bid_size" in reason


def test_quote_that_would_breach_the_position_cap_is_rejected(db):
    ledger.register_agents(["a"])
    database.set_position("a", "T", config.MAX_POSITION - 5, 0)
    ok, reason = ledger.validate_quote_set("a", [quote("T", bid_size=50)])
    assert not ok and "cap" in reason


def test_a_short_needs_no_collateral(db):
    """An ask of 50 contracts carries a 5,000c obligation if the market
    resolves YES. It is admitted against 1c of cash: shorts are uncollateralised
    by design, and both cash and equity are allowed to go negative."""
    ledger.register_agents(["a"])
    database.set_cash_cents("a", 1)
    ok, reason = ledger.validate_quote_set("a", [quote("T", ask=50, ask_size=50)])
    assert ok and reason is None


def test_an_existing_short_does_not_restrict_a_later_quote(db):
    ledger.register_agents(["a"])
    database.set_cash_cents("a", 0)
    database.set_position("a", "OTHER", -30, -1_500)

    ok, _ = ledger.validate_quote_set("a", [quote("T", ask=50, ask_size=20)])
    assert ok


def test_a_null_position_cap_admits_any_size(db, monkeypatch):
    monkeypatch.setattr(config, "MAX_POSITION", None)
    ledger.register_agents(["a"])
    database.set_position("a", "T", 10_000, 0)
    ok, reason = ledger.validate_quote_set("a", [quote("T", bid_size=config.MAX_LOT)])
    assert ok and reason is None


def test_rejection_is_all_or_nothing(db):
    """A bad market poisons the whole set rather than being dropped from it."""
    ledger.register_agents(["a"])
    quotes = [quote("T1"), quote("T2", bid=70, ask=30), quote("T3")]
    ok, reason = ledger.validate_quote_set("a", quotes)
    assert not ok and "T2" in reason


# --- fills and netting -----------------------------------------------------


def test_bid_fill_buys_and_ask_fill_sells(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", +10, 60)
    assert database.position("a", "T") == (10, 600)
    assert database.cash_cents("a") == config.INITIAL_CASH_CENTS - 600

    ledger.apply_fill("a", "T", -4, 70)
    qty, basis = database.position("a", "T")
    assert qty == 6
    assert basis == 360  # 600 scaled to the 6 contracts still held
    assert database.cash_cents("a") == config.INITIAL_CASH_CENTS - 600 + 280


def test_selling_without_inventory_opens_a_short(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", -10, 60)
    assert database.position("a", "T") == (-10, -600)
    assert database.cash_cents("a") == config.INITIAL_CASH_CENTS + 600


def test_position_can_flip_through_zero(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", +10, 60)
    ledger.apply_fill("a", "T", -15, 70)
    qty, basis = database.position("a", "T")
    assert qty == -5
    assert basis == -350  # the new short, at the price it was opened at
    # 10 bought at 60 and sold at 70 is 100c realised.
    assert ledger.pnl("a")["realized_cents"] == 100


# --- settlement ------------------------------------------------------------


def test_settlement_pays_a_yes_resolution(db):
    ledger.register_agents(["long", "short"])
    ledger.apply_fill("long", "T", +10, 60)
    ledger.apply_fill("short", "T", -10, 60)
    database.write_snapshots([snapshot("T", result="yes")])

    ledger.settle(round_id=1)

    assert database.cash_cents("long") == config.INITIAL_CASH_CENTS - 600 + 1_000
    assert database.cash_cents("short") == config.INITIAL_CASH_CENTS + 600 - 1_000
    assert database.position("long", "T") == (0, 0)


def test_settlement_pays_a_no_resolution(db):
    ledger.register_agents(["long", "short"])
    ledger.apply_fill("long", "T", +10, 60)
    ledger.apply_fill("short", "T", -10, 60)
    database.write_snapshots([snapshot("T", result="no")])

    ledger.settle(round_id=1)

    assert database.cash_cents("long") == config.INITIAL_CASH_CENTS - 600
    assert database.cash_cents("short") == config.INITIAL_CASH_CENTS + 600


def test_scalar_settlement_pays_the_partial_value(db):
    """Hand-computed: 10 contracts bought at 60c, settling at 55c.
    Long pays 600c, receives 550c, so is down 50c."""
    ledger.register_agents(["long", "short"])
    ledger.apply_fill("long", "T", +10, 60)
    ledger.apply_fill("short", "T", -10, 60)
    database.write_snapshots([snapshot("T", result="scalar", settlement_value=0.55)])

    settled = ledger.settle(round_id=1)

    assert {s["agent"]: s["payout_cents"] for s in settled} == {
        "long": 550, "short": -550,
    }
    assert database.cash_cents("long") == config.INITIAL_CASH_CENTS - 50
    assert database.cash_cents("short") == config.INITIAL_CASH_CENTS + 50
    assert all(row["scalar"] == 1 for row in database.settlements_for("long"))


def test_scalar_without_a_value_is_left_open_not_paid_zero(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", +10, 60)
    database.write_snapshots([snapshot("T", result="scalar", settlement_value=None)])

    assert ledger.settle(round_id=1) == []
    assert database.position("a", "T") == (10, 600)


def test_unresolved_markets_are_not_settled(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", +10, 60)
    database.write_snapshots([snapshot("T", result="")])

    assert ledger.settle(round_id=1) == []
    assert database.position("a", "T") == (10, 600)


def test_settlement_is_idempotent_across_a_rerun(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", +10, 60)
    database.write_snapshots([snapshot("T", result="yes")])

    ledger.settle(round_id=1)
    after_first = database.cash_cents("a")
    assert ledger.settle(round_id=1) == []
    assert database.cash_cents("a") == after_first
    assert len(database.settlements_for("a")) == 1


# --- pnl -------------------------------------------------------------------


def test_unrealised_is_marked_at_the_last_internal_clear(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", +10, 60)
    database.record_auction(
        {"ticker": "T", "round_id": 1, "ts": 0, "clear_cents": 75,
         "cleared_qty": 10, "tie_lo": 75, "tie_hi": 75},
        [],
    )
    result = ledger.pnl("a")
    assert result["unrealized_cents"] == 150  # (75 - 60) * 10
    assert result["realized_cents"] == 0


def test_position_without_a_print_carries_no_unrealised_mark(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", +10, 60)
    result = ledger.pnl("a")
    assert result["marks"] == {"T": None}
    assert result["unrealized_cents"] == 0


# --- voiding ghost markets -------------------------------------------------


def live(ticker, sector="Politics"):
    """Make a ticker an active live market."""
    database.upsert_markets([{
        "ticker": ticker, "event_ticker": f"EV-{ticker}", "title": "t",
        "rules": "r", "sector": sector, "open_ts": 0, "close_ts": 10**10,
        "first_seen": 0,
    }])
    database.add_live_market(ticker, sector, 0)


def printed(ticker, clear_cents, round_id=1):
    database.record_auction(
        {"ticker": ticker, "round_id": round_id, "ts": 0,
         "clear_cents": clear_cents, "cleared_qty": 10,
         "tie_lo": clear_cents, "tie_hi": clear_cents}, [],
    )


def test_ghost_market_is_voided_at_the_last_internal_clear(db):
    """Retired on the clock with no result: positions close at our own last
    printed price, which is the mark agents were already shown."""
    ledger.register_agents(["long", "short"])
    ledger.apply_fill("long", "T", +10, 50)
    ledger.apply_fill("short", "T", -10, 50)
    printed("T", 60)
    database.write_snapshots([snapshot("T", result="")])
    # Market was never admitted, so it is not live: a ghost.

    voided = ledger.void(round_id=3)

    assert {v["agent"]: v["payout_cents"] for v in voided} == {
        "long": 600, "short": -600,
    }
    assert database.position("long", "T") == (0, 0)
    assert database.cash_cents("long") == config.INITIAL_CASH_CENTS - 500 + 600
    assert database.cash_cents("short") == config.INITIAL_CASH_CENTS + 500 - 600
    assert all(row["void"] == 1 for row in database.settlements_for("long"))


def test_voiding_moves_no_net_cash(db):
    """Positions net to zero across agents, so one uniform price is neutral."""
    ledger.register_agents(["a", "b", "c"])
    ledger.apply_fill("a", "T", +7, 40)
    ledger.apply_fill("b", "T", -5, 40)
    ledger.apply_fill("c", "T", -2, 40)
    printed("T", 73)
    database.write_snapshots([snapshot("T", result="")])

    before = sum(database.cash_cents(x) for x in ("a", "b", "c"))
    ledger.void(round_id=1)
    after = sum(database.cash_cents(x) for x in ("a", "b", "c"))
    assert before == after


def test_live_markets_are_never_voided(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", +10, 50)
    printed("T", 60)
    live("T")
    database.write_snapshots([snapshot("T", result="")])

    assert ledger.void(round_id=1) == []
    assert database.position("a", "T") == (10, 500)


def test_a_retired_market_with_a_result_is_settled_not_voided(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", +10, 50)
    printed("T", 60)
    database.write_snapshots([snapshot("T", result="yes")])

    assert ledger.void(round_id=1) == []
    ledger.settle(round_id=1)
    assert database.cash_cents("a") == config.INITIAL_CASH_CENTS - 500 + 1_000
    assert all(row["void"] == 0 for row in database.settlements_for("a"))


def test_voiding_is_idempotent_across_a_rerun(db):
    ledger.register_agents(["a"])
    ledger.apply_fill("a", "T", +10, 50)
    printed("T", 60)
    database.write_snapshots([snapshot("T", result="")])

    ledger.void(round_id=1)
    cash = database.cash_cents("a")
    assert ledger.void(round_id=1) == []
    assert database.cash_cents("a") == cash
    assert len(database.settlements_for("a")) == 1
