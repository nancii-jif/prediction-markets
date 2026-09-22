"""The round loop, driven by scripted agents. No network, no LLM calls."""

import time

import pytest

from prediction_markets import agents, config, database, live_markets, runner

NOW = int(time.time())


class ScriptedAgent:
    """Stands in for an Agent: same interface, fixed quote, no model."""

    def __init__(self, agent_id, bid, ask, size=10):
        self.agent_id = agent_id
        self.quote_values = (bid, ask, size)

    async def quote(self, ticker, round_id):
        bid, ask, size = self.quote_values
        return {
            "agent": self.agent_id, "ticker": ticker,
            "bid_cents": bid, "bid_size": size,
            "ask_cents": ask, "ask_size": size,
        }


def market_row(ticker, sector="Politics", result="", captured_at=NOW,
               volume=1_000, mid=50):
    database.upsert_markets([{
        "ticker": ticker, "event_ticker": f"EV-{ticker}",
        "title": f"{ticker} title", "rules": f"{ticker} rules",
        "sector": sector, "open_ts": NOW - 3600,
        "close_ts": NOW + 48 * 3600, "first_seen": NOW,
    }])
    database.write_snapshots([{
        "ticker": ticker, "captured_at": captured_at,
        "yes_bid": (mid - 1) / 100, "yes_ask": (mid + 1) / 100,
        "bid_size": 10.0, "ask_size": 10.0, "last": mid / 100,
        "volume_24h": float(volume), "status": "active", "result": result,
        "settlement_value": None,
    }])


@pytest.fixture
def rig(tmp_path, monkeypatch):
    database.init(tmp_path / "test.db")
    monkeypatch.setattr(config, "SECTORS", ("Politics",))
    monkeypatch.setattr(config, "MARKETS_PER_SECTOR", 1)
    # The runner must never reach Kalshi; stream is the only module that may,
    # and here it does nothing at all.
    monkeypatch.setattr(runner.stream, "refresh_live", lambda: None)

    roster = [
        ScriptedAgent("buyer", bid=60, ask=65),
        ScriptedAgent("seller", bid=30, ask=40),
    ]
    monkeypatch.setattr(agents, "build_roster", lambda: list(roster))

    market_row("M1")
    database.add_live_market("M1", "Politics", NOW)
    yield roster
    database.close()


# --- a scripted run --------------------------------------------------------


def test_three_round_run_produces_complete_rows(rig):
    for round_id in (1, 2, 3):
        runner.run_round(round_id)

    auctions = database.auctions_for("M1")
    assert [a["round_id"] for a in auctions] == [1, 2, 3]
    # Bids of 60 against asks of 40 tie across 40..60; the first auction takes
    # the midpoint and later ones stay at the last clear.
    assert [a["clear_cents"] for a in auctions] == [50, 50, 50]
    assert all(a["cleared_qty"] == 10 for a in auctions)

    assert len(database.fills_for("buyer", "M1")) == 3
    assert database.position("buyer", "M1") == (30, 1_500)
    assert database.position("seller", "M1") == (-30, -1_500)
    assert database.cash_cents("buyer") == config.INITIAL_CASH_CENTS - 1_500
    assert database.cash_cents("seller") == config.INITIAL_CASH_CENTS + 1_500

    # Every call and quote is recorded, so the run reconstructs from the DB.
    quotes = database.quotes_for("buyer", "M1")
    assert [q["round_id"] for q in quotes] == [1, 2, 3]
    assert all(q["status"] == "accepted" for q in quotes)


def test_no_print_round_is_recorded(rig, monkeypatch):
    monkeypatch.setattr(agents, "build_roster", lambda: [
        ScriptedAgent("wide1", bid=20, ask=80),
        ScriptedAgent("wide2", bid=21, ask=79),
    ])
    runner.run_round(1)

    auction_row = database.auctions_for("M1")[0]
    assert auction_row["clear_cents"] is None
    assert auction_row["cleared_qty"] == 0
    assert database.fills_for("wide1", "M1") == []


def test_rejected_quote_set_is_recorded_and_excluded_from_the_auction(rig, monkeypatch):
    monkeypatch.setattr(agents, "build_roster", lambda: [
        ScriptedAgent("buyer", bid=60, ask=65),
        ScriptedAgent("crossed", bid=70, ask=30),
    ])
    runner.run_round(1)

    rejected = database.quotes_for("crossed", "M1")[0]
    assert rejected["status"] == "rejected"
    assert "bid < ask" in rejected["reject_reason"]
    # Only one valid quote was left, so nothing could cross.
    assert database.auctions_for("M1")[0]["clear_cents"] is None


# --- settlement timing -----------------------------------------------------


def test_settlement_lands_in_the_round_that_observes_the_resolution(rig):
    runner.run_round(1)
    assert database.position("buyer", "M1") == (10, 500)

    market_row("M1", result="yes", captured_at=NOW + 60)
    runner.run_round(2)

    settlements = database.settlements_for("buyer")
    assert len(settlements) == 1
    assert settlements[0]["round_id"] == 2
    assert settlements[0]["payout_cents"] == 1_000
    assert database.position("buyer", "M1") == (0, 0)
    # A resolved market is not quoted again in the round that settles it.
    assert not database.auction_exists("M1", 2)


def test_a_refilled_market_starts_with_empty_internal_history(rig):
    market_row("M2", volume=500)  # the reserve candidate
    runner.run_round(1)
    assert database.auctions_for("M1")[0]["cleared_qty"] == 10

    market_row("M1", result="yes", captured_at=NOW + 60)
    live_markets.maintain()
    assert database.active_in_sector("Politics") == ["M2"]

    brief = agents.render_brief("buyer", "M2", round_id=2)
    assert "No rounds have been held yet" in brief
    assert "You have not quoted this market before." in brief
    assert "Position in this market: flat" in brief


# --- idempotency -----------------------------------------------------------


def test_rerunning_a_round_duplicates_nothing(rig):
    runner.run_round(1)
    cash = database.cash_cents("buyer")
    position = database.position("buyer", "M1")

    runner.run_round(1)  # as if the process had been killed and restarted

    assert len(database.auctions_for("M1")) == 1
    assert len(database.fills_for("buyer", "M1")) == 1
    assert database.cash_cents("buyer") == cash
    assert database.position("buyer", "M1") == position


def test_next_round_id_resumes_an_incomplete_round(rig):
    market_row("M2")
    database.add_live_market("M2", "Politics", NOW)

    assert runner.next_round_id() == 1
    runner.run_round(1)
    assert runner.next_round_id() == 2  # both markets cleared

    # Simulate a crash after only one market of round 2 cleared.
    database.record_auction(
        {"ticker": "M1", "round_id": 2, "ts": NOW, "clear_cents": 50,
         "cleared_qty": 0, "tie_lo": 50, "tie_hi": 50}, [],
    )
    assert runner.next_round_id() == 2


# --- budget guard ----------------------------------------------------------


def test_runner_refuses_to_start_over_the_daily_call_budget(rig, monkeypatch):
    monkeypatch.setattr(config, "MAX_DAILY_CALLS", 1)
    with pytest.raises(SystemExit):
        runner.check_budget()


def test_budget_check_passes_under_the_ceiling(rig, monkeypatch):
    monkeypatch.setattr(config, "MAX_DAILY_CALLS", 10_000_000)
    assert runner.check_budget() > 0
