"""Membership: seed, retire on resolution only, refill from the same sector.

All synthetic — these never touch the network, which is the point: membership
is pure DB logic downstream of whatever stream last wrote.
"""

import time

import pytest

from prediction_markets import config, database, live_markets

NOW = int(time.time())


@pytest.fixture
def db(tmp_path, monkeypatch):
    database.init(tmp_path / "test.db")
    monkeypatch.setattr(config, "SECTORS", ("Politics", "Economics"))
    monkeypatch.setattr(config, "MARKETS_PER_SECTOR", 2)
    yield
    database.close()


def candidate(ticker, sector="Politics", mid=50, volume=1_000,
              hours_to_close=48, result="", captured_at=NOW,
              event=None, status="active"):
    """A market plus the snapshot that decides whether it qualifies.

    Each candidate gets its own event unless one is named, so tests that are
    not about event stratification are unaffected by it.
    """
    database.upsert_markets([{
        "ticker": ticker, "event_ticker": event or f"EV-{ticker}",
        "title": f"{ticker} title", "rules": f"{ticker} rules",
        "sector": sector, "open_ts": NOW - 3600,
        "close_ts": NOW + int(hours_to_close * 3600), "first_seen": NOW,
    }])
    database.write_snapshots([{
        "ticker": ticker, "captured_at": captured_at,
        "yes_bid": (mid - 1) / 100, "yes_ask": (mid + 1) / 100,
        "bid_size": 10.0, "ask_size": 10.0, "last": mid / 100,
        "volume_24h": float(volume), "status": status, "result": result,
        "settlement_value": None,
    }])


# --- seeding ---------------------------------------------------------------


def test_seed_fills_every_sector_to_target(db):
    for i in range(4):
        candidate(f"P{i}", "Politics", volume=1_000 - i)
        candidate(f"E{i}", "Economics", volume=1_000 - i)

    live_markets.maintain()

    assert len(database.active_in_sector("Politics")) == config.MARKETS_PER_SECTOR
    assert len(database.active_in_sector("Economics")) == config.MARKETS_PER_SECTOR
    # Ranked by 24h volume, so the two busiest win.
    assert database.active_in_sector("Politics") == ["P0", "P1"]


def test_seed_ranks_by_volume_with_lexicographic_tie_break(db):
    candidate("PB", "Politics", volume=500)
    candidate("PA", "Politics", volume=500)
    candidate("PC", "Politics", volume=499)

    live_markets.maintain()
    assert database.active_in_sector("Politics") == ["PA", "PB"]


@pytest.mark.parametrize("kwargs", [
    {"mid": 10},                  # below the uncertainty band
    {"mid": 90},                  # above the uncertainty band
    {"volume": 1},                # under the volume floor
    {"hours_to_close": 6},        # closes too soon
    {"hours_to_close": 400},      # closes too late
    {"result": "yes"},            # already resolved
    {"status": "initialized"},    # listed but not yet trading
    {"status": "inactive"},       # deactivated
    {"status": "determined"},     # resolved, awaiting settlement
])
def test_non_qualifying_candidates_are_not_admitted(db, kwargs):
    candidate("GOOD", "Politics", volume=100, mid=50)
    candidate("BAD", "Politics", **{"volume": 1_000, **kwargs})

    live_markets.maintain()
    assert "BAD" not in database.active_in_sector("Politics")
    assert "GOOD" in database.active_in_sector("Politics")


def test_short_sector_leaves_seats_open_rather_than_lowering_the_bar(db):
    candidate("P0", "Politics", volume=1_000)
    candidate("P1", "Politics", mid=5, volume=1_000)  # out of band

    result = live_markets.maintain()
    assert database.active_in_sector("Politics") == ["P0"]
    assert "Politics" in result["unfilled"]


# --- retirement ------------------------------------------------------------


def test_resolution_retires_and_refills_from_the_same_sector(db):
    candidate("P0", "Politics", volume=1_000)
    candidate("P1", "Politics", volume=900)
    candidate("P2", "Politics", volume=800)  # the reserve
    candidate("E0", "Economics", volume=1_000)
    live_markets.maintain()
    assert database.active_in_sector("Politics") == ["P0", "P1"]

    candidate("P0", "Politics", volume=1_000, result="yes", captured_at=NOW + 60)
    result = live_markets.maintain()

    assert result["retired"] == ["P0"]
    assert result["added"] == ["P2"]
    assert database.active_in_sector("Politics") == ["P1", "P2"]
    # Composition constant, identities rotated.
    assert len(database.active_in_sector("Politics")) == config.MARKETS_PER_SECTOR


def test_a_member_that_drifts_out_of_the_band_is_not_retired(db):
    """Drifting toward 0/100 is the market converging — the phase the exchange
    most needs to trade through, so admission criteria are never re-checked."""
    candidate("P0", "Politics", volume=1_000, mid=50)
    candidate("P1", "Politics", volume=900, mid=50)
    live_markets.maintain()

    # Mid collapses to 5c, volume dries up, and it drifts out of the window.
    candidate("P0", "Politics", volume=0, mid=5, hours_to_close=2,
              captured_at=NOW + 60)
    result = live_markets.maintain()

    assert result["retired"] == []
    assert "P0" in database.active_in_sector("Politics")


def test_maintain_without_resolutions_is_a_no_op(db):
    for sector in config.SECTORS:  # every sector full, so there is no deficit
        candidate(f"{sector}0", sector, volume=1_000)
        candidate(f"{sector}1", sector, volume=900)
    candidate("P0", "Politics", volume=5_000)
    candidate("P1", "Politics", volume=4_000)
    live_markets.maintain()
    before = database.active_in_sector("Politics")

    result = live_markets.maintain()
    assert result == {"retired": [], "added": [], "unfilled": []}
    assert database.active_in_sector("Politics") == before


def test_a_retired_market_is_never_readmitted(db):
    candidate("P0", "Politics", volume=1_000)
    candidate("P1", "Politics", volume=900)
    live_markets.maintain()

    candidate("P0", "Politics", volume=1_000, result="yes", captured_at=NOW + 60)
    live_markets.maintain()
    # P0 is now the highest-volume unresolved candidate again, but it has
    # already had its turn and carries history agents have seen.
    candidate("P0", "Politics", volume=9_999, result="", captured_at=NOW + 120)
    live_markets.maintain()

    assert "P0" not in database.active_in_sector("Politics")


# --- degraded operation ----------------------------------------------------


def test_failed_sweep_retires_without_refilling_then_recovers(db):
    candidate("P0", "Politics", volume=1_000)
    candidate("P1", "Politics", volume=900)
    candidate("P2", "Politics", volume=800)
    live_markets.maintain()

    candidate("P0", "Politics", volume=1_000, result="yes", captured_at=NOW + 60)
    degraded = live_markets.maintain(allow_refill=False)

    assert degraded["retired"] == ["P0"]
    assert degraded["added"] == []
    assert "Politics" in degraded["unfilled"]
    assert database.active_in_sector("Politics") == ["P1"]

    # The next successful sweep fills the seat, with no new retirement needed
    # to trigger it.
    recovered = live_markets.maintain(allow_refill=True)
    assert recovered["retired"] == []
    assert recovered["added"] == ["P2"]
    assert database.active_in_sector("Politics") == ["P1", "P2"]


# --- event stratification --------------------------------------------------


def test_seats_in_a_sector_come_from_distinct_events(db):
    """A Kalshi event lists many strikes on one question; seating several of
    them spends several seats on the same question."""
    candidate("U3-T4.0", "Politics", volume=9_000, event="U3-26AUG")
    candidate("U3-T4.1", "Politics", volume=8_000, event="U3-26AUG")
    candidate("U3-T4.2", "Politics", volume=7_000, event="U3-26AUG")
    candidate("PAYROLLS", "Politics", volume=1_000, event="PAY-26AUG")

    live_markets.maintain()

    # The busiest market represents its event; the other two strikes are
    # skipped in favour of the next event down the ranking.
    assert database.active_in_sector("Politics") == ["PAYROLLS", "U3-T4.0"]


def test_a_sector_short_of_distinct_events_leaves_seats_open(db):
    candidate("U3-T4.0", "Politics", volume=9_000, event="U3-26AUG")
    candidate("U3-T4.1", "Politics", volume=8_000, event="U3-26AUG")

    result = live_markets.maintain()
    assert database.active_in_sector("Politics") == ["U3-T4.0"]
    assert "Politics" in result["unfilled"]


def test_a_refill_cannot_land_on_a_sitting_members_event(db):
    candidate("A1", "Politics", volume=9_000, event="EV-A")
    candidate("B1", "Politics", volume=8_000, event="EV-B")
    candidate("B2", "Politics", volume=7_000, event="EV-B")  # same event as B1
    candidate("C1", "Politics", volume=1_000, event="EV-C")
    live_markets.maintain()
    assert database.active_in_sector("Politics") == ["A1", "B1"]

    candidate("A1", "Politics", volume=9_000, event="EV-A", result="yes",
              captured_at=NOW + 60)
    result = live_markets.maintain()

    # B2 outranks C1 on volume but its event already holds a seat.
    assert result["added"] == ["C1"]
    assert database.active_in_sector("Politics") == ["B1", "C1"]


def test_an_event_becomes_available_again_once_its_market_retires(db):
    candidate("B1", "Politics", volume=9_000, event="EV-B")
    candidate("B2", "Politics", volume=8_000, event="EV-B")
    live_markets.maintain()
    assert database.active_in_sector("Politics") == ["B1"]

    candidate("B1", "Politics", volume=9_000, event="EV-B", result="yes",
              captured_at=NOW + 60)
    result = live_markets.maintain()

    # The constraint is against simultaneous seats, not against the event ever
    # being used again.
    assert result["retired"] == ["B1"]
    assert result["added"] == ["B2"]


def test_a_market_with_no_recorded_event_is_not_admissible(db):
    database.upsert_markets([{
        "ticker": "ORPHAN", "event_ticker": None, "title": "t", "rules": "r",
        "sector": "Politics", "open_ts": NOW - 3600,
        "close_ts": NOW + 48 * 3600, "first_seen": NOW,
    }])
    database.write_snapshots([{
        "ticker": "ORPHAN", "captured_at": NOW, "yes_bid": 0.49,
        "yes_ask": 0.51, "bid_size": 10.0, "ask_size": 10.0, "last": 0.5,
        "volume_24h": 9_999.0, "status": "active", "result": "",
        "settlement_value": None,
    }])
    live_markets.maintain()
    assert "ORPHAN" not in database.active_in_sector("Politics")


# --- status ----------------------------------------------------------------


def test_only_active_markets_are_admitted_regardless_of_volume(db):
    """Volume alone does not establish tradeability: a market can carry stale
    24h volume while being initialized or deactivated."""
    candidate("BUSY-DEAD", "Politics", volume=99_999, status="inactive")
    candidate("BUSY-UNOPENED", "Politics", volume=99_999, status="initialized")
    candidate("QUIET-LIVE", "Politics", volume=200, status="active")

    live_markets.maintain()
    assert database.active_in_sector("Politics") == ["QUIET-LIVE"]


def test_status_is_checked_at_admission_only(db):
    """Consistent with every other admission criterion: a sitting member is
    not evicted for a status change short of resolution."""
    candidate("P0", "Politics", volume=1_000)
    live_markets.maintain()
    assert database.active_in_sector("Politics") == ["P0"]

    candidate("P0", "Politics", volume=1_000, status="inactive",
              captured_at=NOW + 60)
    result = live_markets.maintain()
    assert result["retired"] == []
    assert "P0" in database.active_in_sector("Politics")


# --- clock-based retirement ------------------------------------------------


def test_member_past_close_plus_grace_is_retired_without_a_result(db):
    """The motivating hole: a deactivated market keeps an empty result
    forever, so only the clock can ever free its seat."""
    candidate("P0", "Politics", volume=1_000)
    candidate("P1", "Politics", volume=900)
    candidate("P2", "Politics", volume=800)  # the reserve
    live_markets.maintain()
    assert database.active_in_sector("Politics") == ["P0", "P1"]

    # P0 goes inactive and its close time passes; result stays empty.
    candidate("P0", "Politics", volume=1_000, status="inactive",
              hours_to_close=-(config.RETIRE_GRACE_H + 1), captured_at=NOW + 60)
    result = live_markets.maintain()

    assert result["retired"] == ["P0"]
    assert result["added"] == ["P2"]
    assert database.active_in_sector("Politics") == ["P1", "P2"]


def test_member_past_close_but_within_grace_is_kept(db):
    candidate("P0", "Politics", volume=1_000)
    candidate("P1", "Politics", volume=900)
    live_markets.maintain()

    # Closed 1h ago, well inside the grace window: the usual gap between
    # close and determination, during which the market keeps its seat.
    candidate("P0", "Politics", volume=1_000, hours_to_close=-1,
              captured_at=NOW + 60)
    result = live_markets.maintain()

    assert result["retired"] == []
    assert "P0" in database.active_in_sector("Politics")


def test_result_still_retires_before_the_clock_does(db):
    candidate("P0", "Politics", volume=1_000)
    candidate("P1", "Politics", volume=900)
    live_markets.maintain()

    # Resolved early, close time still in the future: the result trigger
    # fires on its own.
    candidate("P0", "Politics", volume=1_000, result="yes", hours_to_close=40,
              captured_at=NOW + 60)
    result = live_markets.maintain()
    assert result["retired"] == ["P0"]
