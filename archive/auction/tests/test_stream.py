"""Stream: what it asks Kalshi for, and what it does when Kalshi says no.

Kalshi itself is never mocked in integration tests, but these are about the
*shape* of the request load and about failure paths that cannot be provoked
on demand against a live API, so the client is recorded here.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

from prediction_markets import config, database, stream

NOW = int(time.time())
ROOT = Path(__file__).resolve().parent.parent


class RecordingClient:
    """Stands in for kalshi_client, counting calls by entry point.

    One dict of ticker -> payload backs both entry points, so the two stay
    consistent the way the real API is: a sweep asks for status=open and
    therefore cannot return a market that has already resolved, while a
    targeted tickers= fetch returns whatever it is asked for.
    """

    KalshiError = RuntimeError

    def __init__(self, state=(), fail_sweep=False, fail_targeted=False):
        self.state = {m["ticker"]: m for m in state}
        self.fail_sweep = fail_sweep
        self.fail_targeted = fail_targeted
        self.calls = {"get_markets": 0, "get_markets_by_tickers": 0, "get_series": 0}
        self.last_tickers = []

    def resolve(self, ticker, **kwargs):
        self.state[ticker] = payload(ticker, result="yes", **kwargs)

    def get_series(self, **params):
        self.calls["get_series"] += 1
        return [{"ticker": "SER", "category": params.get("category")}]

    def get_markets(self, **params):
        self.calls["get_markets"] += 1
        if self.fail_sweep:
            raise RuntimeError("sweep failed")
        return [m for m in self.state.values() if not m["result"]]

    def get_markets_by_tickers(self, tickers):
        self.calls["get_markets_by_tickers"] += 1
        self.last_tickers = sorted(tickers)
        if self.fail_targeted:
            raise RuntimeError("targeted fetch failed")
        wanted = set(tickers)
        return [m for m in self.state.values() if m["ticker"] in wanted]


def payload(ticker, result="", volume=1_000, mid=50, hours=48, event=None):
    # One event per ticker by default: the series prefix ("SER") is what maps
    # to a sector, while the full event ticker is what stratification keys on,
    # so sharing an event across fakes would silently block refills.
    return {
        "ticker": ticker, "event_ticker": event or f"SER-{ticker}", "title": f"{ticker} title",
        "rules_primary": f"{ticker} rules", "status": "active", "result": result,
        "yes_bid_dollars": f"{(mid - 1) / 100:.4f}",
        "yes_ask_dollars": f"{(mid + 1) / 100:.4f}",
        "yes_bid_size_fp": "10.0", "yes_ask_size_fp": "10.0",
        "last_price_dollars": f"{mid / 100:.4f}", "volume_24h_fp": f"{volume:.1f}",
        "open_time": "2026-08-01T00:00:00Z",
        "close_time": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime(NOW + hours * 3600)
        ),
    }


@pytest.fixture
def db(tmp_path, monkeypatch):
    database.init(tmp_path / "test.db")
    monkeypatch.setattr(config, "SECTORS", ("Politics",))
    monkeypatch.setattr(config, "MARKETS_PER_SECTOR", 1)
    yield
    database.close()


def install(monkeypatch, client):
    monkeypatch.setattr(stream, "kalshi_client", client)
    return client


# --- request shape ---------------------------------------------------------


def test_routine_refresh_makes_only_a_targeted_fetch(db, monkeypatch):
    """The whole routine Kalshi load is one batched tickers= request."""
    client = install(monkeypatch, RecordingClient(
        [payload("M1"), payload("M2", volume=500)]
    ))

    stream.refresh_live()          # cold start: seeds, so it must sweep
    assert client.calls["get_markets"] == 1
    assert database.active_live_tickers() == ["M1"]

    before = dict(client.calls)
    stream.refresh_live()          # routine run, nothing resolved
    stream.refresh_live()

    assert client.calls["get_markets"] == before["get_markets"]  # no sweep
    assert client.calls["get_series"] == before["get_series"]
    assert client.calls["get_markets_by_tickers"] == before["get_markets_by_tickers"] + 2


def test_a_resolution_triggers_a_sweep_and_refill_in_the_same_run(db, monkeypatch):
    client = install(monkeypatch, RecordingClient(
        [payload("M1"), payload("M2", volume=500)]
    ))
    stream.refresh_live()
    assert database.active_live_tickers() == ["M1"]
    sweeps = client.calls["get_markets"]

    client.resolve("M1")
    stats = stream.refresh_live()

    assert stats["retired"] == ["M1"]
    assert stats["swept"] is True
    assert client.calls["get_markets"] == sweeps + 1
    # Retired and refilled inside the one run, off candidate data seconds old.
    assert database.active_live_tickers() == ["M2"]


def test_retired_markets_are_not_re_fetched(db, monkeypatch):
    """Only live members are fetched, even when a position is still open: the
    result was made durable in snapshots while the market was still live, and
    settlement reads it from there."""
    client = install(monkeypatch, RecordingClient(
        [payload("M1"), payload("M2", volume=500)]
    ))
    stream.refresh_live()
    database.set_position("a1", "M1", 10, 500)  # an open position in M1

    client.resolve("M1")
    stream.refresh_live()
    assert "M1" not in database.active_live_tickers()
    assert database.latest_snapshot("M1")["result"] == "yes"  # durable

    client.calls["get_markets_by_tickers"] = 0
    client.state["M2"] = payload("M2", volume=500)
    stream.refresh_live()

    # M1 is retired, so it is no longer asked for despite the open position.
    assert client.last_tickers == ["M2"]


# --- failure paths ---------------------------------------------------------


def test_targeted_fetch_failure_still_runs_maintenance(db, monkeypatch):
    client = install(monkeypatch, RecordingClient(
        [payload("M1"), payload("M2", volume=500)]
    ))
    stream.refresh_live()

    client.fail_targeted = True
    stats = stream.refresh_live()  # must not raise

    assert stats["fetched"] == 0
    assert database.active_live_tickers() == ["M1"]


def test_sweep_failure_retires_without_refill_then_recovers(db, monkeypatch):
    client = install(monkeypatch, RecordingClient(
        [payload("M1"), payload("M2", volume=500)]
    ))
    stream.refresh_live()

    client.fail_sweep = True
    client.resolve("M1")
    stats = stream.refresh_live()

    assert stats["sweep_failed"] is True
    assert database.active_live_tickers() == []  # retired, seat left open

    client.fail_sweep = False
    stream.refresh_live()
    assert database.active_live_tickers() == ["M2"]


def test_snapshots_are_written_only_when_a_tracked_field_moves(db, monkeypatch):
    """Most markets are quiet; writing only on movement is what keeps the
    snapshot table from growing by a row per ticker per refresh."""
    client = install(monkeypatch, RecordingClient([payload("M1")]))
    stream.refresh_live()

    assert stream.refresh_live()["written"] == 0  # nothing moved
    assert stream.refresh_live()["written"] == 0

    client.state["M1"] = payload("M1", mid=70)
    assert stream.refresh_live()["written"] == 1
    assert database.latest_snapshot("M1")["yes_bid"] == pytest.approx(0.69)


def test_out_of_sector_markets_are_not_stored(db, monkeypatch):
    install(monkeypatch, RecordingClient(
        [payload("M1"), payload("X1", event="OTHER-1")]
    ))
    stream.refresh_live()
    assert database.market("X1") is None
    assert database.market("M1") is not None


# --- concurrency -----------------------------------------------------------


WRITER = """
import sys, time
sys.path.insert(0, {root!r})
from prediction_markets import database
conn = database.connect({path!r})
conn.executescript(database.SCHEMA)
for i in range(300):
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO snapshots (ticker, captured_at, yes_bid, "
            "status, result) VALUES (?, ?, ?, 'active', '')",
            ("W1", 1000 + i, 0.5),
        )
conn.close()
"""


def test_a_writer_and_a_reader_do_not_collide(tmp_path):
    """WAL mode: the cron stream writes while the runner reads."""
    path = tmp_path / "concurrent.db"
    database.init(path)

    writer = subprocess.Popen(
        [sys.executable, "-c", WRITER.format(root=str(ROOT), path=str(path))],
        stderr=subprocess.PIPE, text=True,
    )
    reads = 0
    while writer.poll() is None:
        database.latest_snapshots(["W1"])
        database.active_live_tickers()
        reads += 1

    _, errors = writer.communicate()
    assert writer.returncode == 0, errors
    assert "database is locked" not in errors
    assert reads > 0
    assert database.snapshot_count() == 300
    database.close()


# --- clock-based retirement ------------------------------------------------


def test_close_time_amendments_reach_the_db(db, monkeypatch):
    """Routine fetches must refresh close_ts: the clock-based retirement rule
    reads it, and Kalshi amends close times after admission."""
    client = install(monkeypatch, RecordingClient([payload("M1", hours=48)]))
    stream.refresh_live()
    before = database.market("M1")["close_ts"]

    client.state["M1"] = payload("M1", hours=90)
    stream.refresh_live()
    after = database.market("M1")["close_ts"]

    assert after != before
    assert after - before == pytest.approx(42 * 3600, abs=2)


def test_overdue_member_triggers_sweep_retire_and_refill(db, monkeypatch):
    """A member past close + grace with no result (e.g. deactivated) demands
    a sweep exactly like a resolution, and its seat refills in the same run."""
    client = install(monkeypatch, RecordingClient(
        [payload("M1"), payload("M2", volume=500)]
    ))
    stream.refresh_live()
    assert database.active_live_tickers() == ["M1"]
    sweeps = client.calls["get_markets"]

    # Kalshi now reports M1 closed 30h ago, still with an empty result. The
    # refresh must first pull that close_ts into the DB, then retire on it,
    # and the resulting deficit is what pulls the sweep.
    client.state["M1"] = payload(
        "M1", hours=-(config.RETIRE_GRACE_H + 6), volume=1_000
    )
    stats = stream.refresh_live()

    assert stats["retired"] == ["M1"]
    assert client.calls["get_markets"] == sweeps + 1
    assert database.active_live_tickers() == ["M2"]


# --- retire and refill are independent -------------------------------------


def test_an_open_seat_pulls_a_sweep_even_with_nothing_retiring(db, monkeypatch):
    """A seat left open by an earlier failure must be refilled from a fresh
    sweep, not from whatever stale candidate rows happen to be in the DB."""
    client = install(monkeypatch, RecordingClient(
        [payload("M1"), payload("M2", volume=500)]
    ))
    monkeypatch.setattr(config, "MARKETS_PER_SECTOR", 2)

    # First run seeds M1 but only one candidate qualifies for the second seat
    # once M2 is filtered out, leaving the sector short.
    client.state["M2"] = payload("M2", volume=1)  # under the volume floor
    stream.refresh_live()
    assert database.active_live_tickers() == ["M1"]
    sweeps = client.calls["get_markets"]

    # Nothing resolves, nothing is overdue — but the seat is still open, so
    # the next run sweeps anyway and fills it from fresh data.
    client.state["M2"] = payload("M2", volume=500)
    stats = stream.refresh_live()

    assert stats["retired"] == []
    assert client.calls["get_markets"] == sweeps + 1
    assert stats["added"] == ["M2"]
    assert database.active_live_tickers() == ["M1", "M2"]


def test_a_full_roster_never_sweeps(db, monkeypatch):
    client = install(monkeypatch, RecordingClient(
        [payload("M1"), payload("M2", volume=500)]
    ))
    stream.refresh_live()
    sweeps = client.calls["get_markets"]

    for _ in range(3):
        stats = stream.refresh_live()
        assert stats["swept"] is False
    assert client.calls["get_markets"] == sweeps
