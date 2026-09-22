"""Fixed-cohort HTTP wiring and async scheduling, with no network requests."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from prediction_markets import config
from prediction_markets.storage import database as database
from prediction_markets.markets import update as stream
from prediction_markets.exchange import Exchange


NOW = 2_000_000_000


def _utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _payload(index, **changes):
    return {
        "ticker": f"MKT-{index:02d}",
        "event_ticker": f"EVENT-{index:02d}",
        "categories": ["Politics"],
        "title": f"Market {index}",
        "rules_primary": f"Market {index} settles according to its source.",
        "rules_secondary": "Additional rule.",
        "market_type": "binary",
        "status": "active",
        "open_time": _utc(NOW - 86_400),
        "close_time": _utc(NOW + 100_000 + index),
        "expected_expiration_time": _utc(NOW + 100 * 3_600),
        "latest_expiration_time": _utc(NOW + 110 * 3_600),
        "volume_24h_fp": "1000.00",
        "yes_bid_dollars": "0.4100",
        "yes_ask_dollars": "0.4500",
        "last_price_dollars": "0.4300",
        **changes,
    }


class RecordingClient:
    KalshiError = OSError

    def __init__(self, count=12):
        self.state = {
            row["ticker"]: row for row in (
                _payload(index) for index in range(1, count + 1)
            )
        }
        self.discoveries = []
        self.polls = []
        self.failure = None
        self.response = None

    def get_market_candidates(self, *, base_url):
        self.discoveries.append((base_url, threading.get_ident()))
        return list(self.state.values())

    def get_markets_by_tickers(self, tickers, *, base_url):
        self.polls.append((list(tickers), base_url, threading.get_ident()))
        if self.failure is not None:
            raise self.failure
        if self.response is not None:
            return self.response
        return [dict(self.state[ticker]) for ticker in tickers if ticker in self.state]


@pytest.fixture
def venue(tmp_path, monkeypatch):
    settings = replace(
        config.load(config.CONFIG_DIR / "mvp.yaml"),
        expected_expiration_min_hours=72, expected_expiration_max_hours=168,
    )
    conn = database.init(tmp_path / "run.db", run_settings=settings)
    clock = {"now": NOW}
    exchange = Exchange(conn, now=lambda: clock["now"])
    feed = stream.MarketFeed(exchange, settings, now=lambda: clock["now"])
    client = RecordingClient()
    monkeypatch.setattr(stream, "kalshi_client", client)
    try:
        yield feed, exchange, conn, clock, client, settings
    finally:
        database.close()


def _poll_times(conn):
    return dict(conn.execute("SELECT ticker, last_successful_poll FROM markets"))


def _fast_polling(feed, settings):
    # Keep real validated database settings; shorten only this timer's test input.
    feed.settings = SimpleNamespace(**{
        **settings.snapshot(), "market_poll_interval_seconds": 0.001,
    })


def test_startup_discovers_once_and_persists_no_external_prices(venue, monkeypatch):
    feed, exchange, conn, _, client, settings = venue
    main_thread = threading.get_ident()
    storage_threads = []
    original_store = database.store_market_cohort

    def record_storage(rows, **kwargs):
        storage_threads.append(threading.get_ident())
        return original_store(rows, **kwargs)

    monkeypatch.setattr(database, "store_market_cohort", record_storage)
    tickers = asyncio.run(feed.initialize())

    assert tickers == [f"MKT-{index:02d}" for index in range(1, 11)]
    assert len(client.discoveries) == 1
    assert client.discoveries[0][0] == settings.kalshi_base_url
    assert client.discoveries[0][1] != main_thread
    assert storage_threads == [main_thread]
    assert client.polls == []
    forbidden = {"yes_bid_dollars", "yes_ask_dollars", "last_price_dollars", "volume_24h_fp"}
    assert forbidden.isdisjoint(dict(conn.execute("SELECT * FROM markets LIMIT 1").fetchone()))
    for market in exchange.assigned_markets("agent-01"):
        assert forbidden.isdisjoint(market)

    with pytest.raises(RuntimeError, match="once"):
        asyncio.run(feed.initialize())
    replacement_feed = stream.MarketFeed(exchange, settings)
    with pytest.raises(RuntimeError, match="once"):
        asyncio.run(replacement_feed.initialize())
    assert len(client.discoveries) == 1


def test_startup_and_polling_leave_books_empty_and_cash_with_participants(venue):
    feed, exchange, conn, _, client, settings = venue
    tickers = asyncio.run(feed.initialize())
    # Real source quotes and whole-cent prices must never create local orders.
    for payload in client.state.values():
        payload.update(yes_bid_dollars="0.7900", yes_ask_dollars="0.8100", last_price_dollars="0.8000")
    asyncio.run(feed.poll_once())

    assert {row[0] for row in conn.execute("SELECT agent_id FROM accounts")} == set(settings.agent_ids)
    assert conn.execute("SELECT SUM(balance_cents) FROM accounts").fetchone()[0] == (
        settings.participant_count * settings.initial_cash_cents
    )
    for table in ("orders", "trades", "positions"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    for ticker in tickers:
        book = exchange.get_orderbook(ticker)
        assert book["bids"] == book["asks"] == []
        assert exchange.get_recent_trades(ticker)["trades"] == []

    # The first price and exposure come solely from two participants' orders.
    maker = exchange.submit_order("agent-01", tickers[0], "sell", "limit", 5, 61)
    assert maker["ok"] is True and maker["trades"] == []
    taker = exchange.submit_order("agent-02", tickers[0], "buy", "limit", 3, 64)
    assert taker["ok"] is True
    assert [(trade["price_cents"], trade["quantity"]) for trade in taker["trades"]] == [(61, 3)]
    assert exchange.get_orderbook(tickers[0])["asks"] == [{"price_cents": 61, "quantity": 2}]
    assert exchange.get_account("agent-01")["positions"][0]["signed_qty"] == -3
    assert exchange.get_account("agent-02")["positions"][0]["signed_qty"] == 3


def test_raw_first_pull_archive_preserves_original_fields_and_stays_out_of_agent_data(venue):
    feed, exchange, conn, clock, client, settings = venue
    raw = {**client.state["MKT-01"], "categories": ["Original API category"],
           "unknown_future_field": {"data": ["0.3719", None]}}
    client.state["MKT-01"]["_raw_market"] = raw
    asyncio.run(feed.initialize())
    saved = conn.execute("SELECT * FROM market_api_responses WHERE ticker = 'MKT-01'").fetchone()
    assert json.loads(saved["payload_json"]) == raw
    assert saved["captured_at"] == NOW and saved["source_url"] == settings.kalshi_base_url + "/events"
    assert conn.execute("SELECT COUNT(*) FROM market_api_responses").fetchone()[0] == settings.market_count
    assert conn.execute("SELECT COUNT(*) FROM market_api_responses WHERE ticker = 'MKT-12'").fetchone()[0] == 0
    client.state["MKT-01"].update(last_price_dollars="0.9000", title="Later title")
    clock["now"] += 120
    asyncio.run(feed.poll_once())
    assert json.loads(conn.execute("SELECT payload_json FROM market_api_responses WHERE ticker = 'MKT-01'").fetchone()[0]) == raw
    brief = json.dumps(exchange.assigned_markets("agent-01"))
    assert "Later title" in brief
    assert "unknown_future_field" not in brief and "last_price_dollars" not in brief


@pytest.mark.parametrize("response", [[None], [{"ticker": "OTHER"}], [{"ticker": "A"}, {"ticker": "A"}]])
def test_export_backfill_rejects_malformed_or_unexpected_payloads(monkeypatch, response):
    monkeypatch.setattr(stream.kalshi_client, "get_markets_by_tickers", lambda *a, **k: response)
    with pytest.raises(ValueError, match="malformed|unexpected|duplicate"):
        stream.fetch_market_payloads(["A"], base_url="https://example.test")


def test_insufficient_startup_cohort_fails_without_partial_storage(venue):
    feed, _, conn, _, client, _ = venue
    client.state = dict(list(client.state.items())[:9])

    with pytest.raises(ValueError, match="10"):
        asyncio.run(feed.initialize())

    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 0
    with pytest.raises(RuntimeError, match="once"):
        asyncio.run(feed.initialize())
    assert len(client.discoveries) == 1


@pytest.mark.parametrize("changes", [
    {"open_time": _utc(NOW + 1)},
    {"open_time": None},
    {"status": "inactive"},
    {"expected_expiration_time": _utc(NOW + 100 * 86_400),
     "latest_expiration_time": _utc(NOW + 101 * 86_400)},
    {"latest_expiration_time": _utc(NOW + 71 * 3_600)},
    {"expected_expiration_time": None, "latest_expiration_time": None},
    {"last_price_dollars": "0.09"},
    {"last_price_dollars": "0.91"},
    {"last_price_dollars": None},
])
def test_startup_never_fills_cohort_with_untradable_or_wrong_horizon_markets(venue, changes):
    feed, _, conn, _, client, _ = venue
    for index in range(1, 4):
        client.state[f"MKT-{index:02d}"].update(changes)
    with pytest.raises(ValueError, match="only 9 eligible markets.*need 10"):
        asyncio.run(feed.initialize())
    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 0
    assert client.polls == []


@pytest.mark.parametrize("expected_hours", [None, 200])
def test_startup_stores_earlier_latest_expiration_without_rewriting_fields(venue, expected_hours):
    feed, _, conn, _, client, _ = venue
    client.state["MKT-01"].update(
        expected_expiration_time=None if expected_hours is None else _utc(NOW + expected_hours * 3_600),
        latest_expiration_time=_utc(NOW + 100 * 3_600),
    )
    assert "MKT-01" in asyncio.run(feed.initialize())
    row = conn.execute(
        "SELECT expected_expiration_time, latest_expiration_time FROM markets WHERE ticker = 'MKT-01'"
    ).fetchone()
    assert tuple(row) == (None if expected_hours is None else NOW + expected_hours * 3_600,
                          NOW + 100 * 3_600)


def test_startup_excludes_sports_and_crypto_without_replacement_later(venue):
    feed, exchange, _, _, client, _ = venue
    client.state["MKT-01"].update(categories=["Sports"], volume_24h_fp="999999")
    client.state["MKT-02"].update(categories=["Crypto"], volume_24h_fp="999999")
    selected = asyncio.run(feed.initialize())
    assert selected == [f"MKT-{index:02d}" for index in range(3, 13)]
    # Lifecycle polling does not reselect a fixed cohort after reclassification.
    client.state["MKT-03"].update(categories=["Sports"], status="finalized", settlement_value_dollars="1.00")
    assert asyncio.run(feed.poll_once())["settled"] == ["MKT-03"]
    assert len(client.discoveries) == 1
    assert "MKT-01" not in exchange.unsettled_tickers()


def test_uncertainty_is_checked_on_first_pull_only(venue):
    feed, exchange, conn, clock, client, _ = venue
    client.state["MKT-01"]["last_price_dollars"] = "0.09"
    client.state["MKT-02"]["last_price_dollars"] = "0.91"
    selected = asyncio.run(feed.initialize())
    assert selected == [f"MKT-{index:02d}" for index in range(3, 13)]

    # Out-of-range or absent prices on later pulls never remove admitted markets;
    # rejected markets entering the band are never added to the fixed cohort.
    client.state["MKT-03"]["last_price_dollars"] = "0.99"
    del client.state["MKT-04"]["last_price_dollars"]
    client.state["MKT-01"]["last_price_dollars"] = "0.50"
    clock["now"] += 120
    result = asyncio.run(feed.poll_once())
    assert result["updated"] == selected
    assert exchange.unsettled_tickers() == selected
    assert len(client.discoveries) == 1 and client.polls[0][0] == selected
    assert set(_poll_times(conn).values()) == {clock["now"]}


def test_polling_never_refills_and_stops_fetching_settled_tickers(venue, monkeypatch):
    feed, exchange, conn, clock, client, settings = venue
    mutation_threads = []
    original_apply = exchange.apply_market_observation

    def record_apply(row):
        mutation_threads.append(threading.get_ident())
        return original_apply(row)

    monkeypatch.setattr(exchange, "apply_market_observation", record_apply)

    async def scenario():
        original = await feed.initialize()
        clock["now"] += 120
        first = await feed.poll_once()
        assert first == {"updated": original, "settled": [], "failed": False}
        assert set(_poll_times(conn).values()) == {clock["now"]}
        client.state["MKT-01"].update(
            status="finalized", settlement_value_dollars="1.0000",
        )
        clock["now"] += 120
        final = await feed.poll_once()
        assert final["settled"] == ["MKT-01"]
        clock["now"] += 120
        await feed.poll_once()
        assert client.polls[-1][0] == original[1:]
        assert _poll_times(conn)["MKT-01"] == NOW + 240
        assert [row["ticker"] for row in exchange.assigned_markets("agent-01")] == original

    asyncio.run(scenario())
    assert len(client.discoveries) == 1
    assert len(client.polls) == 3
    assert all(call[1] == settings.kalshi_base_url for call in client.polls)
    assert all(call[2] != threading.get_ident() for call in client.polls)
    assert set(mutation_threads) == {threading.get_ident()}
    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 10
    assert "MKT-11" not in exchange.unsettled_tickers()


def test_failed_http_does_not_refresh_data_but_close_cancels_orders(venue):
    feed, exchange, conn, clock, client, _ = venue

    async def scenario():
        await feed.initialize()
        order = exchange.submit_order("agent-01", "MKT-01", "buy", "limit", 10, 40)
        assert order["ok"] is True
        before = _poll_times(conn)
        clock["now"] = NOW + 100_001
        client.failure = OSError("source unavailable")

        assert await feed.poll_once() == {"updated": [], "settled": [], "failed": True}
        assert _poll_times(conn) == before
        assert exchange.get_account("agent-01")["reserved_cents"] == 0
        assert len(exchange.unsettled_tickers()) == 10

    asyncio.run(scenario())


def test_missing_malformed_duplicate_and_unselected_rows_do_not_refresh(venue):
    feed, exchange, conn, clock, client, settings = venue

    async def scenario():
        await feed.initialize()
        clock["now"] += settings.market_stale_after_seconds + 1
        client.response = [
            None, "invalid", {},
            {**client.state["MKT-01"], "close_time": "invalid timestamp"},
            dict(client.state["MKT-02"]),
            {**client.state["MKT-02"], "status": "finalized", "settlement_value_dollars": "1.00"},
            dict(client.state["MKT-11"]),
        ]
        result = await feed.poll_once()
        assert result == {"updated": ["MKT-02"], "settled": [], "failed": False}
        expected = {f"MKT-{index:02d}": NOW for index in range(1, 11)}
        expected["MKT-02"] = clock["now"]
        assert _poll_times(conn) == expected
        assert len(exchange.unsettled_tickers()) == 10
        markets = exchange.assigned_markets("agent-01")
        assert [row["ticker"] for row in markets if row["tradable"]] == ["MKT-02"]

    asyncio.run(scenario())


@pytest.mark.parametrize("value, expected", [
    ("0.0000", 0), ("0.3700", 37), ("1.0000", 100),
])
def test_payout_parser_preserves_exact_cents(value, expected):
    assert stream.payout_cents({"settlement_value_dollars": value}) == expected


@pytest.mark.parametrize("value", [
    None, "", "NaN", "Infinity", "-Infinity", "0.3750", "-0.01", "1.01",
    "not a decimal", "0.3700000000000000000000000000000001", "3.7e-1",
    0.37, 1, True,
])
def test_payout_parser_does_not_round_or_guess_from_result(value):
    assert stream.payout_cents({
        "ticker": "MKT-01", "settlement_value_dollars": value,
        "result": "yes", "settlement_value": 100,
    }) is None


def test_result_label_without_payout_is_not_authoritative():
    assert stream.payout_cents({"result": "yes", "settlement_value": 100}) is None
    assert stream.payout_cents({"result": "no"}) is None


@pytest.mark.parametrize("value", [None, "NaN", "0.3750"])
def test_unsupported_final_payout_stays_pending_and_keeps_polling(venue, value, caplog):
    feed, exchange, conn, _, client, _ = venue

    async def scenario():
        await feed.initialize()
        client.state["MKT-01"].update(
            status="finalized", settlement_value_dollars=value, result="yes",
        )
        result = await feed.poll_once()
        assert result["settled"] == []
        assert "MKT-01" in result["updated"]
        assert conn.execute("SELECT COUNT(*) FROM market_settlements").fetchone()[0] == 0
        assert exchange.assigned_markets("agent-01")[0]["tradable"] is False
        await feed.poll_once()
        assert "MKT-01" in client.polls[-1][0]

    asyncio.run(scenario())
    assert "payout" in caplog.text and "pending" in caplog.text


def test_run_exits_when_every_original_market_settles(venue):
    feed, exchange, conn, _, client, settings = venue
    _fast_polling(feed, settings)

    async def scenario():
        original = await feed.initialize()
        for ticker in original:
            client.state[ticker].update(status="finalized", settlement_value_dollars="0.3700")
        await asyncio.wait_for(feed.run(), timeout=0.5)
        assert exchange.all_markets_settled() is True
        assert exchange.unsettled_tickers() == []
        assert await feed.poll_once() == {"updated": [], "settled": [], "failed": False}
        await feed.run()

    asyncio.run(scenario())
    assert len(client.discoveries) == 1
    assert len(client.polls) == 1
    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 10
    assert conn.execute("SELECT COUNT(*) FROM market_settlements").fetchone()[0] == 10


class BlockingClient(RecordingClient):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def get_markets_by_tickers(self, tickers, *, base_url):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.set()
        try:
            assert self.release.wait(0.5), "test did not release the HTTP worker"
            return super().get_markets_by_tickers(tickers, base_url=base_url)
        finally:
            with self.lock:
                self.active -= 1
            self.returned.set()


def test_blocking_http_keeps_event_loop_available_and_serializes_polls(venue, monkeypatch):
    feed, exchange, _, _, _, _ = venue
    client = BlockingClient()
    monkeypatch.setattr(stream, "kalshi_client", client)

    async def scenario():
        await feed.initialize()
        first = asyncio.create_task(feed.poll_once())
        second = asyncio.create_task(feed.poll_once())
        try:
            assert await asyncio.to_thread(client.entered.wait, 0.5)
            # This local account read executes while the HTTP worker is blocked.
            await asyncio.sleep(0)
            assert exchange.get_account("agent-01")["cash_cents"] == 100_000
            assert client.active == 1
            assert not first.done() and not second.done()
        finally:
            client.release.set()
            await asyncio.wait_for(asyncio.gather(first, second), timeout=0.5)

    asyncio.run(scenario())
    assert len(client.polls) == 2
    assert client.max_active == 1


def test_canceled_poll_discards_late_http_response(venue, monkeypatch):
    feed, _, conn, clock, _, _ = venue
    client = BlockingClient()
    monkeypatch.setattr(stream, "kalshi_client", client)

    async def scenario():
        await feed.initialize()
        before = _poll_times(conn)
        clock["now"] += 120
        client.state["MKT-01"].update(status="finalized", settlement_value_dollars="1.00")
        task = asyncio.create_task(feed.poll_once())
        try:
            assert await asyncio.to_thread(client.entered.wait, 0.5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            client.release.set()
            assert await asyncio.to_thread(client.returned.wait, 0.5)
        await asyncio.sleep(0)
        assert _poll_times(conn) == before
        assert conn.execute("SELECT COUNT(*) FROM market_settlements").fetchone()[0] == 0

    asyncio.run(scenario())


def test_run_cancels_at_cutoff_while_http_is_still_pending(venue, monkeypatch):
    feed, exchange, conn, clock, _, settings = venue
    client = BlockingClient()
    client.state["MKT-01"]["close_time"] = _utc(NOW + 1)
    monkeypatch.setattr(stream, "kalshi_client", client)
    _fast_polling(feed, settings)
    real_wait = asyncio.wait
    calls = []

    async def expire_cutoff_or_finish(tasks, *, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            assert timeout == 1
            assert await asyncio.to_thread(client.entered.wait, 0.5)
            clock["now"] += 1
            return await real_wait(tasks, timeout=0)
        # The scheduler must cancel on the timer before accepting the HTTP result.
        assert client.active == 1
        assert conn.execute("SELECT status FROM orders").fetchone()[0] == "canceled"
        assert conn.execute("SELECT cancel_reason FROM orders").fetchone()[0] == "market_closed"
        assert exchange.get_account("agent-01")["reserved_cents"] == 0
        client.release.set()
        return await real_wait(tasks, timeout=0.5)

    monkeypatch.setattr(stream.asyncio, "wait", expire_cutoff_or_finish)

    async def scenario():
        original = await feed.initialize()
        assert exchange.submit_order("agent-01", "MKT-01", "buy", "limit", 10, 40)["ok"]
        for ticker in original:
            client.state[ticker].update(status="finalized", settlement_value_dollars="1.00")
        try:
            await asyncio.wait_for(feed.run(), timeout=0.5)
        finally:
            client.release.set()
        assert exchange.all_markets_settled() is True

    asyncio.run(scenario())
    assert len(calls) == 2
    assert len(client.polls) == 1
