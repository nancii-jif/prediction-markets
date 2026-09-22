"""Trace exports preserve dispatch timing, causal book changes and full depth."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from prediction_markets.analysis.export import Filters, TraceExport, main, parse_time, utc


def order_args(action="buy", quantity=10, price=40, ticker="MKT-01", order_type="limit"):
    args = {"ticker": ticker, "action": action, "order_type": order_type, "quantity": quantity}
    if order_type == "limit":
        args["price_cents"] = price
    return args


@pytest.fixture
def trace_rig(participant_rig, tmp_path):
    rig = participant_rig
    participants = {}

    def fire(name, args=None, agent="agent-01"):
        if agent not in participants:
            participants[agent], _ = rig.make(agent_id=agent)
        asyncio.run(participants[agent]._handle_response(rig.response(rig.call(name, args))))

    return SimpleNamespace(rig=rig, path=tmp_path / "run.db", fire=fire)


def make_legacy(conn, *, remove_scheduling=False):
    conn.execute("DROP TABLE book_events")
    if remove_scheduling:
        conn.execute("DELETE FROM transcript_entries WHERE entry_type = 'timing'")
    else:
        conn.execute("DELETE FROM transcript_entries WHERE entry_type = 'timing' "
                     "AND json_extract(content_json, '$.event') = 'tool_dispatched'")
    conn.commit()


def archive_market(rig, ticker="MKT-01"):
    payload = {"ticker": ticker, "title": "Original market", "last_price_dollars": "0.3719",
               "unknown_nested_field": {"values": [None, "12.3400", 5]}}
    rig.conn.execute(
        "INSERT INTO market_api_responses VALUES (?, ?, ?, ?)",
        (ticker, rig.clock.now, rig.settings.kalshi_base_url + "/events", json.dumps(payload)),
    )
    rig.conn.commit()
    return payload


@pytest.mark.parametrize("legacy", [False, True])
def test_actions_use_dispatch_after_cooldown_not_model_or_result_time(participant_rig, tmp_path, legacy):
    rig = participant_rig
    start = rig.clock.now

    async def decide(request):
        rig.advance(30)
        return rig.response(rig.call("submit_order", order_args()))

    async def news(**kwargs):
        rig.advance(25)
        return {"ok": True, "results": []}

    participant, _ = rig.make(
        rig.response(rig.call("submit_order", order_args())), decide,
        rig.response(rig.call("search_news", {"query": "latest evidence"})),
        rig.response(rig.call("hold_for_now")),
    )
    participant.tools._async_functions["search_news"] = news
    with pytest.raises(rig.finished):
        asyncio.run(participant.run())
    if legacy:
        make_legacy(rig.conn)
    with TraceExport(tmp_path / "run.db") as trace:
        actions, books = list(trace.actions()), list(trace.orderbooks())
    assert [a["time"] for a in actions] == [utc(start + seconds) for seconds in (0, 120, 240, 840)]
    assert [a["tool_query"]["name"] for a in actions] == [
        "submit_order", "submit_order", "search_news", "hold_for_now",
    ]
    assert actions[2]["tool_query"]["arguments"] == {"query": "latest evidence"}
    assert [b["time"] for b in books] == [a["time"] for a in actions[:2]]
    assert [b["action_id"] for b in books] == [a["action_id"] for a in actions[:2]]
    assert {a["time_source"] for a in actions} == {"action_scheduled" if legacy else "tool_dispatched"}


@pytest.mark.parametrize("legacy", [False, True])
def test_export_only_fired_calls_including_errors_and_holds(participant_rig, tmp_path, legacy):
    rig = participant_rig
    participant, _ = rig.make()
    duplicate = rig.call("get_account")
    responses = [
        None,
        {"finish_reason": "tool_calls", "message": []},
        {"finish_reason": "tool_calls", "message": {"tool_calls": ["bad", {"id": []}]}},
        rig.response(duplicate, duplicate),  # invalid whole response
        rig.response(rig.call("unknown"), rig.call("get_account"), rig.call("get_own_orders")),
        rig.response(rig.call("cancel_order", {"order_id": 999})),  # dispatched business error
        rig.response(rig.call("hold_for_now", {"memory_note": 12})),  # invalid, not dispatched
        rig.response(rig.call("hold_until_next_day", {"memory_note": "wait"})),
    ]
    for response in responses:
        asyncio.run(participant._handle_response(response))
    asyncio.run(participant._handle_response(rig.response(rig.call("get_account")), allow_dispatch=False))
    if legacy:
        make_legacy(rig.conn)
    with TraceExport(tmp_path / "run.db") as trace:
        assert [a["tool_query"]["name"] for a in trace.actions()] == [
            "get_account", "cancel_order", "hold_until_next_day",
        ]
        assert list(trace.orderbooks()) == []


def test_cancelled_news_still_has_dispatch_and_does_not_misattribute_other_tasks(participant_rig, tmp_path):
    rig = participant_rig
    researcher, _ = rig.make()
    trader, _ = rig.make(agent_id="agent-02")

    async def scenario():
        started = asyncio.Event()

        async def news(**kwargs):
            started.set()
            await asyncio.Future()

        researcher.tools._async_functions["search_news"] = news
        task = asyncio.create_task(researcher._handle_response(
            rig.response(rig.call("search_news", {"query": "facts"}))))
        await started.wait()
        rig.advance(7)
        await trader._handle_response(rig.response(rig.call("submit_order", order_args())))
        rig.advance(2)
        rig.exchange.cancel_all_orders()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    with TraceExport(tmp_path / "run.db") as trace:
        actions, books = list(trace.actions()), list(trace.orderbooks())
    assert [a["agent_id"] for a in actions] == ["agent-01", "agent-02"]
    assert actions[0]["tool_query"]["name"] == "search_news"
    assert books[0]["agent_id"] == "agent-02"
    assert books[0]["action_id"] == actions[1]["action_id"]
    assert books[1]["agent_id"] is None and books[1]["action_id"] is None
    assert books[1]["tool"] == "run_shutdown"


@pytest.mark.parametrize("legacy", [False, True])
def test_full_depth_same_second_fills_cancels_and_filters(trace_rig, legacy):
    fixture = trace_rig
    rig, fire = fixture.rig, fixture.fire
    start = rig.clock.now
    # Seven levels, two orders at the best price: no depth truncation or aggregation.
    for price in (30, 31, 32, 33, 34, 35, 36, 36):
        fire("submit_order", order_args(price=price))
    rig.advance(10)
    fire("submit_order", order_args("sell", 12, order_type="market"), agent="agent-02")
    rig.advance(10)
    fire("cancel_order", {"order_id": 8})
    # A market order with no counterparty never becomes a resting quote.
    fire("submit_order", order_args(ticker="MKT-02", order_type="market"))
    if legacy:
        make_legacy(rig.conn)
    with TraceExport(fixture.path) as trace:
        books = list(trace.orderbooks())
        assert len(books) == 10
        assert len({b["event_id"] for b in books[:8]}) == 8
        assert {b["time"] for b in books[:8]} == {utc(start)}
        assert [b["order_id"] for b in books[7]["bids"]] == [7, 8, 6, 5, 4, 3, 2, 1]
        assert books[8]["bids"][0] == {"order_id": 8, "agent_id": "agent-01", "price_cents": 36, "quantity": 8}
        assert len(books[8]["bids"]) == 7
        assert len(books[9]["bids"]) == 6 and books[9]["asks"] == []
        filtered = list(trace.orderbooks(Filters(agent="agent-02", tool="submit_order", start=start+10, end=start+20)))
        assert filtered == books[8:9]
        assert {bid["agent_id"] for bid in filtered[0]["bids"]} == {"agent-01"}
        assert len(list(trace.actions(Filters(tool="cancel_order", market="MKT-01")))) == 1
        assert len(list(trace.actions(Filters(start=start+10, end=start+20)))) == 1
        assert trace.metadata()["book_history_complete"]


def test_lifecycle_cancellations_are_journaled_and_empty_transactions_are_not(trace_rig):
    rig, fire = trace_rig.rig, trace_rig.fire
    for ticker in ("MKT-01", "MKT-02", "MKT-03"):
        fire("submit_order", order_args(ticker=ticker))
    rig.conn.execute("UPDATE markets SET close_time = ? WHERE ticker = 'MKT-01'", (rig.clock.now,))
    rig.conn.commit()
    assert rig.exchange.close_due_markets() == 1
    assert rig.exchange.close_due_markets() == 0
    row = dict(rig.conn.execute("SELECT * FROM markets WHERE ticker = 'MKT-02'").fetchone())
    row.pop("cohort_index")
    row.update(status="finalized", payout_cents=100)
    assert rig.exchange.apply_market_observation(row)["settled"]
    assert rig.exchange.cancel_all_orders() == 1
    assert rig.exchange.cancel_all_orders() == 0
    with TraceExport(trace_rig.path) as trace:
        books = list(trace.orderbooks())
        assert len(books) == 6
        assert [b["tool"] for b in books[-3:]] == ["market_closed", "market_observation", "run_shutdown"]
        assert all(b["bids"] == b["asks"] == [] for b in books[-3:])
        assert trace.metadata()["book_history_complete"]


def test_rejected_and_rolled_back_orders_never_emit_book_changes(trace_rig, monkeypatch):
    rig = trace_rig.rig
    trace_rig.fire("submit_order", order_args())

    def fail(_):
        raise RuntimeError("injected failure after matching")

    monkeypatch.setattr(rig.exchange, "_assert_funded", fail)
    with pytest.raises(RuntimeError, match="injected"):
        rig.exchange.submit_order("agent-02", "MKT-01", "sell", "market", 5)
    assert not rig.conn.in_transaction
    assert rig.conn.execute("SELECT COUNT(*) FROM book_events").fetchone()[0] == 1
    with TraceExport(trace_rig.path) as trace:
        books = list(trace.orderbooks())
        assert books[0]["bids"][0]["quantity"] == 10
        assert trace.metadata()["book_history_complete"]


@pytest.mark.parametrize("with_log", [False, True])
def test_legacy_shutdown_uses_host_log_offset_not_session_timezone(trace_rig, with_log):
    rig = trace_rig.rig
    trace_rig.fire("submit_order", order_args())
    order = dict(rig.conn.execute("SELECT * FROM orders").fetchone())
    start = rig.clock.now
    rig.advance(17)
    rig.exchange.cancel_all_orders()
    make_legacy(rig.conn, remove_scheduling=True)
    if with_log:
        def wall(timestamp):
            return datetime.fromtimestamp(timestamp, timezone(timedelta(hours=-7))).strftime("%Y-%m-%d %H:%M:%S")
        trace_rig.path.with_name("run.log").write_text(
            f"{wall(start)},123 INFO prediction_markets.exchange: order accepted: {order!r}\n"
            f"{wall(start+17)},789 INFO prediction_markets.runner: stopped: duration_elapsed\n"
        )
    with TraceExport(trace_rig.path) as trace:
        actions, books = list(trace.actions()), list(trace.orderbooks())
        assert actions[0]["time"] == utc(start) and actions[0]["time_source"] == "order_created"
        assert trace.metadata()["book_history_complete"] is with_log
        if with_log:
            assert len(books) == 2 and books[-1]["time"] == utc(start+17)
            assert books[-1]["tool"] == "run_shutdown" and books[-1]["bids"] == []
        else:
            assert len(books) == 1
            assert any("Cancellation time unavailable" in warning for warning in trace.warnings)


@pytest.mark.parametrize("format", ["json", "jsonl"])
def test_cli_writes_standard_json_readonly_and_refuses_overwrite(trace_rig, tmp_path, format, capsys):
    trace_rig.fire("submit_order", order_args())
    payload = archive_market(trace_rig.rig)
    before = list(trace_rig.rig.conn.iterdump())
    output = tmp_path / "export"
    argv = [str(trace_rig.path), "--out", str(output), "--format", format,
            "--agent", "agent-01", "--tool", "submit_order", "--start", utc(trace_rig.rig.clock.now)]
    assert main(argv) == 0
    objects = (output / f"actions.{format}").read_text()
    actions = json.loads(objects) if format == "json" else [json.loads(line) for line in objects.splitlines()]
    assert len(actions) == 1 and actions[0]["tool_query"]["name"] == "submit_order"
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["counts"] == {"actions": 1, "orderbooks": 1, "markets": 1}
    assert json.loads((output / "markets.json").read_text()) == [payload]
    assert metadata["warnings"] == []
    assert main(argv) == 1
    assert "already exist" in capsys.readouterr().err
    assert (output / f"actions.{format}").read_text() == objects
    assert list(trace_rig.rig.conn.iterdump()) == before


def test_time_filter_validation():
    assert parse_time("2026-09-19T08:00:00-04:00") == parse_time("2026-09-19T12:00:00Z")
    assert Filters(start="1789819200", end="1789819201").start == 1789819200
    for value in ("2026-09-19T08:00:00", "nan", "inf", "not a date"):
        with pytest.raises(ValueError):
            Filters(start=value)
    with pytest.raises(ValueError, match="start must precede end"):
        Filters(start=10, end=10)


def test_market_exports_preserve_raw_fields_and_follow_filters(trace_rig, monkeypatch):
    from prediction_markets.integrations import kalshi as stream
    monkeypatch.setattr(stream, "fetch_market_payloads", lambda *a, **k: pytest.fail("unexpected HTTP"))
    rig, fire = trace_rig.rig, trace_rig.fire
    first = archive_market(rig)
    second = archive_market(rig, "MKT-02")
    start = rig.clock.now
    fire("submit_order", order_args())
    rig.advance(20)
    fire("submit_order", order_args(ticker="MKT-02"), agent="agent-02")
    rig.advance(10)
    fire("cancel_order", {"order_id": 2}, agent="agent-02")
    with TraceExport(trace_rig.path) as trace:
        assert list(trace.markets(Filters(agent="agent-01"))) == [first]
        assert list(trace.markets(Filters(agent="agent-02", start=start+20, end=start+30))) == [second]
        assert list(trace.markets(Filters(tool="cancel_order"))) == [second]
        assert list(trace.markets(Filters(market="MKT-01"))) == [first]
        assert trace.metadata()["market_payloads"] == {
            "sources": {"MKT-01": {"source": "run_first_pull", "captured_at": utc(start),
                                    "source_url": rig.settings.kalshi_base_url + "/events"}},
            "missing_tickers": [],
        }
        # Default includes the full cohort, but missing API objects stay missing.
        assert list(trace.markets()) == [first, second]
        assert len(trace.metadata()["market_payloads"]["missing_tickers"]) == 8


def test_backfill_only_fetches_missing_objects_and_never_relabels_originals(trace_rig, monkeypatch):
    from prediction_markets.integrations import kalshi as stream
    rig = trace_rig.rig
    first = archive_market(rig)
    current = {"ticker": "MKT-02", "status": "finalized", "last_price_dollars": "0.9900"}
    requests = []

    def fetch(tickers, *, base_url):
        requests.append((tickers, base_url))
        return [current]

    monkeypatch.setattr(stream, "fetch_market_payloads", fetch)
    before = list(rig.conn.iterdump())
    filters = Filters(market=["MKT-01", "MKT-02"])
    with TraceExport(trace_rig.path) as trace:
        assert list(trace.markets(filters)) == [first]
        assert requests == []
        assert list(trace.markets(filters, fetch_missing=True)) == [first, current]
        assert list(trace.markets(filters, fetch_missing=True)) == [first, current]
        metadata = trace.metadata()
        assert metadata["market_payloads"]["sources"]["MKT-01"]["source"] == "run_first_pull"
        backfill = metadata["market_payloads"]["sources"]["MKT-02"]
        assert backfill["source"] == "fetched_at_export"
        assert parse_time(backfill["captured_at"]) is not None
        assert metadata["market_payloads"]["missing_tickers"] == []
        assert any("not historical" in warning for warning in metadata["warnings"])
    assert requests == [(["MKT-02"], rig.settings.kalshi_base_url)]
    assert list(rig.conn.iterdump()) == before


def test_legacy_raw_payloads_are_not_synthesized_and_fetch_failures_leave_no_output(trace_rig, monkeypatch, tmp_path, capsys):
    from prediction_markets.integrations import kalshi as stream
    trace_rig.rig.conn.execute("DROP TABLE market_api_responses")
    trace_rig.rig.conn.commit()
    with TraceExport(trace_rig.path) as trace:
        assert list(trace.markets(Filters(market="MKT-01"))) == []
        assert trace.metadata()["market_payloads"]["missing_tickers"] == ["MKT-01"]

    def fail(*args, **kwargs):
        raise ValueError("Kalshi unavailable")

    monkeypatch.setattr(stream, "fetch_market_payloads", fail)
    output = tmp_path / "failed-export"
    assert main([str(trace_rig.path), "--fetch-missing-markets", "--out", str(output)]) == 1
    assert "Kalshi unavailable" in capsys.readouterr().err
    assert not output.exists()


def test_market_filters_handle_dispatched_calls_with_invalid_values(trace_rig):
    trace_rig.fire("submit_order", order_args(ticker=["invalid"]))
    trace_rig.fire("cancel_order", {"order_id": {"invalid": 1}})
    with TraceExport(trace_rig.path) as trace:
        assert len(list(trace.actions())) == 2  # invoked, then rejected by exchange
        assert list(trace.actions(Filters(market="MKT-01"))) == []
        assert list(trace.markets(Filters(agent="agent-01"))) == []
