"""Continuation invariants with real accounting and no external requests."""

import asyncio
import fcntl
import json
import sqlite3
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from prediction_markets import config
from prediction_markets.storage import database as database
from prediction_markets import resume
from prediction_markets.runtime import runner as runner
from prediction_markets.markets import update as stream
from prediction_markets.integrations.openai_responses import request_body
from conftest import ScriptedTokenizer


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("responses_api", [False, True])
def test_private_history_and_provider_items_survive_restore_without_reexecuting(participant_rig, legacy, responses_api):
    rig = participant_rig
    settings = rig.settings
    if responses_api:
        settings = replace(settings, model_provider="openai", model_revision=None, model_name="gpt-5.6-luna",
                           openai_api="responses", openai_reasoning_effort="medium")
    agent, _ = rig.make(settings=settings)
    call = rig.call("submit_order", {"ticker": "MKT-01", "action": "buy", "order_type": "limit",
                                    "quantity": 5, "price_cents": 35})
    response = rig.response(call)
    if responses_api:
        response.update(responses_output=[
            {"type": "reasoning", "id": "rs_original", "summary": [], "encrypted_content": "opaque-private-state"},
            {"type": "function_call", "call_id": call["id"], "name": "submit_order",
             "arguments": call["function"]["arguments"], "id": "fc_original", "status": "completed"},
        ], usage={"output_tokens_details": {"reasoning_tokens": 211}})
    asyncio.run(agent._handle_response(response))
    asyncio.run(agent._handle_response(rig.response(rig.call("hold_until_next_day", {"memory_note": "my fair value is 43"}))))
    other, _ = rig.make(agent_id="agent-02")
    asyncio.run(other._handle_response(rig.response(rig.call("hold_for_now", {"memory_note": "another agent's secret"}))))
    original_account = rig.exchange.get_account("agent-01")
    sequence = agent._sequence
    if legacy:
        with rig.conn:
            rig.conn.execute("DELETE FROM agent_checkpoints")
    restored, fake = rig.make(settings=settings)
    restored.restore("parent")
    assert restored.private_note == "my fair value is 43"
    assert restored.order_tool_calls == {"MKT-01": 1}
    assert restored._sequence == sequence + 1
    assert restored.next_action_at == agent.next_action_at
    assert rig.exchange.get_account("agent-01") == original_account
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    assert fake.requests == []
    messages = restored.build_messages()
    assert "another agent's secret" not in json.dumps(messages)
    assert any(m["role"] == "tool" and json.loads(m["content"]).get("tool_call_executed") for m in messages)
    if responses_api:
        body = request_body(messages=messages, tools=restored.tools.schemas, model=restored.model,
                            max_tokens=512, sampling={}, reasoning_effort="medium")
        assert response["responses_output"][0] in body["input"]
        assert sum(item.get("type") == "function_call" and item["call_id"] == call["id"] for item in body["input"]) == 1
        assert sum(item.get("type") == "function_call_output" and item["call_id"] == call["id"] for item in body["input"]) == 1


@pytest.mark.parametrize("legacy", [False, True])
def test_interrupted_group_is_history_never_replayed(participant_rig, legacy):
    rig = participant_rig
    agent, _ = rig.make()
    call = rig.call("search_news", {"query": "an unfinished search"})
    response = rig.response(call)
    # The tool dispatch has started but has no saved result (e.g. canceled HTTP).
    agent._groups = [resume.assistant_message(response, rig.settings)]
    agent._groups = [agent._groups]
    agent._record_response(response)
    agent._save("timing", {"event": "tool_dispatched", "tool": "search_news", "tool_call_id": call["id"]})
    if legacy:
        with rig.conn:
            rig.conn.execute("DELETE FROM agent_checkpoints")
    restored, _ = rig.make()
    restored.restore("parent")
    messages = restored.build_messages()
    assert "unknown execution status" in json.dumps(messages)
    assert not any(m.get("tool_calls") for m in messages)
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_restore_counts_downtime_and_finishes_hold_before_next_session(participant_rig):
    rig = participant_rig
    rig.clock.now = int(datetime.fromisoformat("2026-09-19T15:30:00-04:00").timestamp())
    agent, _ = rig.make()
    asyncio.run(agent._prepare_inference())
    asyncio.run(agent._handle_response(rig.response(rig.call("hold_for_now", {"memory_note": "carry belief"}))))
    deadline = agent.next_action_at  # 16:30, after the session close.
    rig.advance(10 * 60)
    rig.clock.mono = 5000  # Different process's monotonic origin.
    restored, _ = rig.make()
    restored.restore("parent")
    asyncio.run(restored._prepare_inference())
    assert rig.clock.sleeps == [50 * 60, 15.5 * 3600]
    assert deadline < rig.clock.now
    assert rig.clock.now == int(datetime.fromisoformat("2026-09-20T08:00:00-04:00").timestamp())
    assert restored.private_note == "carry belief"
    assert restored.next_action_at is None


def test_restore_preserves_same_day_counters_and_resets_only_next_session(participant_rig):
    rig = participant_rig
    agent, _ = rig.make()
    asyncio.run(agent._prepare_inference())
    asyncio.run(agent._handle_response(rig.response(rig.call("get_account"))))
    rig.advance(600)
    restored, _ = rig.make()
    restored.restore("parent")
    asyncio.run(restored._prepare_inference())
    assert restored.shared_tool_calls == 1 and not rig.clock.sleeps
    rig.advance(86400)
    again, _ = rig.make()
    again.restore("child")
    asyncio.run(again._prepare_inference())
    assert again.shared_tool_calls == 0


def stopped_source(rig):
    path = Path(rig.conn.execute("PRAGMA database_list").fetchone()[2])
    (path.parent / "run.log").write_text("2026-09-19 prediction_markets.runner: stopped: duration_elapsed\n")
    return path


def test_legacy_database_backup_preserves_source_and_all_ids(participant_rig, tmp_path):
    rig = participant_rig
    agent, _ = rig.make()
    asyncio.run(agent._handle_response(rig.response(rig.call("submit_order", {
        "ticker": "MKT-01", "action": "buy", "order_type": "limit", "price_cents": 30, "quantity": 5,
    }))))
    rig.exchange.cancel_all_orders()
    source = stopped_source(rig)
    with rig.conn:
        rig.conn.execute("DROP TABLE agent_checkpoints")
        rig.conn.execute("DROP TABLE run_segments")
    database.close()
    original = source.read_bytes()
    saved = resume.load_settings(source)
    assert saved == rig.settings
    child = database.fork(tmp_path / "child" / "run.db", source, run_settings=saved)
    assert source.read_bytes() == original
    order = child.execute("SELECT * FROM orders").fetchone()
    assert order["order_id"] == 1 and order["status"] == "canceled"
    assert child.execute("SELECT COUNT(*) FROM book_events").fetchone()[0] == 2
    assert child.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 10
    assert child.execute("SELECT COUNT(*) FROM agent_checkpoints").fetchone()[0] == 0
    assert child.execute("SELECT COUNT(*) FROM transcript_entries").fetchone()[0] > 0
    with sqlite3.connect(source) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='agent_checkpoints'").fetchone() is None


@pytest.mark.parametrize("change", [{"trading_day_start": "09:00"}, {"participant_count": 9},
                                   {"run_duration_seconds": 99}, {"hold_for_now_minutes": 5}])
def test_config_overrides_fail_before_copy_or_requests(participant_rig, tmp_path, monkeypatch, change):
    rig = participant_rig
    source = stopped_source(rig)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "children")

    async def unexpected(**request):
        pytest.fail("configuration mismatch must not make hosted requests")

    with pytest.raises(ValueError, match="entire saved configuration"):
        asyncio.run(runner.run(replace(rig.settings, **change), infer=unexpected,
                               tokenizer=ScriptedTokenizer(), resume_from=str(source)))
    assert not config.RUNS_DIR.exists()


def test_live_parent_lock_is_rejected(participant_rig):
    source = stopped_source(participant_rig)
    with (source.parent / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="running experiment"):
            resume.load_settings(source)
    assert resume.load_settings(source) == participant_rig.settings


def test_source_snapshot_reads_committed_wal_without_touching_source(participant_rig):
    rig = participant_rig
    source = stopped_source(rig)
    wal = Path(f"{source}-wal")
    with rig.conn:
        rig.conn.execute("UPDATE agent_state SET private_note='committed in WAL' WHERE agent_id='agent-01'")
    before = {path.name: path.read_bytes() for path in source.parent.iterdir() if path.is_file()}
    assert wal.stat().st_size > 0
    with resume.source_snapshot(source) as conn:
        assert conn.execute("SELECT private_note FROM agent_state WHERE agent_id='agent-01'").fetchone()[0] == "committed in WAL"
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("UPDATE agent_state SET private_note='bad'")
    after = {path.name: path.read_bytes() for path in source.parent.iterdir() if path.is_file()}
    assert after == before


def test_legacy_unfinished_run_is_rejected(participant_rig):
    source = stopped_source(participant_rig)
    (source.parent / "run.log").write_text("still trading")
    with pytest.raises(ValueError, match="stopped run"):
        resume.load_settings(source)


@pytest.mark.parametrize("payloads", [[], OSError("unavailable")])
def test_failed_resume_poll_never_trusts_old_freshness(participant_rig, monkeypatch, payloads):
    rig = participant_rig

    def refresh(*args, **kwargs):
        if isinstance(payloads, Exception):
            raise payloads
        return payloads

    monkeypatch.setattr(stream.kalshi_client, "get_markets_by_tickers", refresh)
    feed = stream.MarketFeed(rig.exchange, rig.settings, now=lambda: rig.clock.now)
    with pytest.raises(ValueError, match="fresh observations"):
        asyncio.run(feed.resume())
    assert not any(m["tradable"] for m in rig.exchange.assigned_markets("agent-01"))
