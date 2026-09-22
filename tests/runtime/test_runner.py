"""Continuous runtime acceptance with real SQLite and mocked external I/O."""

import asyncio
import json
import sqlite3
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from functools import partial
from types import SimpleNamespace

import httpx
import pytest

from prediction_markets import config
from prediction_markets.storage import database as database
from prediction_markets.integrations import model_client as model_client
from prediction_markets.runtime import runner as runner
from prediction_markets.markets import update as stream
from prediction_markets.agents import ContextLimitError, PermanentInferenceError, Participant
from conftest import ScriptedTokenizer


def response(*calls):
    return {"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": f"call-{i}", "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments),
        }} for i, (name, arguments) in enumerate(calls)],
    }}


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0)
    await asyncio.wait_for(wait(), 2)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    settings = replace(
        config.load(config.CONFIG_DIR / "mvp.yaml"), run_duration_seconds=1,
        expected_expiration_min_hours=72, expected_expiration_max_hours=168,
    )
    # Provider/exchange integration uses immediate intraday costs. Actual
    # elapsed scheduling and close boundaries are covered by test_timing.
    class IntegrationParticipant(Participant):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs,
                             now=lambda: int(datetime.fromisoformat("2026-09-19T08:00:00-04:00").timestamp()))

        async def _finish_cooldown(self):
            if self._deadline is not None and not self._defer_inference:
                self._deadline = self._monotonic()
            await super()._finish_cooldown()

    monkeypatch.setattr(runner, "Participant", IntegrationParticipant)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    now = int(time.time())

    def utc(timestamp):
        return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()

    candidates = [{
        "ticker": f"MKT-{i:02d}", "event_ticker": f"EVENT-{i:02d}",
        "categories": ["Politics"],
        "title": f"Market {i}", "rules_primary": f"Complete rules for market {i}.",
        "market_type": "binary", "status": "active", "volume_24h_fp": "1000.00",
        "open_time": utc(now - 86400),
        "close_time": utc(now + 100_000), "expected_expiration_time": utc(now + 100 * 3600),
        "yes_bid_dollars": "0.9900", "yes_ask_dollars": "0.9900",
        "last_price_dollars": "0.4300",
    } for i in range(1, 11)]
    discoveries = []

    def discover(**kwargs):
        discoveries.append(kwargs)
        return candidates

    def unexpected_poll(*args, **kwargs):
        raise AssertionError("the one-second test must finish before the 300s poll")

    monkeypatch.setattr(stream.kalshi_client, "get_market_candidates", discover)
    monkeypatch.setattr(stream.kalshi_client, "get_markets_by_tickers", unexpected_poll)

    def connect():
        conn = sqlite3.connect(tmp_path / "runs" / "test" / "run.db")
        conn.row_factory = sqlite3.Row
        return conn

    yield SimpleNamespace(settings=settings, candidates=candidates, discoveries=discoveries,
                          connect=connect, path=tmp_path / "runs" / "test")
    database.close()


def test_eight_http_loops_trade_then_shutdown_preserves_positions_and_private_notes(rig):
    calls = Counter()

    async def handler(request):
        body = json.loads(request.content)
        brief = json.loads(body["messages"][-1]["content"])
        agent = brief["account"]["agent_id"]
        calls[agent] += 1
        assert len(brief["markets"]) == 10
        assert "yes_bid_dollars" not in json.dumps(body)
        assert all(f"note-agent-{i:02d}" not in json.dumps(body) for i in range(1, 9) if f"agent-{i:02d}" != agent)
        if calls[agent] == 1:
            choice = response(("get_account", {}))
        elif calls[agent] == 2 and agent in {"agent-01", "agent-02", "agent-03"}:
            choice = response(("submit_order", {
                "ticker": "MKT-02" if agent == "agent-03" else "MKT-01",
                "action": "sell" if agent == "agent-01" else "buy", "order_type": "limit",
                "quantity": 10, "price_cents": 20 if agent == "agent-03" else 40,
            }))
        else:
            choice = response(("hold_until_next_day", {"memory_note": f"note-{agent}"}))
        return httpx.Response(200, json={"choices": [choice]})

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            infer = partial(model_client.infer, client=client, base_url="https://model.example/v1",
                            api_key="private-test-key", timeout_seconds=120)
            return await runner.run(rig.settings, infer=infer, tokenizer=ScriptedTokenizer(), run_id="test")

    summary = asyncio.run(go())
    assert summary["stop_reason"] == "duration_elapsed"
    assert summary["trade_count"] == 1 and summary["canceled_order_count"] == 1
    assert len(rig.discoveries) == 1 and len(calls) == 8
    assert all(count >= 2 for count in calls.values())
    assert all(account["reserved_cents"] == 0 for account in summary["accounts"])
    assert sum(a["cash_cents"] for a in summary["accounts"]) == 8 * 100_000 - 1000
    with rig.connect() as conn:
        assert {row[0] for row in conn.execute("SELECT agent_id FROM accounts")} == set(rig.settings.agent_ids)
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM positions WHERE signed_qty != 0").fetchone()[0] == 2
        assert conn.execute("SELECT COUNT(*) FROM market_settlements").fetchone()[0] == 0
        order = conn.execute("SELECT * FROM orders WHERE ticker = 'MKT-02'").fetchone()
        assert order["status"] == "canceled" and order["cancel_reason"] == "run_shutdown"
        for state in conn.execute("SELECT * FROM agent_state"):
            assert state["private_note"] == f"note-{state['agent_id']}"
            assert state["scheduled_wake_at"] >= state["activity_day_started_at"] + 28800
            # Every agent read its account once; the three that traded charged
            # their order to that market's budget, not to the shared pool.
            assert state["shared_tool_calls"] == 1
        traded = dict(conn.execute(
            "SELECT agent_id, ticker FROM agent_market_calls WHERE order_tool_calls = 1"
        ))
        assert traded == {"agent-01": "MKT-01", "agent-02": "MKT-01", "agent-03": "MKT-02"}
        assert conn.execute("SELECT COUNT(DISTINCT agent_id) FROM transcript_entries WHERE entry_type='model_response'").fetchone()[0] == 8
        assert "private-test-key" not in conn.execute("SELECT config_json FROM run_settings").fetchone()[0]
    logs = (rig.path / "run.log").read_text()
    assert "fill:" in logs and "final account" in logs and "unresolved" in logs
    assert "private-test-key" not in logs
    with pytest.raises(RuntimeError, match="not been initialized"):
        database.db()


def test_duration_cancels_all_eight_inflight_requests(rig):
    started, canceled = set(), set()

    async def infer(**request):
        agent = json.loads(request["messages"][-1]["content"])["account"]["agent_id"]
        started.add(agent)
        try:
            await asyncio.Future()
        finally:
            canceled.add(agent)

    summary = asyncio.run(runner.run(rig.settings, infer=infer, tokenizer=ScriptedTokenizer(), run_id="test"))
    assert started == canceled == set(rig.settings.agent_ids)
    assert summary["trade_count"] == 0 and summary["stop_reason"] == "duration_elapsed"
    assert "zero trades" in (rig.path / "run.log").read_text()


def test_permanent_inference_failure_stops_every_other_loop_and_polling(rig):
    started, canceled = set(), set()

    async def infer(**request):
        agent = json.loads(request["messages"][-1]["content"])["account"]["agent_id"]
        started.add(agent)
        try:
            if agent == "agent-01":
                await until(lambda: len(started) == 8)
                raise PermanentInferenceError("HTTP 401")
            await asyncio.Future()
        finally:
            canceled.add(agent)

    with pytest.raises(PermanentInferenceError, match="401"):
        asyncio.run(runner.run(rig.settings, infer=infer, tokenizer=ScriptedTokenizer(), run_id="test"))
    assert started == canceled == set(rig.settings.agent_ids)
    assert "task_failed" in (rig.path / "run.log").read_text()
    assert "PermanentInferenceError: HTTP 401" in (rig.path / "run.log").read_text()


def test_all_settled_ends_sleeping_loops_early_and_never_refills(rig, monkeypatch):
    class SettlingFeed(runner.MarketFeed):
        async def run(self):
            conn = rig.connect()
            await until(lambda: conn.execute("SELECT COUNT(*) FROM agent_state WHERE private_note='sleeping'").fetchone()[0] == 8)
            for stored in conn.execute("SELECT * FROM markets").fetchall():
                row = dict(stored)
                row.pop("cohort_index")
                self.exchange.apply_market_observation({**row, "status": "finalized", "payout_cents": 100})
            conn.close()

    monkeypatch.setattr(runner, "MarketFeed", SettlingFeed)

    async def infer(**request):
        return response(("hold_until_next_day", {"memory_note": "sleeping"}))

    summary = asyncio.run(runner.run(replace(rig.settings, run_duration_seconds=30), infer=infer,
                                    tokenizer=ScriptedTokenizer(), run_id="test"))
    assert summary["stop_reason"] == "all_markets_settled"
    assert summary["settled_market_count"] == 10 and summary["unsettled_markets"] == []
    assert len(rig.discoveries) == 1


def test_insufficient_cohort_makes_no_inference_request(rig):
    rig.candidates.pop()

    async def infer(**request):
        pytest.fail("selection must finish before inference starts")

    with pytest.raises(ValueError, match="10"):
        asyncio.run(runner.run(rig.settings, infer=infer, tokenizer=ScriptedTokenizer(), run_id="test"))
    with rig.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM transcript_entries").fetchone()[0] == 0


def test_oversized_mandatory_context_fails_preflight_before_inference(rig):
    class TooLarge:
        def apply_chat_template(self, *args, **kwargs):
            return [0] * 12001

    async def infer(**request):
        pytest.fail("mandatory context must fit before inference starts")

    with pytest.raises(ContextLimitError):
        asyncio.run(runner.run(rig.settings, infer=infer, tokenizer=TooLarge(), run_id="test"))


def test_cancellation_honors_concurrency_bound_and_preserves_started_states(rig):
    started, canceled = set(), set()

    async def infer(**request):
        agent = json.loads(request["messages"][-1]["content"])["account"]["agent_id"]
        started.add(agent)
        try:
            await asyncio.Future()
        finally:
            canceled.add(agent)

    async def go():
        task = asyncio.create_task(runner.run(replace(rig.settings, max_concurrent_inference=2),
                                   infer=infer, tokenizer=ScriptedTokenizer(), run_id="test"))
        await until(lambda: len(started) == 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(started) == 2 and started == canceled

    asyncio.run(go())
    assert "canceled" in (rig.path / "run.log").read_text()
    with rig.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM agent_state WHERE activity_day_started_at IS NOT NULL").fetchone()[0] == 8


def test_existing_run_is_refused_without_modifying_artifacts(rig):
    rig.path.mkdir(parents=True)
    marker = rig.path / "keep.txt"
    marker.write_text("preserve")

    with pytest.raises(FileExistsError, match="overwrite or resume"):
        asyncio.run(runner.run(rig.settings, infer=None, tokenizer=ScriptedTokenizer(), run_id="test"))
    assert list(rig.path.iterdir()) == [marker] and marker.read_text() == "preserve"


def test_live_entry_validates_env_before_tokenizer_or_discovery(rig, monkeypatch):
    monkeypatch.setattr(config, "load_credentials", lambda: None)
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setattr(runner, "load_tokenizer", lambda settings: pytest.fail("missing credentials"))
    with pytest.raises(PermanentInferenceError):
        asyncio.run(runner.run_live(rig.settings))
    assert rig.discoveries == [] and not rig.path.exists()


def test_live_entry_binds_env_and_shared_client_without_credentials_in_requests_or_storage(rig, monkeypatch):
    monkeypatch.setattr(config, "load_credentials", lambda: None)
    monkeypatch.setenv("MODEL_BASE_URL", "https://model.example/v1")
    monkeypatch.setenv("MODEL_API_KEY", "private-test-key")
    monkeypatch.setattr(runner, "load_tokenizer", lambda settings: ScriptedTokenizer())
    actual_client = httpx.AsyncClient
    clients, requests, probes = [], [], []

    def handler(request):
        assert request.headers["Authorization"] == "Bearer private-test-key"
        assert "private-test-key" not in request.content.decode()
        if request.url.path.endswith("/models"):
            probes.append(request)
            return httpx.Response(200, json={"data": [{"id": rig.settings.model_name}]})
        # Participants only ever run against an endpoint already proven ready.
        assert probes
        requests.append(request)
        return httpx.Response(200, json={"choices": [response(("hold_until_next_day", {}))]})

    def client_factory(**kwargs):
        assert kwargs["limits"].max_connections == 8
        client = actual_client(transport=httpx.MockTransport(handler), **kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(runner.httpx, "AsyncClient", client_factory)
    summary = asyncio.run(runner.run_live(rig.settings))
    assert len(probes) == 1 and len(requests) == 8
    assert len(clients) == 1 and clients[0].is_closed
    with sqlite3.connect(f"{summary['run_directory']}/run.db") as conn:
        assert "private-test-key" not in "\n".join(conn.iterdump())


@pytest.mark.parametrize("openai_api", ["chat_completions", "responses"])
def test_openai_live_entry_reuses_agent_loops_and_trades_without_a_gpu_endpoint(rig, monkeypatch, openai_api):
    settings = replace(
        rig.settings, model_provider="openai", model_name="gpt-4.1-2025-04-14",
        model_revision=None, model_base_url_env="OPENAI_BASE_URL",
        model_api_key_env="OPENAI_API_KEY", top_p=None,
        openai_api=openai_api,
    )
    monkeypatch.setattr(config, "load_credentials", lambda: None)
    monkeypatch.setenv("OPENAI_API_KEY", "private-openai-test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("MODEL_BASE_URL", "https://unused-gpu.example/v1")
    monkeypatch.setattr(runner, "load_tokenizer", lambda settings: ScriptedTokenizer())
    monkeypatch.setattr(runner, "wait_for_endpoint", lambda *a, **kw: pytest.fail("GPU readiness probe"))
    actual_client = httpx.AsyncClient
    calls = Counter()

    def handler(request):
        path = "responses" if openai_api == "responses" else "chat/completions"
        assert request.method == "POST" and str(request.url) == f"https://api.openai.com/v1/{path}"
        assert request.headers["Authorization"] == "Bearer private-openai-test-key"
        body = json.loads(request.content)
        assert body["model"] == settings.model_name and body["parallel_tool_calls"] is False
        output_key = "max_output_tokens" if openai_api == "responses" else "max_completion_tokens"
        assert body[output_key] == settings.max_output_tokens
        assert {"top_k", "seed", "chat_template_kwargs", "max_tokens", "top_p"}.isdisjoint(body)
        history = body["input"] if openai_api == "responses" else body["messages"]
        brief = json.loads(history[-1]["content"])
        agent = brief["account"]["agent_id"]
        calls[agent] += 1
        if calls[agent] == 1:
            choice = response(("get_orderbook", {"ticker": "MKT-01"}))
        elif calls[agent] == 2 and agent in {"agent-01", "agent-02"}:
            if openai_api == "responses":
                assert history[1]["encrypted_content"] == f"opaque-{agent}"
                assert history[2]["type"] == "function_call"
                assert history[3]["type"] == "function_call_output"
                assert history[2]["call_id"] == history[3]["call_id"]
            else:
                history_call = history[1]["tool_calls"][0]
                assert isinstance(history_call["function"]["arguments"], str)
            choice = response(("submit_order", {
                "ticker": "MKT-01", "action": "sell" if agent == "agent-01" else "buy",
                "order_type": "limit", "quantity": 3, "price_cents": 40,
            }))
        else:
            choice = response(("hold_until_next_day", {}))
        if openai_api == "responses":
            return httpx.Response(200, json={
                "status": "completed", "model": settings.model_name,
                "usage": {"input_tokens": 100, "output_tokens": 20,
                          "output_tokens_details": {"reasoning_tokens": 5}},
                "output": [{"type": "reasoning", "id": f"rs_{agent}_{calls[agent]}",
                            "summary": [], "encrypted_content": f"opaque-{agent}"}] + [
                    {"type": "function_call", "call_id": call["id"],
                     "name": call["function"]["name"], "arguments": call["function"]["arguments"]}
                    for call in choice["message"]["tool_calls"]
                ],
            })
        return httpx.Response(200, json={
            "choices": [choice], "model": settings.model_name,
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        })

    monkeypatch.setattr(runner.httpx, "AsyncClient", lambda **kw: actual_client(
        transport=httpx.MockTransport(handler), **kw,
    ))
    summary = asyncio.run(runner.run_live(settings, run_id="test"))
    assert summary["trade_count"] == 1 and len(calls) == settings.participant_count
    with rig.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] == settings.participant_count
        assert conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 2
        saved = json.loads(conn.execute("SELECT config_json FROM run_settings").fetchone()[0])
        assert saved["model_provider"] == "openai" and saved["model_revision"] is None
        entry = json.loads(conn.execute(
            "SELECT content_json FROM transcript_entries WHERE entry_type='model_response' LIMIT 1"
        ).fetchone()[0])
        assert entry["usage"]["completion_tokens"] == 20 and entry["model"] == settings.model_name
        assert "private-openai-test-key" not in "\n".join(conn.iterdump())


def test_openai_missing_key_fails_before_tokenizer_or_market_discovery(rig, monkeypatch):
    settings = replace(rig.settings, model_provider="openai", model_revision=None,
                       model_api_key_env="OPENAI_API_KEY", model_base_url_env="OPENAI_BASE_URL")
    monkeypatch.setattr(config, "load_credentials", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(runner, "load_tokenizer", lambda settings: pytest.fail("missing credentials"))
    with pytest.raises(PermanentInferenceError, match="API key"):
        asyncio.run(runner.run_live(settings))
    assert rig.discoveries == [] and not rig.path.exists()


def test_polling_failure_cancels_participants_and_is_not_silenced(rig, monkeypatch):
    class FailedFeed(runner.MarketFeed):
        async def run(self):
            await asyncio.sleep(0)
            raise RuntimeError("polling implementation failure")

    monkeypatch.setattr(runner, "MarketFeed", FailedFeed)

    async def infer(**request):
        await asyncio.Future()

    with pytest.raises(RuntimeError, match="polling implementation"):
        asyncio.run(runner.run(rig.settings, infer=infer, tokenizer=ScriptedTokenizer(), run_id="test"))
    assert "task_failed" in (rig.path / "run.log").read_text()


def test_cli_selects_fresh_config_or_resume_without_legacy_flags(capsys):
    with pytest.raises(SystemExit) as error:
        runner.main(["--help"])
    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config" in help_text and "--resume-from" in help_text
    assert all(flag not in help_text for flag in ("--rounds", "--dry-run"))
    with pytest.raises(SystemExit) as error:
        runner.main(["--config", "configs/openai.yaml", "--resume-from", "parent"])
    assert error.value.code == 2


def test_resume_and_resume_again_carry_balances_memories_and_settle_once(rig, monkeypatch):
    calls = Counter()

    async def parent_infer(**request):
        agent = json.loads(request["messages"][-1]["content"])["account"]["agent_id"]
        calls[agent] += 1
        if calls[agent] == 1 and agent in {"agent-01", "agent-02"}:
            return response(("submit_order", {"ticker": "MKT-01", "action": "sell" if agent == "agent-01" else "buy",
                                              "order_type": "limit", "price_cents": 40, "quantity": 10}))
        return response(("hold_until_next_day", {"memory_note": f"private-{agent}"}))

    parent = asyncio.run(runner.run(rig.settings, infer=parent_infer, tokenizer=ScriptedTokenizer(), run_id="test"))
    assert parent["trade_count"] == 1
    original = (rig.path / "run.db").read_bytes()
    polls = []

    def refresh(tickers, **kwargs):
        polls.append(tickers)
        payloads = []
        for index, row in enumerate(rig.candidates):
            if index == 9:  # Missing but previously active must be stale immediately.
                continue
            if index == 0:
                payloads.append({**row, "status": "finalized", "settlement_value_dollars": "1.0000"})
            elif index == 1:
                # Selection filters no longer apply: keep even with zero volume
                # and a price/horizon outside initial selection bounds.
                payloads.append({**row, "volume_24h_fp": "0", "last_price_dollars": "0.99",
                                 "expected_expiration_time": None})
            else:
                payloads.append({**row, "status": "closed"})
        return payloads

    monkeypatch.setattr(stream.kalshi_client, "get_markets_by_tickers", refresh)
    current_day = [20]

    class NextDayParticipant(Participant):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, now=lambda: int(datetime.fromisoformat(
                f"2026-09-{current_day[0]}T08:00:00-04:00").timestamp()))

        async def _finish_cooldown(self):
            if not self._defer_inference:
                self._deadline = self._monotonic()
            await super()._finish_cooldown()

    monkeypatch.setattr(runner, "Participant", NextDayParticipant)
    seen = Counter()

    async def child_infer(**request):
        brief = json.loads(request["messages"][-1]["content"])
        agent = brief["account"]["agent_id"]
        seen[agent] += 1
        assert brief["private_note"] == f"private-{agent}"
        assert [m["ticker"] for m in brief["markets"]] == ["MKT-02"]
        assert len(brief["unavailable_markets"]) == 9
        assert any(message["role"] == "tool" for message in request["messages"])
        if seen[agent] == 1 and agent in {"agent-01", "agent-02"}:
            return response(("submit_order", {"ticker": "MKT-02", "action": "buy" if agent == "agent-01" else "sell",
                                              "order_type": "limit", "price_cents": 30, "quantity": 4}))
        return response(("hold_until_next_day", {"memory_note": f"private-{agent}"}))

    child = asyncio.run(runner.run(rig.settings, infer=child_infer, tokenizer=ScriptedTokenizer(),
                                   run_id="child", resume_from="test"))
    assert set(seen) == set(rig.settings.agent_ids)
    assert child["parent_run_id"] == child["root_run_id"] == "test"
    assert child["trade_count"] == 2 and child["new_trade_count"] == 1
    assert child["settled_market_count"] == 1
    assert (rig.path / "run.db").read_bytes() == original
    current_day[0] = 21

    async def hold(**request):
        return response(("hold_until_next_day", {}))

    grandchild = asyncio.run(runner.run(rig.settings, infer=hold, tokenizer=ScriptedTokenizer(),
                                        run_id="grandchild", resume_from="child"))
    assert grandchild["parent_run_id"] == "child" and grandchild["root_run_id"] == "test"
    assert grandchild["new_trade_count"] == 0 and grandchild["trade_count"] == 2
    assert grandchild["accounts"] == child["accounts"]
    assert grandchild["settled_market_count"] == 1
    assert len(rig.discoveries) == 1
    assert polls[0] == [row["ticker"] for row in rig.candidates]
    assert "MKT-01" not in polls[1]  # Already settled, no repeat payout.
    with sqlite3.connect(config.RUNS_DIR / "grandchild" / "run.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_segments").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 10
        for agent in rig.settings.agent_ids:
            seqs = [r[0] for r in conn.execute("SELECT sequence_number FROM transcript_entries WHERE agent_id=? ORDER BY sequence_number", (agent,))]
            assert seqs == list(range(1, len(seqs) + 1))


@pytest.mark.parametrize("status", ["closed", "finalized"])
def test_resume_without_open_markets_does_not_infer(rig, monkeypatch, status):
    async def hold(**request):
        return response(("hold_until_next_day", {}))

    asyncio.run(runner.run(rig.settings, infer=hold, tokenizer=ScriptedTokenizer(), run_id="test"))
    monkeypatch.setattr(stream.kalshi_client, "get_markets_by_tickers", lambda *args, **kwargs: [
        {**row, "status": status, "settlement_value_dollars": "0.0000"} for row in rig.candidates
    ])

    async def unexpected(**request):
        pytest.fail("no model request without tradable markets")

    result = asyncio.run(runner.run(rig.settings, infer=unexpected, tokenizer=ScriptedTokenizer(),
                                    run_id="child", resume_from="test"))
    assert result["stop_reason"] == ("all_markets_settled" if status == "finalized" else "no_tradable_markets")
    assert len(rig.discoveries) == 1


class ProbeClient:
    """Scripted /models responses; each entry is a status code or an exception.

    The final entry repeats, so a never-ready endpoint is one short script.
    """

    def __init__(self, *script):
        self.script = list(script)
        self.requests = []

    async def get(self, url, *, headers, timeout):
        self.requests.append((url, headers, timeout))
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, BaseException):
            raise step
        status, payload = step if isinstance(step, tuple) else (step, None)
        if payload is None:
            payload = {"data": [{"id": "Qwen/Qwen3.5-9B"}]}
        return httpx.Response(status, json=payload)


def probe(client, **changes):
    arguments = dict(
        base_url="https://host.test/v1", api_key="token", model="Qwen/Qwen3.5-9B",
        budget=1.0, probe_timeout=0.1, retry_after=0.0,
    )
    arguments.update(changes)
    return asyncio.run(runner.wait_for_endpoint(client, **arguments))


def test_endpoint_probe_retries_cold_start_then_reports_readiness(caplog):
    client = ProbeClient(
        httpx.ConnectTimeout("still booting"), 503, httpx.ConnectError("no route"), 200,
    )
    with caplog.at_level("INFO", logger="prediction_markets.runner"):
        probe(client)
    assert len(client.requests) == 4
    url, headers, timeout = client.requests[0]
    assert url == "https://host.test/v1/models"
    assert headers == {"Authorization": "Bearer token"}
    assert timeout == 0.1
    assert "endpoint ready" in caplog.text
    # A booting container is reported while waiting, never silently retried.
    assert "ConnectTimeout" in caplog.text and "HTTP 503" in caplog.text


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_credentials_stop_the_run_without_burning_the_budget(status):
    client = ProbeClient(status)
    with pytest.raises(PermanentInferenceError, match="rejected the API key"):
        probe(client, budget=600.0)
    assert len(client.requests) == 1


def test_serving_a_different_model_is_a_configuration_failure():
    client = ProbeClient((200, {"data": [{"id": "some/other-model"}]}))
    with pytest.raises(PermanentInferenceError, match="pinned model Qwen/Qwen3.5-9B"):
        probe(client, budget=600.0)
    assert len(client.requests) == 1


def test_malformed_model_list_is_a_configuration_failure():
    with pytest.raises(PermanentInferenceError, match="malformed model list"):
        probe(ProbeClient((200, {"models": []})), budget=600.0)


def test_exhausted_budget_names_the_last_symptom_and_never_starts_agents():
    client = ProbeClient(502)
    with pytest.raises(PermanentInferenceError) as error:
        probe(client, budget=0.05, retry_after=0.01)
    assert "not ready within 0s (HTTP 502)" in str(error.value)
    assert "check the serving logs" in str(error.value)
    assert len(client.requests) > 1
