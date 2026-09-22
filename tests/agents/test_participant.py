"""Scripted participant loops: elapsed costs, holds, context, and inference boundaries."""

import asyncio
import copy
import json
from dataclasses import replace

import pytest

from conftest import ScriptFinished

from prediction_markets.agents import (
    ContextLimitError, PermanentInferenceError, TransientInferenceError,
    build_prompt, parse_response,
)


def run_script(participant):
    with pytest.raises(ScriptFinished):
        asyncio.run(participant.run())


def tool_results(rig, agent_id="agent-01"):
    return [row["content"] for row in rig.entries(agent_id) if row["entry_type"] == "tool_result"]


def _order(rig, ticker="MKT-01", action="buy"):
    return rig.call("submit_order", {"ticker": ticker, "action": action,
                    "order_type": "limit", "quantity": 1, "price_cents": 40})


@pytest.mark.parametrize("inference_seconds,expected_time,expected_sleep", [(30, 120, [90]), (180, 180, [])])
def test_next_action_waits_for_max_of_inference_and_previous_cost(participant_rig, inference_seconds,
                                                                  expected_time, expected_sleep):
    rig = participant_rig
    initial = rig.clock.now
    async def decide(request):
        # The tool result and fresh account are available BEFORE its cost ends.
        assert rig.clock.now == initial
        assert any(m["role"] == "tool" for m in request["messages"])
        assert len(rig.exchange.get_own_orders("agent-01")["orders"]) == 1
        rig.advance(inference_seconds)
        return rig.response(_order(rig))
    participant, model = rig.make(rig.response(_order(rig)), decide)
    run_script(participant)
    assert rig.clock.now == initial + expected_time
    assert rig.clock.sleeps == expected_sleep
    assert rig.market_calls() == {"MKT-01": 2}
    assert len(model.requests) == 3
    results = [row for row in rig.entries() if row["entry_type"] == "tool_result"]
    assert results[1]["created_at"] == initial + expected_time


def test_every_read_including_own_orders_has_five_minute_cost(participant_rig):
    rig = participant_rig
    calls = [("get_orderbook", {"ticker": "MKT-01"}), ("get_recent_trades", {"ticker": "MKT-01"}),
             ("get_own_orders", {}), ("get_account", {})]
    participant, _ = rig.make(*(rig.response(rig.call(name, args)) for name, args in calls),
                              rig.response(_order(rig)))
    run_script(participant)
    assert rig.clock.sleeps == [300] * 4
    assert rig.state()["shared_tool_calls"] == 4


def test_cancel_and_rejected_orders_still_cost_two_minutes(participant_rig):
    rig = participant_rig
    participant, _ = rig.make(rig.response(_order(rig)),
                              rig.response(rig.call("cancel_order", {"order_id": 1})),
                              rig.response(rig.call("cancel_order", {"order_id": 999})),
                              rig.response(_order(rig, "INVENTED")),
                              rig.response(rig.call("get_account")))
    run_script(participant)
    assert rig.clock.sleeps == [120] * 4
    assert tool_results(rig)[2]["result"]["error"]["code"] == "not_found"
    assert tool_results(rig)[3]["result"]["error"]["code"] == "unknown_market"


def test_tool_latency_and_inference_overlap_the_same_cost(participant_rig):
    rig = participant_rig
    async def news(**kwargs):
        rig.advance(25)
        return {"ok": True, "results": []}
    async def decide(request):
        rig.advance(45)
        return rig.response(_order(rig))
    participant, _ = rig.make(rig.response(rig.call("search_news", {"query": "facts"})), decide)
    participant.tools._async_functions["search_news"] = news
    run_script(participant)
    assert rig.clock.sleeps == [600 - 25 - 45]


def test_only_one_dispatched_tool_per_response_and_no_quote_tool(participant_rig):
    rig = participant_rig
    participant, _ = rig.make(rig.response(rig.call("submit_quote"), rig.call("get_account"), _order(rig)))
    run_script(participant)
    results = tool_results(rig)
    assert [r["result"]["tool_call_executed"] for r in results] == [False, True, False]
    assert results[0]["result"]["error"]["code"] == "unknown_tool"
    assert results[-1]["result"]["error"]["code"] == "one_tool_call_per_response"
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_no_order_or_shared_read_count_quota(participant_rig):
    rig = participant_rig
    participant, _ = rig.make(*(rig.response(_order(rig)) for _ in range(55)),
                              *(rig.response(rig.call("get_account")) for _ in range(201)),
                              settings=rig.tight(read_tool_minutes=1))
    run_script(participant)
    assert rig.market_calls() == {"MKT-01": 55}
    assert rig.state()["shared_tool_calls"] == 201
    assert all(r["result"]["tool_call_executed"] for r in tool_results(rig))


@pytest.mark.parametrize("minutes", [7, 60])
def test_hold_duration_comes_from_config_and_inference_can_overlap(participant_rig, minutes):
    rig = participant_rig
    participant, _ = rig.make(rig.response(rig.call("hold_for_now", {"memory_note": "thesis"})),
                              rig.response(_order(rig)), settings=rig.tight(hold_for_now_minutes=minutes))
    run_script(participant)
    assert rig.clock.sleeps == [minutes * 60]
    assert rig.state()["private_note"] == "thesis"


def test_no_tools_implies_configured_hold(participant_rig):
    rig = participant_rig
    participant, _ = rig.make(rig.response(content="Hold."), rig.response(_order(rig)))
    run_script(participant)
    assert rig.clock.sleeps == [3600]
    assert tool_results(rig)[0]["name"] == "hold_for_now"


def test_invalid_tools_have_retry_delay_without_mutation(participant_rig):
    rig = participant_rig
    participant, _ = rig.make(rig.response(rig.call("hold_for_now", {"seconds": 1, "memory_note": "bad"})),
                              rig.response(rig.call("get_account", raw="broken JSON")))
    run_script(participant)
    assert rig.clock.sleeps == [60, 60]
    assert rig.state()["private_note"] == ""
    assert rig.state()["shared_tool_calls"] == 0


def test_retry_during_hold_never_shortens_the_hold(participant_rig):
    rig = participant_rig
    participant, _ = rig.make(rig.response(rig.call("hold_for_now")),
                              TransientInferenceError("busy"), rig.response(_order(rig)))
    run_script(participant)
    assert rig.clock.sleeps == [3600]


@pytest.mark.parametrize("finish", ["length", "content_filter", None, [], "unknown"])
def test_bad_finish_reason_never_executes_even_a_valid_order(participant_rig, finish):
    rig = participant_rig
    response = rig.response(rig.call("submit_order", {
        "ticker": "MKT-01", "action": "buy", "order_type": "limit", "quantity": 1, "price_cents": 40,
    }))
    response["finish_reason"] = finish
    participant, _ = rig.make(response)
    run_script(participant)
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    assert rig.state()["shared_tool_calls"] == 0
    assert tool_results(rig)[0]["result"]["error"]["code"] == "invalid_model_response"
    assert rig.clock.sleeps == [60]


def test_malformed_envelope_rejects_entire_batch_before_first_order(participant_rig):
    rig = participant_rig
    valid = rig.call("submit_order", {
        "ticker": "MKT-01", "action": "buy", "order_type": "limit", "quantity": 1, "price_cents": 40,
    })
    participant, _ = rig.make(rig.response(valid, {"id": "malformed"}))
    run_script(participant)
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_duplicate_call_id_is_not_dispatched_twice(participant_rig):
    rig = participant_rig
    call = rig.call("get_account")
    with pytest.raises(ValueError, match="unique"):
        parse_response(rig.response(call, copy.deepcopy(call)))


def test_transient_failure_waits_and_retries_without_spending_tool_budget(participant_rig):
    rig = participant_rig
    participant, fake = rig.make(
        TransientInferenceError("busy"), rig.response(rig.call("get_account")),
        sleep=rig.advance_sleep,
    )
    with pytest.raises(rig.finished):
        asyncio.run(participant.run())
    assert len(fake.requests) == 3
    assert rig.clock.sleeps == [60]
    assert rig.state()["shared_tool_calls"] == 1
    assert len([entry for entry in rig.entries() if entry["entry_type"] == "model_response"]) == 1


def test_permanent_inference_failure_propagates_immediately(participant_rig):
    rig = participant_rig
    participant, fake = rig.make(PermanentInferenceError("bad endpoint credential"))
    with pytest.raises(PermanentInferenceError):
        asyncio.run(participant.run())
    assert len(fake.requests) == 1 and rig.clock.sleeps == []
    assert rig.state()["shared_tool_calls"] == 0


def test_inference_timeout_cancels_request_before_retry_sleep(participant_rig):
    rig = participant_rig
    canceled = []

    async def never_respond(request):
        try:
            await asyncio.Future()
        finally:
            canceled.append(True)

    participant, fake = rig.make(never_respond, settings=replace(rig.settings, request_timeout_seconds=1))
    run_script(participant)
    assert canceled == [True]
    assert len(fake.requests) == 2
    assert rig.clock.sleeps == [60]


def test_a_state_changing_tool_is_never_retried_on_failure(participant_rig, monkeypatch):
    rig = participant_rig
    calls = []

    def failed_write(*args, **kwargs):
        calls.append((args, kwargs))
        raise RuntimeError("database write failed")

    monkeypatch.setattr(rig.exchange, "submit_order", failed_write)
    participant, fake = rig.make(rig.response(rig.call("submit_order", {
        "ticker": "MKT-01", "action": "buy", "order_type": "limit", "quantity": 1, "price_cents": 40,
    })))
    with pytest.raises(RuntimeError, match="database write failed"):
        asyncio.run(participant.run())
    assert len(calls) == 1 and len(fake.requests) == 1


class SizedTokenizer:
    """80 for the cached prefix, 30 per stored group, and the brief by length."""

    def apply_chat_template(self, messages, **kwargs):
        assert len(kwargs["tools"]) == 9
        assert kwargs["tokenize"] and kwargs["add_generation_prompt"]
        groups = 30 * sum(message["role"] == "assistant" for message in messages)
        brief = max(
            len(message["content"]) for message in messages if message["role"] == "user"
        ) // 100
        return [0] * (80 + groups + brief)


def _brief_tokens(rig):
    """What the unabridged brief costs on top of the cached fixed prefix."""
    probe, _ = rig.make(
        tokenizer=SizedTokenizer(),
        settings=replace(rig.settings, working_token_budget=10_000),
    )
    return probe._count(probe.build_messages()) - probe.fixed_prefix_tokens


def test_context_trimming_drops_complete_groups_and_preserves_rules_and_note(participant_rig):
    rig = participant_rig
    # Room for the brief and exactly one stored group, never two.
    participant, _ = rig.make(
        tokenizer=SizedTokenizer(),
        settings=replace(rig.settings, working_token_budget=_brief_tokens(rig) + 45),
    )
    participant.private_note = "important private note"
    old = rig.call("get_account")
    recent = rig.call("get_own_orders")
    participant._groups = [
        [rig.response(call)["message"], {"role": "tool", "tool_call_id": call["id"], "content": "{}"}]
        for call in [old, recent]
    ]
    messages = participant.build_messages()
    ids = [message["tool_call_id"] for message in messages if message["role"] == "tool"]
    assert ids == [recent["id"]]
    assert messages[1]["tool_calls"][0]["id"] == recent["id"]
    brief = json.loads(messages[-1]["content"])
    assert len(brief["markets"]) == 10
    assert all(market["rules"].startswith("Full unabridged rules") for market in brief["markets"])
    assert brief["private_note"] == "important private note"


def test_mandatory_context_overflow_fails_before_inference(participant_rig):
    rig = participant_rig
    # The fixed prefix is always granted; overflow now means the brief alone
    # cannot fit the working budget, which no amount of trimming can fix.
    participant, fake = rig.make(
        tokenizer=SizedTokenizer(),
        settings=replace(rig.settings, working_token_budget=_brief_tokens(rig) - 1),
    )
    with pytest.raises(ContextLimitError, match="market rules"):
        asyncio.run(participant.run())
    assert fake.requests == []


def test_input_limit_is_the_measured_cached_prefix_plus_the_working_budget(participant_rig):
    rig = participant_rig
    participant, _ = rig.make(
        tokenizer=SizedTokenizer(),
        settings=replace(rig.settings, working_token_budget=500),
    )
    # The prefix is the system prompt and tool schemas with no brief or history.
    assert participant.fixed_prefix_tokens == 80
    assert participant.input_token_limit == 580

    # A longer prompt does not eat into the working budget; it raises the limit.
    longer, _ = rig.make(
        prompt="x" * 4_000, tokenizer=SizedTokenizer(),
        settings=replace(rig.settings, working_token_budget=500),
    )
    assert longer.input_token_limit - longer.fixed_prefix_tokens == 500


def test_prefix_that_cannot_fit_the_server_context_fails_before_inference(participant_rig):
    rig = participant_rig
    # 19_456 + 1_024 is exactly the server context, so the configuration check
    # passes; the measured 80-token prefix is what tips it over.
    settings = replace(rig.settings, working_token_budget=19_456)
    assert settings.working_token_budget + settings.max_output_tokens == settings.server_context_tokens
    with pytest.raises(ContextLimitError, match="server context"):
        rig.make(tokenizer=SizedTokenizer(), settings=settings)


def test_model_and_prompt_constructor_arguments_are_kept(participant_rig):
    rig = participant_rig
    participant, fake = rig.make(rig.response(), model="custom-model", prompt="custom prompt")
    run_script(participant)
    assert fake.requests[0]["model"] == "custom-model"
    assert fake.requests[0]["messages"][0]["content"] == "custom prompt"


def test_default_prompt_explains_waits_funded_trading_and_separate_time_fields(participant_rig):
    prompt = build_prompt(participant_rig.settings)
    for text in ("hold_until_next_day", "100", "America/New_York", "close_time", "expected_expiration_time", "latest_expiration_time", "or hold", "100-p", "60 real minutes"):
        assert text in prompt
    assert "cash may go negative" not in prompt
    assert "web_search" not in prompt and "get_news" not in prompt


def test_one_participant_cannot_start_two_inference_loops(participant_rig):
    rig = participant_rig
    participant, fake = rig.make(rig.response())
    run_script(participant)
    with pytest.raises(RuntimeError, match="already started"):
        asyncio.run(participant.run())
    assert len(fake.requests) == 2


def test_sampling_is_explicit_and_seeded_per_agent(participant_rig):
    rig = participant_rig
    first, first_model = rig.make(rig.response(rig.call("hold_for_now")), agent_id="agent-01")
    second, second_model = rig.make(rig.response(rig.call("hold_for_now")), agent_id="agent-02")
    run_script(first)
    run_script(second)

    sent = first_model.requests[0]["sampling"]
    assert sent == {
        "temperature": rig.settings.temperature, "top_p": rig.settings.top_p,
        "top_k": rig.settings.top_k, "seed": rig.settings.sampling_seed,
    }
    # One run-level seed, offset per agent, so the eight do not draw alike.
    assert second_model.requests[0]["sampling"]["seed"] == rig.settings.sampling_seed + 1
    assert len({rig.settings.seed_for(a) for a in rig.settings.agent_ids}) == 8


def test_participant_cannot_mutate_the_shared_sampling_settings(participant_rig):
    rig = participant_rig
    participant, fake = rig.make(rig.response(rig.call("hold_for_now")))
    run_script(participant)
    fake.requests[0]["sampling"]["temperature"] = 99
    assert participant.sampling["temperature"] == rig.settings.temperature


@pytest.mark.parametrize(("name", "arguments"), [
    ("get_orderbook", {"ticker": "MKT-01"}),
    ("get_recent_trades", {"ticker": "MKT-01"}),
    ("get_own_orders", {}),
    ("get_account", {}),
])
def test_reads_are_available_before_any_quotes_or_trades(participant_rig, name, arguments):
    rig = participant_rig
    participant, model = rig.make(
        rig.response(rig.call(name, arguments)),
        rig.response(rig.call("hold_for_now")),
    )
    run_script(participant)
    result = tool_results(rig)[0]["result"]
    assert result["ok"] is True and result["tool_call_executed"] is True
    assert rig.state()["shared_tool_calls"] == 1
    assert rig.market_calls() == {}
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    brief = json.loads(model.requests[0]["messages"][-1]["content"])
    assert "opening_quotes_still_required" not in brief


@pytest.mark.parametrize("action", ["buy", "sell"])
def test_first_trade_can_be_a_single_sided_limit_order(participant_rig, action):
    rig = participant_rig
    participant, _ = rig.make(
        rig.response(rig.call("submit_order", {
            "ticker": "MKT-01", "action": action, "order_type": "limit",
            "quantity": 5, "price_cents": 40,
        })),
        rig.response(rig.call("hold_for_now")),
    )
    run_script(participant)
    result = tool_results(rig)[0]["result"]
    assert result["ok"] is True and result["order"]["status"] == "open"
    assert result["trades"] == []
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    assert rig.market_calls() == {"MKT-01": 1}
