"""Responses compatibility, reasoning replay and order safety; entirely offline."""

import asyncio
import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from prediction_markets.integrations import model_client as model_client
from prediction_markets.integrations import openai_responses as openai_responses
from prediction_markets.integrations import tokenization as tokenization
from prediction_markets.agents import TransientInferenceError, parse_response


def payload(*, name="get_account", arguments="{}", status="completed"):
    return {
        "id": "resp_1", "model": "gpt-5.6-luna", "status": status,
        "output": [
            {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque-state"},
            {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed",
             "phase": "commentary", "content": [{"type": "output_text", "text": "Inspect my account.", "annotations": []}]},
            {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": name,
             "arguments": arguments, "status": "completed"},
        ],
        "usage": {"input_tokens": 120, "output_tokens": 80, "total_tokens": 200,
                  "output_tokens_details": {"reasoning_tokens": 60}},
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
    }


def settings_for(rig):
    return replace(rig.settings, model_provider="openai", model_name="gpt-5.6-luna",
                   model_revision=None, openai_api="responses", openai_reasoning_effort="medium",
                   temperature=None, top_p=None)


def test_responses_round_trip_preserves_reasoning_phase_and_tool_results(participant_rig):
    rig = participant_rig
    settings = settings_for(rig)
    participant, _ = rig.make(settings=settings, model=settings.model_name)
    sent = []

    def handler(request):
        assert str(request.url) == "https://model.example/v1/responses"
        body = json.loads(request.content)
        sent.append(body)
        assert body["reasoning"] == {"effort": "medium"}
        assert body["store"] is False and body["parallel_tool_calls"] is False
        assert body["include"] == ["reasoning.encrypted_content"]
        assert body["max_output_tokens"] == settings.max_output_tokens
        assert {"messages", "n", "max_completion_tokens", "max_tokens", "temperature", "top_p",
                "seed", "top_k", "chat_template_kwargs", "previous_response_id"}.isdisjoint(body)
        assert all(tool["strict"] is False for tool in body["tools"])
        order = next(tool for tool in body["tools"] if tool["name"] == "submit_order")
        assert "price_cents" not in order["parameters"]["required"]
        if len(sent) == 2:
            assert body["input"][1:4] == payload()["output"]
            result = body["input"][4]
            assert result["type"] == "function_call_output" and result["call_id"] == "call_1"
            assert json.loads(result["output"])["tool_call_executed"] is True
            assert body["input"][-1]["role"] == "user"
            # The normalized assistant text/calls were not emitted twice.
            assert sum(item.get("type") == "function_call" for item in body["input"]) == 1
        return httpx.Response(200, json=payload())

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            async def infer():
                return await model_client.infer(
                    messages=participant.build_messages(), tools=participant.tools.schemas,
                    model=settings.model_name, max_tokens=settings.max_output_tokens,
                    enable_thinking=False, sampling=settings.sampling("agent-01"), client=client,
                    base_url="https://model.example/v1", api_key="private-test-key", timeout_seconds=10,
                    provider="openai", openai_api="responses", reasoning_effort="medium",
                )
            first = await infer()
            assert first["usage"]["completion_tokens"] == 80
            await participant._handle_response(first)
            assert participant.shared_tool_calls == 1
            assert participant._groups[0][0]["_responses"]["reasoning_tokens"] == 60
            await infer()

    asyncio.run(go())
    other, _ = rig.make(agent_id="agent-02", settings=settings)
    assert "opaque-state" not in json.dumps(other.build_messages())
    saved = next(row["content"] for row in rig.entries() if row["entry_type"] == "model_response")
    assert saved["responses_output"] == payload()["output"]
    assert "private-test-key" not in json.dumps(rig.entries())


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress", "cancelled"])
def test_incomplete_response_never_places_complete_looking_order(participant_rig, status):
    rig = participant_rig
    participant, _ = rig.make(settings=settings_for(rig))
    result = openai_responses.completion(payload(
        name="submit_order", status=status,
        arguments=json.dumps({"ticker": "MKT-01", "action": "buy", "order_type": "limit",
                              "price_cents": 50, "quantity": 1}),
    ))
    asyncio.run(participant._handle_response(result))
    assert participant.next_action_at == rig.clock.now + rig.settings.inference_retry_seconds
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    assert participant.order_tool_calls == {}
    assert "opaque-state" not in json.dumps(participant.build_messages())


def test_invalid_arguments_are_quoted_without_opaque_reasoning(participant_rig):
    rig = participant_rig
    participant, _ = rig.make(settings=settings_for(rig))
    asyncio.run(participant._handle_response(openai_responses.completion(payload(arguments="broken"))))
    history = participant.build_messages()
    assert "previous_response_and_results" in json.dumps(history)
    assert "opaque-state" not in json.dumps(history)
    assert participant.shared_tool_calls == 0


@pytest.mark.parametrize("bad", [None, [], {}, {"status": "completed", "output": None},
                                    {"status": "completed", "output": [None]},
                                    {"status": "completed", "output": [{"type": "web_search_call"}]}])
def test_malformed_or_unexpected_output_is_not_guessed(bad):
    with pytest.raises(TransientInferenceError, match="malformed Responses"):
        openai_responses.completion(bad)


def test_responses_text_refusals_and_legacy_history():
    result = openai_responses.completion({"status": "completed", "output": [{
        "type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "Cannot comply."}],
    }]})
    assert parse_response(result)["content"] == "Cannot comply."
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "function": {
            "name": "wait", "arguments": {"seconds": 1},
        }}]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
    ]
    original = copy.deepcopy(messages)
    body = openai_responses.request_body(messages=messages, tools=[], model="gpt-5.6-luna",
                                         max_tokens=4096, sampling={}, reasoning_effort=None)
    assert body["input"][1]["arguments"] == '{"seconds": 1}'
    assert body["input"][2]["type"] == "function_call_output"
    assert "reasoning" not in body and "tools" not in body
    assert messages == original


def test_token_counter_uses_reasoning_tokens_not_ciphertext(monkeypatch):
    import tiktoken
    monkeypatch.setattr(tiktoken, "encoding_for_model", lambda model: SimpleNamespace(
        encode=lambda text, **kw: list(text.encode()),
    ))
    counter = tokenization.OpenAITokenizer("gpt-5.6-luna")
    messages = [{"role": "assistant", "content": "hi", "_responses": {
        "reasoning_tokens": 60, "output": payload()["output"],
    }}]
    count = len(counter.apply_chat_template(messages, tools=[]))
    messages[0]["_responses"]["output"][0]["encrypted_content"] = "opaque" * 10000
    original = copy.deepcopy(messages)
    assert len(counter.apply_chat_template(messages, tools=[])) == count
    assert messages == original
    messages[0]["_responses"]["reasoning_tokens"] = 600
    assert len(counter.apply_chat_template(messages, tools=[])) > count
