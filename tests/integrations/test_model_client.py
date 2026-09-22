"""HTTP boundary tests use an in-memory transport, never a hosted model."""

import asyncio
import copy
import json
from dataclasses import replace

import httpx
import pytest

from prediction_markets import config
from prediction_markets.integrations import model_client as model_client
from prediction_markets.runtime import runner as runner
from prediction_markets.agents import PermanentInferenceError, TransientInferenceError, parse_response


def request_kwargs():
    return dict(
        messages=[{"role": "user", "content": "brief"}], tools=[],
        model="Qwen/Qwen3.5-9B", max_tokens=1024, enable_thinking=False,
        sampling={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 11},
        base_url="https://model.example/v1/", api_key="private-test-key", timeout_seconds=120,
    )


def send(handler, **changes):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await model_client.infer(client=client, **{**request_kwargs(), **changes})
    return asyncio.run(go())


def test_request_body_auth_timeout_and_history_encoding():
    messages = [
        {"role": "user", "content": "brief"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function", "function": {"name": "wait", "arguments": {"seconds": 7}},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": '{"ok":true}'},
    ]
    original = copy.deepcopy(messages)
    choice = {"message": {"role": "assistant", "content": "abstain"}, "finish_reason": "stop"}

    def handler(request):
        assert str(request.url) == "https://model.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer private-test-key"
        assert all(seconds == 120 for seconds in request.extensions["timeout"].values())
        body = json.loads(request.content)
        assert body["model"] == "Qwen/Qwen3.5-9B"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["max_tokens"] == 1024 and body["stream"] is False
        assert body["tool_choice"] == "auto" and body["n"] == 1
        assert json.loads(body["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {"seconds": 7}
        return httpx.Response(200, json={"choices": [choice], "usage": {"prompt_tokens": 123}})

    result = send(handler, messages=messages)
    assert parse_response(result)["content"] == "abstain"
    assert result["usage"]["prompt_tokens"] == 123
    assert messages == original


def test_openai_preserves_tool_protocol_without_vllm_options():
    messages = [
        {"role": "system", "content": "Use one tool per response."},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-news", "type": "function",
            "function": {"name": "search_news", "arguments": {"query": "latest news"}},
        }]},
        {"role": "tool", "tool_call_id": "call-news", "content": '{"ok":true}'},
        {"role": "user", "content": "next brief"},
    ]
    original = copy.deepcopy(messages)
    tools = [{"type": "function", "function": {
        "name": "wait", "parameters": {"type": "object", "properties": {}},
    }}]

    def handler(request):
        assert str(request.url) == "https://api.openai.com/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer private-test-key"
        body = json.loads(request.content)
        assert body["model"] == "gpt-4.1-2025-04-14"
        assert body["max_completion_tokens"] == 1024
        assert body["parallel_tool_calls"] is False and body["store"] is False
        assert body["tools"] == tools and body["tool_choice"] == "auto"
        assert body["temperature"] == 0.7 and body["top_p"] == 0.8
        assert {"top_k", "seed", "chat_template_kwargs", "max_tokens", "reasoning_effort"}.isdisjoint(body)
        assert json.loads(body["messages"][1]["tool_calls"][0]["function"]["arguments"]) == {"query": "latest news"}
        return httpx.Response(200, json={
            "model": body["model"], "system_fingerprint": "fp_test",
            "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            "choices": [{"finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-wait", "type": "function",
                    "function": {"name": "wait", "arguments": "{}"},
                }],
            }}],
        })

    result = send(handler, provider="openai", model="gpt-4.1-2025-04-14",
                  base_url="https://api.openai.com/v1", messages=messages, tools=tools)
    assert parse_response(result)["tool_calls"][0]["function"]["name"] == "wait"
    assert result["model"] == "gpt-4.1-2025-04-14" and result["system_fingerprint"] == "fp_test"
    assert result["usage"]["completion_tokens"] == 20
    assert messages == original


def test_openai_reasoning_and_omitted_sampling_are_explicit():
    def handler(request):
        body = json.loads(request.content)
        assert body["reasoning_effort"] == "low"
        assert body["max_completion_tokens"] == 4096
        assert {"temperature", "top_p", "seed", "top_k", "tools", "tool_choice", "parallel_tool_calls"}.isdisjoint(body)
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"role": "assistant", "content": "hold"},
        }]})
    send(handler, provider="openai", reasoning_effort="low", max_tokens=4096,
         sampling={"temperature": None, "top_p": None, "top_k": 20, "seed": 10})


def test_luna_sends_explicit_none_with_real_function_tools(participant_rig):
    settings = replace(config.load(config.ROOT / "configs" / "openai.yaml"),
                       model_name="gpt-5.6-luna", openai_reasoning_effort="none")
    participant, _ = participant_rig.make()

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "gpt-5.6-luna"
        assert body["reasoning_effort"] == "none"
        assert body["tools"] == participant.tools.schemas
        assert body["parallel_tool_calls"] is False
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-wait", "type": "function",
                    "function": {"name": "wait", "arguments": '{"seconds":1}'},
                }],
            },
        }]})

    result = send(handler, provider=settings.model_provider, model=settings.model_name,
                  reasoning_effort=settings.openai_reasoning_effort,
                  sampling=settings.sampling("agent-01"), tools=participant.tools.schemas)
    assert parse_response(result)["tool_calls"][0]["function"]["name"] == "wait"


@pytest.mark.parametrize("effort", [None, "low", "medium"])
def test_luna_reasoning_rejection_names_the_config_fix_without_raw_body(effort):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(400, json={"error": {
            "type": "invalid_request_error", "code": None, "param": "reasoning_effort",
            "message": "Function tools with reasoning_effort are not supported; private-test-key",
        }})

    with pytest.raises(PermanentInferenceError) as error:
        send(handler, provider="openai", model="gpt-5.6-luna", reasoning_effort=effort,
             tools=[{"type": "function", "function": {"name": "wait"}}])
    detail = str(error.value)
    assert "HTTP 400" in detail and "param=reasoning_effort" in detail
    assert "openai_reasoning_effort: none (not null)" in detail
    assert "Responses API" in detail
    assert "private-test-key" not in detail and "Function tools with" not in detail
    assert len(requests) == 1


@pytest.mark.parametrize("payload", [None, [], {}, {"error": []}, {"error": {
    "type": "private-test-key", "code": ["private-test-key"],
    "param": {"private-test-key": True}, "message": "secret body",
}}, {"error": {"type": "private-test-key", "code": "private-test-key", "param": "private-test-key"}}])
def test_error_diagnostics_ignore_unknown_or_malformed_fields(payload):
    with pytest.raises(PermanentInferenceError) as error:
        send(lambda request: httpx.Response(400, json=payload))
    assert "HTTP 400" in str(error.value)
    assert "secret" not in str(error.value) and "private-test-key" not in str(error.value)


def test_parameter_rejection_exposes_known_identifiers_only():
    with pytest.raises(PermanentInferenceError) as error:
        send(lambda request: httpx.Response(400, json={"error": {
            "type": "invalid_request_error", "code": "unsupported_value",
            "param": "temperature", "message": "private-test-key",
        }}), provider="openai", model="gpt-4.1")
    assert "code=unsupported_value; param=temperature" in str(error.value)
    assert "Luna" not in str(error.value) and "private-test-key" not in str(error.value)


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504])
def test_retryable_status_is_reported_once_without_body_or_key(code):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(code, text="secret error body")

    with pytest.raises(TransientInferenceError) as error:
        send(handler)
    assert str(code) in str(error.value)
    assert "secret" not in str(error.value) and "private-test-key" not in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize("code", [301, 307, 400, 401, 403, 404, 422])
def test_permanent_status_including_redirect_never_retries_or_forwards_key(code):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(code, headers={"Location": "https://other.example/v1"}, text="private-test-key")

    with pytest.raises(PermanentInferenceError) as error:
        send(handler)
    assert str(code) in str(error.value) and "private-test-key" not in str(error.value)
    assert len(requests) == 1


@pytest.mark.parametrize("payload", [None, [], {}, {"choices": []}, {"choices": [1]}, {"choices": [{}, {}]}])
def test_malformed_http_envelope_is_not_guessed(payload):
    with pytest.raises(TransientInferenceError, match="malformed"):
        send(lambda request: httpx.Response(200, json=payload))


def test_bad_json_is_transient_but_truncated_choice_reaches_agent_validation():
    with pytest.raises(TransientInferenceError, match="malformed"):
        send(lambda request: httpx.Response(200, text="broken JSON"))
    result = send(lambda request: httpx.Response(200, json={"choices": [{
        "message": {"role": "assistant", "content": None}, "finish_reason": "length",
    }]}))
    with pytest.raises(ValueError, match="truncated"):
        parse_response(result)


@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError, httpx.RemoteProtocolError])
def test_transport_failures_do_not_leak_exception_details(error_type):
    def handler(request):
        raise error_type("private-test-key", request=request)

    with pytest.raises(TransientInferenceError) as error:
        send(handler)
    assert "private-test-key" not in str(error.value)


def test_cancellation_propagates_to_pending_http_without_retry():
    async def go():
        started, canceled = asyncio.Event(), asyncio.Event()

        async def handler(request):
            started.set()
            try:
                await asyncio.Future()
            finally:
                canceled.set()

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            task = asyncio.create_task(model_client.infer(client=client, **request_kwargs()))
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert canceled.is_set()
    asyncio.run(go())


@pytest.mark.parametrize("url", [
    "", "http://model.example/v1", "https://private-test-key@model.example/v1",
    "https://model.example/v1?key=private-test-key", "https://model.example/v1#key",
    "https://model.example", "https://model.example:bad/v1",
])
def test_invalid_endpoint_is_rejected_without_echoing_credentials(url):
    with pytest.raises(PermanentInferenceError) as error:
        model_client.validate_endpoint(url, "private-test-key")
    assert "private-test-key" not in str(error.value)


def test_tokenizer_download_is_revision_pinned_and_weights_excluded(monkeypatch, tmp_path):
    import huggingface_hub
    from transformers import AutoTokenizer
    from types import SimpleNamespace

    settings = config.load(config.CONFIG_DIR / "mvp.yaml")
    seen = {}

    def download(name, **kwargs):
        seen.update(name=name, **kwargs)
        return str(tmp_path)

    def load(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return SimpleNamespace(chat_template="template")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    runner.load_tokenizer(settings)
    assert seen["revision"] == settings.model_revision
    assert seen["name"] == settings.model_name
    assert "chat_template.jinja" in seen["allow_patterns"]
    assert not any("*" in pattern or pattern.endswith(("bin", "safetensors")) for pattern in seen["allow_patterns"])


@pytest.mark.parametrize("raw", ['{}', '{"seconds":7}', 'broken', '[]', '{"a":1,"a":2}', '{"seconds":1e400}'])
def test_real_cached_qwen_template_handles_valid_and_rejected_history(participant_rig, raw):
    """Optional local artifact check: never downloads anything during pytest."""
    from transformers import AutoTokenizer

    rig = participant_rig
    cache = config.ROOT / "data" / "huggingface"
    artifact = cache / "models--Qwen--Qwen3.5-9B" / "snapshots" / rig.settings.model_revision
    if not (artifact / "chat_template.jinja").exists():
        pytest.skip("pinned tokenizer not cached; run runner.load_tokenizer(settings) once to enable this offline check")
    tokenizer = AutoTokenizer.from_pretrained(str(artifact), local_files_only=True, trust_remote_code=False)
    participant, _ = rig.make(tokenizer=tokenizer)
    response = rig.response(rig.call("wait", raw=raw), rig.call("get_account"))
    try:
        asyncio.run(participant._handle_response(response))
    except asyncio.CancelledError:
        pass
    messages = participant.build_messages()
    tokens = tokenizer.apply_chat_template(messages, tools=participant.tools.schemas, tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False)
    assert len(tokens) <= participant.input_token_limit
    rendered = tokenizer.apply_chat_template(messages, tools=participant.tools.schemas, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    assert rendered.endswith("<think>\n\n</think>\n\n")
    original = next(row["content"] for row in rig.entries() if row["entry_type"] == "model_response")
    assert original["message"]["tool_calls"][0]["function"]["arguments"] == raw
    for message in messages:
        for call in message.get("tool_calls", []):
            assert isinstance(call["function"]["arguments"], dict)

    # Mirror the server's JSON-string -> object conversion and compare exact tokens.
    def handler(request):
        wire = json.loads(request.content)
        for message in wire["messages"]:
            for call in message.get("tool_calls", []):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
        assert tokenizer.apply_chat_template(wire["messages"], tools=wire["tools"], tokenize=True,
                                             add_generation_prompt=True, **wire["chat_template_kwargs"]) == tokens
        return httpx.Response(200, json={"choices": [rig.response(content="wait")]})

    send(handler, messages=messages, tools=participant.tools.schemas)
