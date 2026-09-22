"""Shared HTTP transport for vLLM Chat Completions and OpenAI APIs."""

from __future__ import annotations

import copy
import json
from urllib.parse import urlsplit

import httpx

from .model_protocol import PermanentInferenceError, TransientInferenceError
from . import openai_responses


def _permanent_error(response: httpx.Response, *, body: dict, provider: str) -> str:
    """Expose known error identifiers, never arbitrary server text or values."""
    detail = f"model endpoint returned HTTP {response.status_code}"
    try:
        payload = response.json()
    except ValueError:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        allowed = {
            "type": {"invalid_request_error", "authentication_error", "permission_error"},
            "code": {"unsupported_parameter", "unsupported_value", "invalid_value",
                     "invalid_api_key", "model_not_found", "context_length_exceeded"},
            # Also recognize omitted parameters that the endpoint requires.
            "param": set(body) | {"reasoning_effort", "temperature", "top_p"},
        }
        for field, known in allowed.items():
            value = error.get(field)
            if isinstance(value, str) and value in known:
                detail += f"; {field}={value}"
        if (response.status_code == 400 and provider == "openai"
                and "messages" in body
                and body.get("model") == "gpt-5.6-luna" and body.get("tools")
                and body.get("reasoning_effort") != "none"
                and error.get("param") == "reasoning_effort"):
            return detail + (
                "; Luna function tools on Chat Completions require "
                "openai_reasoning_effort: none (not null). "
                "For reasoning with Luna tools, set openai_api: responses (Responses API)."
            )
    return detail + "; check model parameters, endpoint, authentication and serving configuration"


def validate_endpoint(base_url: str, api_key: str) -> str:
    """Require TLS and separate credentials; never echo a malformed URL/key."""
    try:
        url = urlsplit(base_url)
        valid = (
            url.scheme == "https" and url.hostname and url.port in (None, 443)
            and not url.username and not url.password and not url.query and not url.fragment
            and url.path.rstrip("/").endswith("/v1")
        )
    except ValueError:
        valid = False
    if not valid:
        raise PermanentInferenceError("model base URL must be HTTPS ending in /v1, without embedded credentials or query parameters")
    if not isinstance(api_key, str) or not api_key.strip() or any(c.isspace() for c in api_key):
        raise PermanentInferenceError("model API key must be a nonempty bearer token without whitespace")
    return base_url.rstrip("/")


async def infer(
    *, messages: list[dict], tools: list[dict], model: str, max_tokens: int,
    enable_thinking: bool, sampling: dict, client: httpx.AsyncClient,
    base_url: str, api_key: str, timeout_seconds: int,
    provider: str = "vllm", reasoning_effort: str | None = None,
    openai_api: str = "chat_completions",
) -> dict:
    """Send once; participants own timeouts/backoff and all tool execution.

    Input history uses HF's argument objects so its exact chat template can
    count tokens. The wire format requires JSON argument strings. Provider-
    supported sampling options are sent when configured. Never parse
    prose as a tool call, retry a write, or log credentials/raw HTTP error bodies.
    """
    if provider not in {"vllm", "openai"}:
        raise PermanentInferenceError("unsupported model provider")
    if openai_api not in {"chat_completions", "responses"} or (provider != "openai" and openai_api != "chat_completions"):
        raise PermanentInferenceError("unsupported model API for this provider")
    base_url = validate_endpoint(base_url, api_key)
    use_responses = provider == "openai" and openai_api == "responses"
    path = "responses" if use_responses else "chat/completions"
    if use_responses:
        body = openai_responses.request_body(
            messages=messages, tools=tools, model=model, max_tokens=max_tokens,
            sampling=sampling, reasoning_effort=reasoning_effort,
        )
    else:
        wire_messages = copy.deepcopy(messages)
        for message in wire_messages:
            message.pop("_responses", None)
            for call in message.get("tool_calls", []):
                arguments = call["function"]["arguments"]
                if not isinstance(arguments, dict):
                    raise PermanentInferenceError("historical tool arguments must be parsed objects")
                call["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        body = {
            "model": model, "messages": wire_messages,
            "stream": False, "n": 1,
        }
        if tools:
            body.update(tools=tools, tool_choice="auto")
        if provider == "openai":
            # vLLM-only options never reach OpenAI.
            body.update(max_completion_tokens=max_tokens, store=False)
            if tools:
                body["parallel_tool_calls"] = False
            body.update({key: sampling[key] for key in ("temperature", "top_p")
                         if sampling.get(key) is not None})
            if reasoning_effort is not None:
                body["reasoning_effort"] = reasoning_effort
        else:
            body.update(
                tools=tools, tool_choice="auto", max_tokens=max_tokens,
                chat_template_kwargs={"enable_thinking": enable_thinking}, **sampling,
            )
    try:
        response = await client.post(
            f"{base_url}/{path}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=timeout_seconds, follow_redirects=False,
        )
    except httpx.TransportError as exc:
        raise TransientInferenceError(f"model transport failed ({type(exc).__name__})") from None
    if response.status_code in {408, 429} or response.status_code >= 500:
        raise TransientInferenceError(f"model endpoint returned HTTP {response.status_code}")
    if response.status_code != 200:
        raise PermanentInferenceError(_permanent_error(response, body=body, provider=provider))
    try:
        payload = response.json()
        if use_responses:
            return openai_responses.completion(payload)
        choices = payload["choices"]
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError("expected exactly one choice")
    except (ValueError, KeyError, TypeError):
        raise TransientInferenceError("model endpoint returned a malformed completion envelope") from None
    # Participant validates finish_reason/message before executing any tools.
    # Preserve optional usage for transcript inspection, not quota enforcement.
    return {
        **choices[0], "usage": payload.get("usage"),
        "model": payload.get("model"), "system_fingerprint": payload.get("system_fingerprint"),
    }
