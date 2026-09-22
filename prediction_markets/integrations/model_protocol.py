"""Shared completion validation and errors for every model integration."""

import copy
from typing import Any


class TransientInferenceError(Exception):
    """The model transport may be tried again after the ordinary retry wait."""


class PermanentInferenceError(RuntimeError):
    """Configuration/authentication failure: propagate to stop the run."""


class ContextLimitError(PermanentInferenceError):
    """Mandatory input cannot fit; never silently remove market rules."""


def parse_response(response: Any) -> dict:
    """Validate the complete tool envelope before dispatching any of its calls.

    The injected client returns {message: <assistant message>, finish_reason:
    'stop' or 'tool_calls'}. Argument JSON is validated by local tool dispatch.
    A truncated or structurally malformed envelope never executes any order.
    """
    if (
        not isinstance(response, dict)
        or not isinstance(response.get("finish_reason"), str)
        or response["finish_reason"] not in {"stop", "tool_calls"}
    ):
        raise ValueError("missing, unsupported, or truncated finish_reason")
    message = response.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ValueError("response must contain an assistant message")
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("assistant content must be text or null")
    calls = message.get("tool_calls", [])
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be a list")
    ids = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise ValueError("each tool call must have type function")
        call_id, function = call.get("id"), call.get("function")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in ids:
            raise ValueError("tool call IDs must be nonempty and unique within the response")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not isinstance(function.get("arguments"), str):
            raise ValueError("each tool call must contain a name and JSON argument string")
        ids.add(call_id)
    result = {"role": "assistant", "content": content}
    if calls:
        result["tool_calls"] = copy.deepcopy(calls)
    return result


