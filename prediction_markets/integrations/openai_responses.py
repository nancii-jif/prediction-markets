"""Translate the shared participant protocol to stateless OpenAI Responses."""

from __future__ import annotations

import copy
import json

from .model_protocol import PermanentInferenceError, TransientInferenceError


def request_body(*, messages, tools, model, max_tokens, sampling, reasoning_effort):
    inputs = []
    for message in messages:
        role = message["role"]
        if role == "assistant" and "_responses" in message:
            # Replay the complete ordered output, including encrypted reasoning,
            # phases, item IDs and call IDs; never duplicate normalized calls.
            inputs.extend(copy.deepcopy(message["_responses"]["output"]))
        elif role == "tool":
            inputs.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                           "output": message["content"]})
        else:
            if message.get("content") is not None:
                inputs.append({"role": role, "content": message["content"]})
            for call in message.get("tool_calls", []):
                function = call["function"]
                if not isinstance(function["arguments"], dict):
                    raise PermanentInferenceError("historical tool arguments must be parsed objects")
                inputs.append({
                    "type": "function_call", "call_id": call["id"], "name": function["name"],
                    "arguments": json.dumps(function["arguments"], ensure_ascii=False, allow_nan=False),
                })
    body = {
        "model": model, "input": inputs, "stream": False, "store": False,
        "max_output_tokens": max_tokens, "include": ["reasoning.encrypted_content"],
    }
    if tools:
        body.update(
            # Retain optional arguments and local validation; Responses defaults
            # to strict normalization, unlike the existing Chat tool schemas.
            tools=[{"type": "function", "strict": False, **copy.deepcopy(tool["function"])}
                   for tool in tools],
            tool_choice="auto", parallel_tool_calls=False,
        )
    body.update({key: sampling[key] for key in ("temperature", "top_p")
                 if sampling.get(key) is not None})
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def completion(payload):
    """Normalize only function calls/text; incomplete output never executes."""
    try:
        status = payload["status"]
        output = payload["output"]
        if not isinstance(status, str) or not isinstance(output, list):
            raise ValueError("invalid response envelope")
        calls, text = [], []
        for item in output:
            kind = item["type"]
            if kind == "function_call":
                calls.append({"id": item["call_id"], "type": "function", "function": {
                    "name": item["name"], "arguments": item["arguments"],
                }})
            elif kind == "message":
                if item["role"] != "assistant" or not isinstance(item["content"], list):
                    raise ValueError("invalid assistant message")
                for part in item["content"]:
                    if part["type"] == "output_text":
                        text.append(part["text"])
                    elif part["type"] == "refusal":
                        text.append(part["refusal"])
                    else:
                        raise ValueError("unsupported assistant content")
            elif kind != "reasoning":
                raise ValueError("unexpected hosted tool or output item")
        finish = "tool_calls" if calls else "stop"
        if status != "completed":
            # Participant.parse_response rejects every non-completed envelope,
            # even if it contains a complete-looking order before truncation.
            finish = "length" if status == "incomplete" else "invalid_response"
        usage = copy.deepcopy(payload.get("usage"))
        if isinstance(usage, dict):
            # Preserve native usage as well as the existing transcript fields.
            for source, target in (("input_tokens", "prompt_tokens"),
                                   ("output_tokens", "completion_tokens"),
                                   ("input_tokens_details", "prompt_tokens_details"),
                                   ("output_tokens_details", "completion_tokens_details")):
                if source in usage:
                    usage[target] = copy.deepcopy(usage[source])
        return {
            "finish_reason": finish,
            "message": {"role": "assistant", "content": "\n".join(text) or None, "tool_calls": calls},
            "responses_output": copy.deepcopy(output), "response_id": payload.get("id"),
            "response_status": status, "incomplete_details": payload.get("incomplete_details"),
            "usage": usage, "model": payload.get("model"), "system_fingerprint": None,
        }
    except (ValueError, KeyError, TypeError):
        raise TransientInferenceError("model endpoint returned a malformed Responses envelope") from None
