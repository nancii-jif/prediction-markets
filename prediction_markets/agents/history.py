"""Conversation formatting and restoration; historical calls are never replayed."""

import copy
import json
from typing import Any

from .. import config
from ..integrations.model_protocol import parse_response
from .tools import parse_arguments


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _template_group(group: list[dict]) -> list[dict]:
    """HF templates take argument objects; the HTTP client encodes them again.

    Malformed arguments have already been rejected by dispatch. Keep such a
    complete group as quoted data, not a fake/repaired tool invocation. Raw
    responses/results remain unchanged in SQLite and in the private history.
    """
    result = copy.deepcopy(group)
    try:
        for message in result:
            for call in message.get("tool_calls", []):
                call["function"]["arguments"] = parse_arguments(call["function"]["arguments"])
    except (ValueError, TypeError, RecursionError):
        # Opaque reasoning cannot be replayed as user-visible quoted text.
        quoted = [{key: value for key, value in message.items() if key != "_responses"}
                  for message in group]
        return [{"role": "user", "content": _json({"previous_response_and_results": quoted})}]
    return result


def assistant_message(response: dict, settings: config.Settings) -> dict:
    message = parse_response(response)
    if (settings.model_provider == "openai" and settings.openai_api == "responses"
            and "responses_output" in response):
        usage = response.get("usage") or {}
        details = usage.get("output_tokens_details") if isinstance(usage, dict) else None
        message["_responses"] = {
            "output": copy.deepcopy(response["responses_output"]),
            "reasoning_tokens": details.get("reasoning_tokens", 0) if isinstance(details, dict) else 0,
        }
    return message


def complete_groups(groups: list[list[dict]]) -> list[list[dict]]:
    """Retain complete provider exchanges; quote interrupted groups as history.

    No function is dispatched here. A missing result is unknown, since a
    process could stop between an exchange commit and recording its result.
    """
    result = []
    for group in groups:
        if not group:
            continue
        calls = group[0].get("tool_calls", [])
        expected = [call["id"] for call in calls]
        actual = [message["tool_call_id"] for message in group if message["role"] == "tool"]
        if expected == actual:
            result.append(copy.deepcopy(group))
        else:
            visible = [{key: value for key, value in message.items() if key != "_responses"}
                       for message in group]
            result.append([{"role": "user", "content": json.dumps({
                "interrupted_previous_decision": visible,
                "notice": "This decision was interrupted. Calls without recorded results have unknown "
                          "execution status. No old calls will be replayed. Check current account, "
                          "orders and trade history before deciding afresh.",
            })}])
    return result


def recover_groups(conn, agent_id: str, settings: config.Settings) -> list[list[dict]]:
    """Legacy recovery from only this agent's chronological transcript."""
    groups, group = [], None
    for row in conn.execute(
        "SELECT entry_type, content_json FROM transcript_entries WHERE agent_id=? ORDER BY sequence_number",
        (agent_id,),
    ):
        kind, encoded = row
        content = json.loads(encoded)
        if kind == "model_response":
            try:
                group = [assistant_message(content, settings)]
            except ValueError:
                group = None
            else:
                groups.append(group)
        elif kind == "tool_result":
            call_id = content.get("tool_call_id")
            if group is not None and call_id is not None:
                if call_id in {call["id"] for call in group[0].get("tool_calls", [])}:
                    group.append({"role": "tool", "tool_call_id": call_id,
                                  "content": json.dumps(content["result"])})
            elif content.get("name") is None:
                groups.append([{"role": "user", "content": json.dumps(content["result"])}])
            elif group is not None and content.get("name") == "hold_for_now":
                group.append({"role": "user", "content": json.dumps({"hold_for_now": content["result"]})})
    return complete_groups(groups)
