"""Identity binding, dispatch accounting, and local wait validation."""

import json

import pytest

from prediction_markets.agents.tools import ParticipantTools


def test_only_nine_tools_are_registered_and_none_accept_identity(participant_rig):
    rig = participant_rig
    tools = ParticipantTools(rig.exchange, "agent-01", rig.settings)
    assert {tool["function"]["name"] for tool in tools.schemas} == {
        "submit_order", "cancel_order", "get_orderbook", "get_recent_trades",
        "get_own_orders", "get_account", "hold_for_now", "hold_until_next_day", "search_news",
    }
    for tool in tools.schemas:
        parameters = tool["function"]["parameters"]
        assert parameters["additionalProperties"] is False
        assert "agent_id" not in parameters["properties"]


@pytest.mark.parametrize("name, arguments", [
    ("unknown", "{}"), ("get_account", "[]"), ("get_account", "not json"),
    ("get_account", '{"agent_id":"agent-02"}'),
    ("submit_order", '{"ticker":"MKT-01"}'),
    ("get_account", '{"x":1,"x":2}'), ("get_account", '{"x":NaN}'),
    ("get_account", {"agent_id": "agent-02"}),
])
def test_invalid_shape_is_rejected_before_dispatch(participant_rig, name, arguments):
    rig = participant_rig
    outcome = ParticipantTools(rig.exchange, "agent-01", rig.settings).dispatch(name, arguments)
    assert outcome.dispatched is False
    assert outcome.result["ok"] is False
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


@pytest.mark.parametrize("name, arguments", [
    ("get_account", {}), ("get_own_orders", {}),
    ("get_orderbook", {"ticker": "MKT-01"}),
    ("get_recent_trades", {"ticker": "MKT-01"}),
    ("cancel_order", {"order_id": 999}),
    ("submit_order", {"ticker": "MKT-01", "action": "buy", "order_type": "limit", "quantity": 0, "price_cents": 40}),
])
def test_all_six_tools_dispatch_including_business_and_value_errors(participant_rig, name, arguments):
    rig = participant_rig
    outcome = ParticipantTools(rig.exchange, "agent-01", rig.settings).dispatch(name, json.dumps(arguments))
    assert outcome.dispatched is True
    assert outcome.wait_seconds is None
    if name in {"cancel_order", "submit_order"}:
        assert outcome.result["ok"] is False


@pytest.mark.parametrize("name, arguments, duration, next_day", [
    ("hold_for_now", {}, 3600, False),
    ("hold_for_now", {"memory_note": "private"}, 3600, False),
    ("hold_until_next_day", {"memory_note": ""}, None, True),
])
def test_wait_tools_return_directives_without_sleep_or_state_mutation(participant_rig, name, arguments, duration, next_day):
    rig = participant_rig
    before = rig.state()
    outcome = ParticipantTools(rig.exchange, "agent-01", rig.settings).dispatch(name, json.dumps(arguments))
    assert outcome.dispatched is True
    assert outcome.wait_seconds == duration and outcome.next_day is next_day
    assert outcome.memory_note == arguments.get("memory_note")
    assert rig.state() == before
    assert rig.clock.sleeps == []


@pytest.mark.parametrize("name, arguments", [
    ("hold_for_now", {"seconds": 0}), ("hold_for_now", {"seconds": 60}),
    ("hold_for_now", {"minutes": 1}), ("hold_for_now", {"memory_note": 12}),
    ("hold_until_next_day", {"seconds": 8}),
    ("hold_until_next_day", {"memory_note": "x" * 2001}),
    ("submit_quote", {}), ("wait", {}), ("wait_until_next_day", {}),
])
def test_invalid_wait_has_no_effect(participant_rig, name, arguments):
    rig = participant_rig
    outcome = ParticipantTools(rig.exchange, "agent-01", rig.settings).dispatch(name, json.dumps(arguments))
    assert outcome.dispatched is False and outcome.wait_seconds is None
    assert outcome.result["ok"] is False
