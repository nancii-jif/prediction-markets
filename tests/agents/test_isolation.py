"""Private participant state and concurrent loops on the shared local exchange."""

import asyncio
import json
import re
from pathlib import Path

import pytest

from prediction_markets.markets import selection as live_markets
from prediction_markets.agents.tools import ParticipantTools


async def cancel_tasks(*tasks):
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def test_kalshi_access_stays_outside_agents_and_exchange():
    import ast
    from prediction_markets import config
    allowed = {"integrations/kalshi.py", "markets/update.py", "analysis/dataset.py",
               "analysis/prices.py", "kalshi_client.py"}
    offenders = []
    package = config.ROOT / "prediction_markets"
    for path in package.rglob("*.py"):
        if str(path.relative_to(package)) in allowed:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                names.append(getattr(node, "module", "") or "")
                if any("kalshi" in name for name in names):
                    offenders.append(str(path.relative_to(package)))
    assert offenders == []


def test_external_prices_and_other_agents_state_cannot_enter_briefs_or_tools(participant_rig):
    rig = participant_rig
    raw = {
        "ticker": "MKT-01", "event_ticker": "EVENT-01", "market_type": "binary",
        "title": "Updated title", "rules_primary": "Full mandatory rules.",
        "status": "active", "close_time": "2033-06-01T00:00:00Z",
        "expected_expiration_time": "2033-08-01T00:00:00Z",
        "yes_bid_dollars": "0.3719", "yes_ask_dollars": "0.4197",
        "last_price_dollars": "0.3953", "volume_24h_fp": "98765.43",
    }
    rig.exchange.apply_market_observation(live_markets.market_row(raw, rig.clock.now))
    first, _ = rig.make(rig.response(rig.call("hold_for_now", {"memory_note": "SECRET_FIRST_AGENT"})))
    with pytest.raises(rig.finished):
        asyncio.run(first.run())
    second, _ = rig.make(agent_id="agent-02")
    brief = json.dumps(second.build_messages())
    assert "SECRET_FIRST_AGENT" not in brief
    assert "agent-01" not in brief
    assert "Updated title" in brief and "Full mandatory rules." in brief
    for field in ("yes_bid_dollars", "yes_ask_dollars", "last_price_dollars", "volume_24h_fp"):
        assert field not in brief and raw[field] not in brief
    assert rig.state("agent-01")["private_note"] == "SECRET_FIRST_AGENT"
    assert rig.state("agent-02")["private_note"] == ""
    assert rig.entries("agent-02") == []

    order = rig.exchange.submit_order("agent-01", "MKT-02", "sell", "limit", 2, 40)["order"]
    tools = ParticipantTools(rig.exchange, "agent-02", rig.settings)
    assert tools.dispatch("get_own_orders", "{}").result["orders"] == []
    spoof = tools.dispatch("get_account", '{"agent_id":"agent-01"}')
    assert spoof.dispatched is False
    cancel = tools.dispatch("cancel_order", json.dumps({"order_id": order["order_id"]}))
    assert cancel.result["error"]["code"] == "not_owner"
    public = json.dumps(tools.dispatch("get_orderbook", '{"ticker":"MKT-02"}').result)
    assert "agent-01" not in public and "order_id" not in public


def test_sleeping_agent_has_no_inference_while_another_agent_fills_its_order(participant_rig):
    rig = participant_rig
    initial = rig.clock.now

    async def scenario():
        asleep = asyncio.Event()
        durations = []

        async def long_sleep(seconds):
            durations.append(seconds)
            asleep.set()
            await asyncio.Future()

        async def hold_after_thinking(request):
            rig.advance(120)
            return rig.response(rig.call("hold_until_next_day", {"memory_note": "private thesis"}))

        sleeper, sleepy_model = rig.make(rig.response(rig.call("submit_order", {
            "ticker": "MKT-01", "action": "sell", "order_type": "limit", "quantity": 10, "price_cents": 40,
        })), hold_after_thinking, sleep=long_sleep)
        task = asyncio.create_task(sleeper.run())
        try:
            await asyncio.wait_for(asleep.wait(), 1)
            assert not rig.conn.in_transaction
            before = rig.state()
            buyer, buyer_model = rig.make(
                rig.response(rig.call("submit_order", {"ticker": "MKT-01", "action": "buy",
                             "order_type": "market", "quantity": 10})),
                rig.response(rig.call("get_recent_trades", {"ticker": "MKT-01"})),
                agent_id="agent-02",
            )
            with pytest.raises(rig.finished):
                await buyer.run()
            assert len(sleepy_model.requests) == 2 and len(buyer_model.requests) == 3
            assert durations == [86400 - 120]
            assert not task.done() and rig.state() == before
            account = rig.exchange.get_account("agent-01")
            assert account["cash_cents"] == 99400
            assert account["positions"][0]["signed_qty"] == -10
            assert rig.conn.execute("SELECT quantity FROM trades").fetchone()[0] == 10
            buyer_transcript = json.dumps([entry["content"] for entry in rig.entries("agent-02")])
            assert "private thesis" not in buyer_transcript and "agent-01" not in buyer_transcript
        finally:
            await asyncio.wait_for(cancel_tasks(task), 1)
        assert rig.market_calls() == {"MKT-01": 1}
        assert rig.state()["activity_day_started_at"] == initial
        assert rig.state()["scheduled_wake_at"] == initial + 86400
        assert rig.state()["private_note"] == "private thesis"

    asyncio.run(scenario())


def test_overnight_hold_blocks_inference_then_observes_fresh_state(participant_rig):
    rig = participant_rig

    async def scenario():
        sleeping, release = asyncio.Event(), asyncio.Event()
        initial = rig.clock.now

        async def controlled_sleep(seconds):
            assert seconds == 86400
            sleeping.set()
            await release.wait()
            rig.advance(seconds)

        participant, fake = rig.make(rig.response(rig.call("hold_until_next_day")), sleep=controlled_sleep)
        task = asyncio.create_task(participant.run())
        try:
            await asyncio.wait_for(sleeping.wait(), 1)
            for _ in range(3):
                await asyncio.sleep(0)
            assert len(fake.requests) == 1
            assert rig.state()["activity_day_started_at"] == initial
            rig.conn.execute("UPDATE markets SET title = 'NEW DEFINITION' WHERE ticker = 'MKT-01'")
            rig.conn.commit()
            release.set()
            with pytest.raises(rig.finished):
                await asyncio.wait_for(task, 1)
            assert len(fake.requests) == 2
            assert rig.state()["activity_day_started_at"] == initial + 86400
            brief = json.loads(fake.requests[-1]["messages"][-1]["content"])
            assert brief["markets"][0]["title"] == "NEW DEFINITION"
            assert "order_tool_calls_remaining_by_market" not in brief
        finally:
            await cancel_tasks(task)

    asyncio.run(scenario())


def test_slow_inference_does_not_block_another_agent(participant_rig):
    rig = participant_rig

    async def scenario():
        pending = asyncio.Event()

        async def slow_response(request):
            pending.set()
            await asyncio.Future()

        slow, slow_model = rig.make(slow_response)
        slow_task = asyncio.create_task(slow.run())
        try:
            await asyncio.wait_for(pending.wait(), 1)
            fast, fast_model = rig.make(
                rig.response(rig.call("submit_order", {"ticker": "MKT-01", "action": "buy", "order_type": "limit", "quantity": 1, "price_cents": 30})),
                rig.response(rig.call("hold_for_now")),
                agent_id="agent-02", sleep=rig.stop_sleep,
            )
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(fast.run(), 1)
            assert not slow_task.done()
            assert len(slow_model.requests) == 1 and len(fast_model.requests) == 2
            assert rig.conn.execute("SELECT agent_id FROM orders").fetchone()[0] == "agent-02"
        finally:
            await asyncio.wait_for(cancel_tasks(slow_task), 1)

    asyncio.run(scenario())


def test_eight_independent_requests_can_be_in_flight_together(participant_rig):
    rig = participant_rig

    async def scenario():
        all_pending, release = asyncio.Event(), asyncio.Event()
        seen = set()

        async def shared_model(request):
            agent_id = json.loads(request["messages"][-1]["content"])["account"]["agent_id"]
            assert agent_id not in seen
            seen.add(agent_id)
            if len(seen) == 8:
                all_pending.set()
            await release.wait()
            return rig.response(rig.call("hold_until_next_day"))

        participants = [rig.make(shared_model, agent_id=agent_id, sleep=rig.stop_sleep) for agent_id in rig.settings.agent_ids]
        tasks = [asyncio.create_task(participant.run()) for participant, _ in participants]
        try:
            await asyncio.wait_for(all_pending.wait(), 1)
            assert all(not task.done() for task in tasks)
            assert len(seen) == 8
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
            assert all(isinstance(result, asyncio.CancelledError) for result in results)
            assert all(len(fake.requests) == 1 for _, fake in participants)
            assert all(rig.state(agent_id)["shared_tool_calls"] == 0 for agent_id in seen)
        finally:
            await cancel_tasks(*tasks)

    asyncio.run(scenario())
