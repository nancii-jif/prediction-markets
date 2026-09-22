"""Calendar boundaries and EOD carryover, with real elapsed costs and fake clocks."""

import asyncio
import json
from datetime import datetime

import pytest

from prediction_markets.runtime.schedule import TradingSchedule


def stamp(value):
    return int(datetime.fromisoformat(value).timestamp())


def finish(rig, participant):
    with pytest.raises(rig.finished):
        asyncio.run(participant.run())


def order(rig):
    return rig.call("submit_order", {"ticker": "MKT-01", "action": "buy", "order_type": "limit",
                                     "quantity": 1, "price_cents": 40})


@pytest.mark.parametrize("current,next_day,hours", [
    ("2026-03-07T08:00:00-05:00", "2026-03-08T08:00:00-04:00", 23),
    ("2026-10-31T08:00:00-04:00", "2026-11-01T08:00:00-05:00", 25),
    ("2026-09-18T08:00:00-04:00", "2026-09-19T08:00:00-04:00", 24),
])
def test_next_open_uses_local_dates_including_dst_and_weekends(current, next_day, hours):
    schedule = TradingSchedule("America/New_York", "08:00", "16:00")
    assert schedule.next_day_open(stamp(current)) == stamp(next_day)
    assert stamp(next_day) - stamp(current) == hours * 3600


@pytest.mark.parametrize("start,expected", [
    ("2026-09-19T07:00:00-04:00", "2026-09-19T08:00:00-04:00"),
    ("2026-09-19T16:00:00-04:00", "2026-09-20T08:00:00-04:00"),
    ("2026-09-19T23:00:00-04:00", "2026-09-20T08:00:00-04:00"),
])
def test_no_inference_before_open_or_at_and_after_close(participant_rig, start, expected):
    rig = participant_rig
    rig.advance(stamp(start) - rig.clock.now)
    async def first(request):
        assert rig.clock.now == stamp(expected)
        return rig.response(order(rig))
    participant, _ = rig.make(first)
    finish(rig, participant)
    assert rig.state()["activity_day_started_at"] == stamp(expected)


def test_1530_hold_finishes_at_1630_then_decides_afresh_at_0800(participant_rig):
    rig = participant_rig
    rig.advance(stamp("2026-09-19T15:30:00-04:00") - rig.clock.now)
    async def tomorrow(request):
        assert rig.clock.now == stamp("2026-09-20T08:00:00-04:00")
        brief = json.loads(request["messages"][-1]["content"])
        assert brief["private_note"] == "end of day"
        assert brief["session"]["next_action_at"] is None
        return rig.response(order(rig))
    participant, model = rig.make(rig.response(rig.call("hold_for_now", {"memory_note": "end of day"})),
                                  tomorrow)
    finish(rig, participant)
    assert rig.clock.sleeps == [3600, 15.5 * 3600]
    assert len(model.requests) == 3
    completions = [r for r in rig.entries() if r["entry_type"] == "timing"
                   and r["content"]["event"] == "cost_completed"]
    assert completions[0]["created_at"] == stamp("2026-09-19T16:30:00-04:00")


@pytest.mark.parametrize("name,args,minutes", [
    ("submit_order", {"ticker": "MKT-01", "action": "buy", "order_type": "limit",
                      "quantity": 1, "price_cents": 40}, 2),
    ("cancel_order", {"order_id": 999}, 2),
    ("get_orderbook", {"ticker": "MKT-01"}, 5),
    ("get_recent_trades", {"ticker": "MKT-01"}, 5),
    ("get_own_orders", {}, 5),
    ("get_account", {}, 5),
    ("search_news", {"query": "facts"}, 10),
])
def test_every_tool_can_finish_after_close_without_an_overnight_decision(participant_rig, name, args, minutes):
    rig = participant_rig
    rig.advance(stamp("2026-09-19T15:59:00-04:00") - rig.clock.now)
    async def news(**kwargs):
        return {"ok": True, "results": [{"content": "EOD data"}]}
    async def tomorrow(request):
        assert rig.clock.now == stamp("2026-09-20T08:00:00-04:00")
        assert any(m["role"] == "tool" for m in request["messages"])
        assert "news_tool_calls_remaining" not in json.loads(request["messages"][-1]["content"])
        return rig.response(order(rig))
    participant, _ = rig.make(rig.response(rig.call(name, args)), tomorrow)
    participant.tools._async_functions["search_news"] = news
    finish(rig, participant)
    assert rig.clock.sleeps[0] == minutes * 60
    assert sum(rig.clock.sleeps) == (16 * 60 + 1) * 60


def test_tool_ending_exactly_at_close_defers_to_next_open(participant_rig):
    rig = participant_rig
    rig.advance(stamp("2026-09-19T15:58:00-04:00") - rig.clock.now)
    participant, _ = rig.make(rig.response(order(rig)))
    finish(rig, participant)
    assert rig.clock.sleeps == [120, 16 * 3600]


def test_slow_network_response_after_close_is_retained_as_eod_data(participant_rig):
    rig = participant_rig
    rig.advance(stamp("2026-09-19T15:50:00-04:00") - rig.clock.now)
    async def slow_news(**kwargs):
        rig.advance(15 * 60)
        return {"ok": True, "results": [{"content": "arrived after close"}]}
    participant, model = rig.make(rig.response(rig.call("search_news", {"query": "facts"})))
    participant.tools._async_functions["search_news"] = slow_news
    finish(rig, participant)
    assert rig.clock.sleeps == [15 * 3600 + 55 * 60]
    assert "arrived after close" in json.dumps(model.requests[-1])
    result = next(r for r in rig.entries() if r["entry_type"] == "tool_result")
    assert result["created_at"] == stamp("2026-09-19T16:05:00-04:00")


def test_inference_that_crosses_close_never_executes_its_stale_order(participant_rig):
    rig = participant_rig
    rig.advance(stamp("2026-09-19T15:57:00-04:00") - rig.clock.now)
    async def late(request):
        rig.advance(4 * 60)
        return rig.response(order(rig))
    participant, model = rig.make(rig.response(order(rig)), late)
    finish(rig, participant)
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
    results = [r["content"]["result"] for r in rig.entries() if r["entry_type"] == "tool_result"]
    assert results[-1]["error"]["code"] == "session_closed"
    assert results[-1]["tool_call_executed"] is False
    assert json.loads(model.requests[-1]["messages"][-1]["content"])["observed_at"].startswith("2026-09-20T12:00")


def test_long_configured_hold_is_never_reset_by_an_opening(participant_rig):
    rig = participant_rig
    participant, _ = rig.make(rig.response(rig.call("hold_for_now")), rig.response(order(rig)),
                              settings=rig.tight(hold_for_now_minutes=25 * 60))
    finish(rig, participant)
    assert rig.clock.sleeps == [25 * 3600]
    assert rig.clock.now == stamp("2026-09-20T09:00:00-04:00")


def test_hold_until_next_day_does_not_mean_eight_hours(participant_rig):
    rig = participant_rig
    participant, _ = rig.make(rig.response(rig.call("hold_until_next_day")))
    finish(rig, participant)
    assert rig.clock.sleeps == [24 * 3600]
    assert rig.clock.now == stamp("2026-09-20T08:00:00-04:00")


def test_prompt_construction_crossing_close_never_starts_an_overnight_request(participant_rig, monkeypatch):
    rig = participant_rig
    rig.advance(stamp("2026-09-19T15:59:59-04:00") - rig.clock.now)
    participant, model = rig.make(rig.response(order(rig)))
    original = participant.build_messages
    delayed = False

    def slow_brief():
        nonlocal delayed
        brief = original()
        if not delayed:
            delayed = True
            rig.advance(2)
        return brief

    monkeypatch.setattr(participant, "build_messages", slow_brief)
    finish(rig, participant)
    first = json.loads(model.requests[0]["messages"][-1]["content"])
    assert first["observed_at"].startswith("2026-09-20T12:00")
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1


def test_market_cutoff_is_rechecked_when_a_delayed_order_executes(participant_rig):
    rig = participant_rig
    rig.conn.execute("UPDATE markets SET close_time = ? WHERE ticker = 'MKT-01'", (rig.clock.now + 60,))
    rig.conn.commit()
    participant, _ = rig.make(rig.response(rig.call("get_orderbook", {"ticker": "MKT-01"})),
                              rig.response(order(rig)))
    finish(rig, participant)
    assert rig.clock.sleeps == [300]
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
    results = [r["content"]["result"] for r in rig.entries() if r["entry_type"] == "tool_result"]
    assert results[-1]["error"]["code"] == "market_not_tradable"


def test_cancellation_during_cooldown_never_executes_the_queued_order(participant_rig):
    rig = participant_rig
    participant, model = rig.make(rig.response(order(rig)), rig.response(order(rig)), sleep=rig.stop_sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(participant.run())
    assert len(model.requests) == 2
    assert rig.state()["status"] == "cooldown"
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
