"""Shared pytest setup for the MVP migration.

Each new suite loads its settings explicitly and uses pytest's temporary-path
fixtures. This avoids the legacy suite's collection-time global configuration
and prevents tests from ever writing into ``runs/``.
"""

import asyncio
import copy
import json
from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest
import yaml

from prediction_markets import config
from prediction_markets.storage import database as database
from prediction_markets.agents import Participant
from prediction_markets.exchange import Exchange


@pytest.fixture(scope="session", autouse=True)
def fixed_test_config(tmp_path_factory):
    """Keep synthetic exchange scenarios independent of live run tuning."""
    values = yaml.safe_load((config.ROOT / "configs" / "mvp.yaml").read_text())
    values.update(
        market_count=10, participant_count=8, max_concurrent_inference=8,
        volume_24h_floor=500, uncertainty_price_range_cents=[10, 90],
        allowed_market_categories=[
            "Elections", "Politics", "Entertainment", "Commodities", "Climate and Weather",
            "Economics", "Mentions", "Financials", "Science and Technology",
        ],
    )
    directory = tmp_path_factory.mktemp("test-config")
    (directory / "mvp.yaml").write_text(yaml.safe_dump(values))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(config, "CONFIG_DIR", directory)
        yield


class ScriptFinished(Exception):
    """End an offline script without adding a production inference quota."""


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def __call__(self, **request):
        self.requests.append(copy.deepcopy(request))
        if not self.responses:
            raise ScriptFinished()
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if callable(response):
            return await response(request)
        return copy.deepcopy(response)


class ScriptedTokenizer:
    """Offline template stand-in only; production requires the model tokenizer."""
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["add_generation_prompt"] is True
        assert kwargs["enable_thinking"] is False
        assert len(kwargs["tools"]) == 9
        return [0] * (len(json.dumps([messages, kwargs["tools"]])) // 4)


@pytest.fixture
def participant_rig(tmp_path):
    # News tests mock every HTTP request.
    settings = config.load(config.CONFIG_DIR / "mvp.yaml")
    conn = database.init(tmp_path / "run.db", run_settings=settings)
    clock = SimpleNamespace(now=int(datetime.fromisoformat("2026-09-19T08:00:00-04:00").timestamp()),
                            mono=0.0, sleeps=[])
    database.store_market_cohort([
        {
            "ticker": f"MKT-{i:02d}", "event_ticker": f"EVENT-{i:02d}",
            "title": f"Market {i}", "rules": f"Full unabridged rules for market {i}.",
            "market_type": "binary", "status": "active",
            "close_time": clock.now + 7 * 86400, "expected_expiration_time": clock.now + 100 * 3_600,
            "latest_expiration_time": None, "last_successful_poll": clock.now, "payout_cents": None,
        }
        for i in range(1, 11)
    ])
    exchange = Exchange(conn, now=lambda: clock.now)
    counter = 0

    def call(name, arguments=None, *, raw=None):
        nonlocal counter
        counter += 1
        return {"id": f"call-{counter}", "type": "function", "function": {
            "name": name, "arguments": raw if raw is not None else json.dumps(arguments or {}),
        }}

    def response(*calls, content=None, finish=None):
        return {"finish_reason": finish or ("tool_calls" if calls else "stop"),
                "message": {"role": "assistant", "content": content, "tool_calls": list(calls)}}

    async def stop_sleep(seconds):
        clock.sleeps.append(seconds)
        raise asyncio.CancelledError()

    def advance(seconds):
        clock.mono += seconds
        clock.now += int(seconds)
        # The live feed would refresh these markets while elapsed time passes.
        conn.execute("UPDATE markets SET last_successful_poll = ?", (clock.now,))
        conn.commit()

    async def advance_sleep(seconds):
        clock.sleeps.append(seconds)
        advance(seconds)
        await asyncio.sleep(0)

    def make(*responses, agent_id="agent-01", **kwargs):
        fake = ScriptedModel(responses)
        arguments = dict(
            agent_id=agent_id, model=settings.model_name, exchange=exchange,
            conn=conn, settings=settings, infer=fake, tokenizer=ScriptedTokenizer(),
            now=lambda: clock.now, monotonic=lambda: clock.mono, sleep=advance_sleep,
        )
        arguments.update(kwargs)
        return Participant(**arguments), fake

    def state(agent_id="agent-01"):
        return dict(conn.execute("SELECT * FROM agent_state WHERE agent_id = ?", (agent_id,)).fetchone())

    def market_calls(agent_id="agent-01"):
        return {
            row["ticker"]: row["order_tool_calls"]
            for row in conn.execute(
                "SELECT ticker, order_tool_calls FROM agent_market_calls "
                "WHERE agent_id = ? AND order_tool_calls > 0", (agent_id,),
            )
        }

    def tight(**changes):
        """Override individual settings for an offline scenario."""
        return replace(settings, **changes)

    def entries(agent_id="agent-01"):
        return [
            {**dict(row), "content": json.loads(row["content_json"])}
            for row in conn.execute(
                "SELECT * FROM transcript_entries WHERE agent_id = ? ORDER BY sequence_number", (agent_id,),
            )
        ]

    try:
        yield SimpleNamespace(
            settings=settings, conn=conn, clock=clock, exchange=exchange,
            call=call, response=response, make=make, state=state, entries=entries,
            market_calls=market_calls, tight=tight,
            advance=advance, advance_sleep=advance_sleep, stop_sleep=stop_sleep, finished=ScriptFinished,
        )
    finally:
        database.close()
