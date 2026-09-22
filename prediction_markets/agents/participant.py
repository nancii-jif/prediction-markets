"""Independent participant tool loops; inference and tokenizer are injected.

No model hosting, network client, rounds, or shared conversation lives here.
All storage/exchange operations are synchronous on the event-loop thread;
inference, news searches and cancellable sleeps are awaited.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import math
import sqlite3
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from ..config import Settings
from ..exchange import Exchange
from ..runtime.schedule import TradingSchedule
from .tools import ParticipantTools, ToolOutcome, error, parse_arguments

log = logging.getLogger(__name__)


from ..integrations.model_protocol import (
    ContextLimitError, PermanentInferenceError, TransientInferenceError, parse_response,
)
from .prompt import build_prompt
from .history import _json, _template_group, assistant_message, complete_groups, recover_groups


class Participant:
    """One portfolio, private memory, and at most one pending model request.

    infer is an async callable accepting messages, tools, model, max_tokens and
    enable_thinking; the runner binds endpoint configuration.
    tokenizer provides apply_chat_template for token counting: the serving HF
    tokenizer for vLLM, or the local context estimator for OpenAI.
    Tests inject a scripted model, tokenizer, time functions and cancellable sleep.
    """

    def __init__(
        self, agent_id: str, model: str, prompt: str | None = None, *,
        exchange: Exchange, conn: sqlite3.Connection, settings: Settings,
        infer: Callable[..., Awaitable[dict]], tokenizer: Any,
        now: Callable[[], int] | None = None,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ):
        if not exchange.get_account(agent_id)["ok"]:
            raise ValueError("participant must have an initialized account")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty string")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError("prompt must be text")
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise PermanentInferenceError("the model tokenizer with a chat template is required")
        self.agent_id, self.model = agent_id, model
        self.prompt = build_prompt(settings) if prompt is None else prompt
        self.exchange, self._conn, self.settings = exchange, conn, settings
        self._infer, self._tokenizer = infer, tokenizer
        self._now = now if now is not None else lambda: int(time.time())
        self._monotonic = monotonic if monotonic is not None else time.monotonic
        self._sleep = sleep if sleep is not None else asyncio.sleep
        self.schedule = TradingSchedule(settings.trading_timezone, settings.trading_day_start,
                                        settings.trading_day_end)
        self.tools = ParticipantTools(exchange, agent_id, settings)
        self.sampling = settings.sampling(agent_id)
        # The system prompt and tool schemas are byte-identical for every
        # participant and every turn, so the server caches that prefix. Measure
        # it once and give the working budget on top, keeping room for history
        # independent of how long the prompt itself grows.
        self.fixed_prefix_tokens = self._count(
            [{"role": "system", "content": self.prompt}, {"role": "user", "content": ""}]
        )
        self.input_token_limit = self.fixed_prefix_tokens + settings.working_token_budget
        if self.input_token_limit + settings.max_output_tokens > settings.server_context_tokens:
            raise ContextLimitError(
                f"fixed prompt prefix ({self.fixed_prefix_tokens} tokens) plus the working "
                f"budget and output budget exceed the {settings.server_context_tokens}-token "
                "server context"
            )
        self.private_note = ""
        self.activity_day_started_at = None
        # Keep daily activity counters for analysis; they never limit dispatch.
        self.order_tool_calls: dict[str, int] = {}
        self.shared_tool_calls = 0
        self.news_tool_calls = 0
        self.scheduled_wake_at = None
        self.next_action_at = None
        self.status = "ready"
        self._deadline: float | None = None
        self._cost_session_close: int | None = None
        self._holding = False
        self._defer_inference = False
        self._groups: list[list[dict]] = []
        self._sequence = 0
        self._started = False
        self._resumed = False
        self._continuation = None

    def restore(self, parent_run_id: str) -> None:
        """Restore private context and absolute deadlines, never execute history."""

        if self._started or self._resumed:
            raise RuntimeError("participant has already started or restored")
        state = self._conn.execute("SELECT * FROM agent_state WHERE agent_id=?", (self.agent_id,)).fetchone()
        for name in ("private_note", "activity_day_started_at", "shared_tool_calls", "news_tool_calls",
                     "scheduled_wake_at", "next_action_at", "status"):
            setattr(self, name, state[name])
        self.order_tool_calls = dict(self._conn.execute(
            "SELECT ticker, order_tool_calls FROM agent_market_calls WHERE agent_id=?", (self.agent_id,),
        ))
        self._sequence = self._conn.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) FROM transcript_entries WHERE agent_id=?", (self.agent_id,),
        ).fetchone()[0]
        checkpoint = self._conn.execute("SELECT * FROM agent_checkpoints WHERE agent_id=?", (self.agent_id,)).fetchone()
        saved = json.loads(checkpoint["state_json"]) if checkpoint else {}
        if saved and saved.get("version") != 1:
            raise ValueError("unsupported participant checkpoint version")
        if checkpoint and checkpoint["sequence_number"] == self._sequence:
            self.prompt = saved["prompt"]
            self._groups = complete_groups(saved["groups"])
        else:
            self._groups = recover_groups(self._conn, self.agent_id, self.settings)
        self.fixed_prefix_tokens = self._count(
            [{"role": "system", "content": self.prompt}, {"role": "user", "content": ""}]
        )
        self.input_token_limit = self.fixed_prefix_tokens + self.settings.working_token_budget
        if self.input_token_limit + self.settings.max_output_tokens > self.settings.server_context_tokens:
            raise ContextLimitError("saved prompt and configured budgets exceed the server context")
        # Downtime counts toward elapsed costs. A future overnight wake is also
        # a deadline; monotonic clocks themselves cannot survive a process.
        deadline = self.next_action_at or self.scheduled_wake_at
        if deadline is not None:
            self.next_action_at = self.scheduled_wake_at = deadline
            self._deadline = self._monotonic() + max(0, deadline - self._now())
            self._cost_session_close = saved.get("cost_session_close") or self.schedule.session_at(self._now()).closes_at
            self._holding = saved.get("holding", self.status in {"holding", "overnight"})
            # Begin with a fresh observation after any outstanding pre-resume
            # cost, rather than advancing an interrupted model decision.
            self._defer_inference = True
        self._resumed = True
        self._continuation = {
            "parent_run_id": parent_run_id,
            "notice": "Continuation of the same experiment. Cash, positions and history carry over; "
                      "no new endowment. Completed actions are not repeated. Orders canceled at prior "
                      "shutdown stay canceled; inspect current orders before replacing quotes. Only currently "
                      "tradable original markets appear in markets; other holdings remain in your account.",
        }
        self._save("timing", {"event": "run_resumed", "parent_run_id": parent_run_id})

    def _save(self, entry_type: str | None = None, content: Any = None) -> int | None:
        if self._conn.in_transaction:
            raise RuntimeError("participant storage cannot enter another transaction")
        with self._conn:
            self._conn.execute(
                "UPDATE agent_state SET private_note = ?, activity_day_started_at = ?, "
                "shared_tool_calls = ?, news_tool_calls = ?, "
                "scheduled_wake_at = ?, next_action_at = ?, status = ? WHERE agent_id = ?",
                (self.private_note, self.activity_day_started_at, self.shared_tool_calls,
                 self.news_tool_calls, self.scheduled_wake_at, self.next_action_at,
                 self.status, self.agent_id),
            )
            self._conn.executemany(
                "INSERT INTO agent_market_calls (agent_id, ticker, order_tool_calls) "
                "VALUES (?, ?, ?) ON CONFLICT(agent_id, ticker) DO UPDATE SET "
                "order_tool_calls = excluded.order_tool_calls",
                ((self.agent_id, ticker, used) for ticker, used in self.order_tool_calls.items()),
            )
            if entry_type is not None:
                cursor = self._conn.execute(
                    "INSERT INTO transcript_entries "
                    "(agent_id, sequence_number, entry_type, content_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (self.agent_id, self._sequence + 1, entry_type, _json(content), self._now()),
                )
            self._conn.execute(
                "INSERT INTO agent_checkpoints (agent_id, sequence_number, state_json) VALUES (?, ?, ?) "
                "ON CONFLICT(agent_id) DO UPDATE SET sequence_number=excluded.sequence_number, "
                "state_json=excluded.state_json",
                (self.agent_id, self._sequence + int(entry_type is not None), _json({
                    "version": 1, "prompt": self.prompt, "groups": self._groups,
                    "cost_session_close": self._cost_session_close,
                    "holding": self._holding, "defer_inference": self._defer_inference,
                })),
            )
        if entry_type is not None:
            self._sequence += 1
            return cursor.lastrowid
        return None

    def _count(self, messages: list[dict]) -> int:
        return len(self._tokenizer.apply_chat_template(
            messages, tools=self.tools.schemas, tokenize=True,
            add_generation_prompt=True, enable_thinking=self.settings.enable_thinking,
        ))

    def build_messages(self) -> list[dict]:
        """Fresh observations plus complete private groups, counted with tools."""
        brief = {
            "observed_at": datetime.fromtimestamp(self._now(), timezone.utc).isoformat(),
            "markets": self.exchange.assigned_markets(self.agent_id),
            "account": self.exchange.get_account(self.agent_id),
            "private_note": self.private_note,
            "activity_day_started_at": self.activity_day_started_at,
            "session": {
                "timezone": self.settings.trading_timezone,
                "opens_at": self.schedule.session_at(self._now()).opens_at,
                "closes_at": self.schedule.session_at(self._now()).closes_at,
                "next_action_at": self.next_action_at,
            },
        }
        if self._resumed:
            brief["continuation"] = self._continuation
            brief["unavailable_markets"] = [
                {key: market[key] for key in ("ticker", "status", "settled")}
                for market in brief["markets"] if not market["tradable"]
            ]
            brief["markets"] = [market for market in brief["markets"] if market["tradable"]]
        while True:
            messages = [{"role": "system", "content": self.prompt}]
            messages.extend(message for group in self._groups for message in _template_group(group))
            messages.append({"role": "user", "content": _json(brief)})
            if self._count(messages) <= self.input_token_limit:
                return copy.deepcopy(messages)
            if not self._groups:
                raise ContextLimitError(
                    "mandatory prompt, market rules, account and tools exceed the "
                    f"{self.input_token_limit}-token input limit"
                )
            self._groups.pop(0)

    def _bucket(self, name: str, values: dict) -> str | None:
        """Attribute activity to a market where possible; never enforce a quota."""
        if name == "submit_order":
            ticker = values.get("ticker")
            # An invented ticker cannot be stored against a real market.
            return ticker if self.exchange.has_market(ticker) else None
        if name == "cancel_order":
            return self.exchange.order_ticker(self.agent_id, values.get("order_id"))
        return None

    def _schedule(self, seconds: int, *, holding: bool = False, defer_inference: bool = False) -> None:
        # A retry must never shorten a tool cost or an unfinished hold.
        now, mono = self._now(), self._monotonic()
        self._deadline = max(self._deadline or mono, mono + seconds)
        self.next_action_at = now + math.ceil(self._deadline - mono)
        self.scheduled_wake_at = self.next_action_at
        self._cost_session_close = self.schedule.session_at(now).closes_at
        self._holding = self._holding or holding
        self._defer_inference = self._defer_inference or defer_inference
        self._save("timing", {"event": "action_scheduled", "cost_seconds": seconds,
                              "next_action_at": self.next_action_at})

    async def _finish_cooldown(self) -> None:
        if self._deadline is None:
            return
        self.status = "holding" if self._holding else "cooldown"
        self._save()
        while (remaining := self._deadline - self._monotonic()) > 0:
            await self._sleep(remaining)
        self._deadline = None
        self._cost_session_close = None
        self._holding = self._defer_inference = False
        self.next_action_at = self.scheduled_wake_at = None
        self.status = "ready"
        self._save("timing", {"event": "cost_completed"})

    async def _prepare_inference(self) -> None:
        now = self._now()
        session = self.schedule.session_at(now)
        # Costs crossing close finish before the next opening and fresh brief.
        if self._deadline is not None and (
            self._defer_inference or not session.contains(now)
            or self.next_action_at >= self._cost_session_close
            or now >= self._cost_session_close
        ):
            await self._finish_cooldown()
        while not self.schedule.session_at(self._now()).contains(self._now()):
            self.status = "overnight"
            self.scheduled_wake_at = int(self.schedule.open_at_or_after(self._now()))
            self._save("timing", {"event": "awaiting_open", "scheduled_wake_at": self.scheduled_wake_at})
            await self._sleep(max(0, self.scheduled_wake_at - self._now()))
        session = self.schedule.session_at(self._now())
        if self.activity_day_started_at != session.opens_at:
            self.order_tool_calls = dict.fromkeys(self.order_tool_calls, 0)
            self.shared_tool_calls = self.news_tool_calls = 0
            self.activity_day_started_at = session.opens_at
            self._save("timing", {"event": "session_started", "opens_at": session.opens_at,
                                  "closes_at": session.closes_at})
        self.scheduled_wake_at = self.next_action_at
        self.status = "ready"
        self._save()

    def _tool_result(self, group: list[dict], call: dict, outcome: ToolOutcome) -> None:
        result = {**outcome.result, "tool_call_executed": outcome.dispatched}
        group.append({"role": "tool", "tool_call_id": call["id"], "content": _json(result)})
        self._save("tool_result", {
            "tool_call_id": call["id"], "name": call["function"]["name"], "result": result,
        })

    def _record_response(self, response: Any) -> Any:
        try:
            encoded = _json(response)
        except (TypeError, ValueError):
            encoded = _json({"invalid_response": repr(response)})
        response = json.loads(encoded)
        self._save("model_response", response)
        return response

    async def _handle_response(self, response: Any, *, record: bool = True,
                               allow_dispatch: bool = True) -> None:
        if record:
            response = self._record_response(response)
        execution_session = self.schedule.session_at(self._now())
        try:
            message = parse_response(response)
        except ValueError as exc:
            log.warning("%s rejected model response: %s", self.agent_id, exc)
            rejection = error("invalid_model_response", str(exc))
            self._groups.append([{"role": "user", "content": _json(rejection)}])
            self._save("tool_result", {"name": None, "result": {**rejection, "tool_call_executed": False}})
            if allow_dispatch:
                self._schedule(self.settings.inference_retry_seconds, defer_inference=True)
            return
        message = assistant_message(response, self.settings)
        group = [message]
        self._groups.append(group)
        calls = message.get("tool_calls", [])
        if not allow_dispatch:
            for call in calls:
                self._tool_result(group, call, ToolOutcome(error(
                    "session_closed", "Not executed: this decision's session ended. Decide afresh next session.",
                )))
            return
        if not calls:
            self._schedule(self.settings.hold_for_now_minutes * 60, holding=True)
            result = {"ok": True, "implicit": True, "next_action_at": self.next_action_at}
            group.append({"role": "user", "content": _json({"hold_for_now": result})})
            self._save("tool_result", {"name": "hold_for_now", "result": result})
            return
        stop_reason = None
        dispatched = False
        action_context = {}

        def count_dispatch(name, values):
            fired_at = self._now()
            action_entry_id = self._save("timing", {
                "event": "tool_dispatched", "tool_call_id": call["id"],
                "tool": name, "arguments": values, "fired_at": fired_at,
            })
            action_context.update(fired_at=fired_at, action_entry_id=action_entry_id,
                                  agent_id=self.agent_id, tool=name)
            if name in {"hold_for_now", "hold_until_next_day"}:
                return None
            if name == "search_news":
                self.news_tool_calls += 1
            else:
                bucket = self._bucket(name, values)
                if bucket is None:
                    self.shared_tool_calls += 1
                else:
                    self.order_tool_calls[bucket] = self.order_tool_calls.get(bucket, 0) + 1
            # Anchor the cost before invoking the tool, including network I/O.
            self._schedule(self.tools.time_cost_seconds(name))
            self.status = "tool_running"
            self._save()
            return None

        for call in calls:
            function = call["function"]
            if stop_reason is not None:
                outcome = ToolOutcome(error(stop_reason, "Not executed: this response's dispatch has ended."))
            elif not execution_session.contains(self._now()):
                stop_reason = "session_closed"
                outcome = ToolOutcome(error(stop_reason, "Not executed: the session ended before dispatch."))
            else:
                action_context = {}
                with self.exchange.action_context(action_context):
                    outcome = await self.tools.dispatch_async(
                        function["name"], function["arguments"], on_dispatch=count_dispatch,
                    )
                if outcome.dispatched:
                    dispatched = True
                    stop_reason = "one_tool_call_per_response"
                    if outcome.next_day or outcome.wait_seconds is not None:
                        seconds = (self.schedule.next_day_open(self._now()) - self._now()
                                   if outcome.next_day else outcome.wait_seconds)
                        self._schedule(seconds, holding=True, defer_inference=outcome.next_day)
                        if outcome.memory_note is not None:
                            self.private_note = outcome.memory_note
                    outcome.result["next_action_at"] = self.next_action_at
                    outcome.result["session_closes_at"] = self._cost_session_close
                    self.status = "holding" if self._holding else "cooldown"
                log.info("%s tool %s: ok=%s dispatched=%s", self.agent_id,
                         function["name"], outcome.result["ok"], outcome.dispatched)
            self._tool_result(group, call, outcome)
        if not dispatched:
            self._schedule(self.settings.inference_retry_seconds, defer_inference=True)

    async def run(self) -> None:
        """Overlap one inference with the previous action's elapsed time cost."""
        state = self._conn.execute(
            "SELECT activity_day_started_at, status FROM agent_state WHERE agent_id = ?", (self.agent_id,),
        ).fetchone()
        if self._started or state is None or (not self._resumed and (state[0] is not None or state[1] != "ready")):
            raise RuntimeError("participant loop has already started; restore explicitly before resuming")
        self._started = True
        while not self.exchange.all_markets_settled():
            await self._prepare_inference()
            if self.exchange.all_markets_settled():
                return
            if self._resumed and not any(m["tradable"] for m in self.exchange.assigned_markets(self.agent_id)):
                await self._sleep(self.settings.market_poll_interval_seconds)
                continue
            session = self.schedule.session_at(self._now())
            messages = self.build_messages()
            if not session.contains(self._now()):
                # Token counting/context construction can itself cross close.
                continue
            self.status = "thinking"
            started_at, started_mono = self._now(), self._monotonic()
            self._save("timing", {"event": "inference_started", "next_action_at": self.next_action_at})
            try:
                response = await asyncio.wait_for(
                    self._infer(
                        messages=messages, tools=copy.deepcopy(self.tools.schemas), model=self.model,
                        max_tokens=self.settings.max_output_tokens,
                        enable_thinking=self.settings.enable_thinking,
                        sampling=dict(self.sampling),
                    ),
                    timeout=self.settings.request_timeout_seconds,
                )
            except (TransientInferenceError, asyncio.TimeoutError, ConnectionError) as exc:
                log.warning("%s inference failed (%s); waiting before retry", self.agent_id, type(exc).__name__)
                self._schedule(self.settings.inference_retry_seconds, defer_inference=True)
                continue
            response = self._record_response(response)
            self._save("timing", {"event": "inference_completed", "started_at": started_at,
                                  "elapsed_seconds": self._monotonic() - started_mono})
            if self.exchange.all_markets_settled():
                return
            await self._finish_cooldown()
            if self.exchange.all_markets_settled():
                return
            # A late inference cannot execute a decision from another session.
            await self._handle_response(response, record=False,
                                        allow_dispatch=session.contains(self._now()))
