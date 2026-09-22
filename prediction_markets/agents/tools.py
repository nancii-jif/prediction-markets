"""Shared participant tools, with identity and time costs bound by the harness."""

from __future__ import annotations

import inspect
import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from ..config import Settings
from ..exchange import Exchange
from ..integrations.tavily import NEWS_TOOL_SCHEMA, asearch_news, search_news


def error(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate argument key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def parse_arguments(arguments: str) -> dict:
    """Strict JSON used for dispatch and for rendering previous tool calls."""
    if not isinstance(arguments, str):
        raise ValueError("tool arguments must be a JSON object string")
    values = json.loads(arguments, object_pairs_hook=_unique_object,
                        parse_constant=_invalid_constant, parse_float=_finite_float)
    if not isinstance(values, dict):
        raise ValueError("tool arguments must decode to an object")
    return values


@dataclass
class ToolOutcome:
    result: dict
    dispatched: bool = False
    wait_seconds: int | None = None
    next_day: bool = False
    memory_note: str | None = None


class ParticipantTools:
    """Participant owns counters, memory and sleep; only news uses async I/O."""

    def __init__(self, exchange: Exchange, agent_id: str, settings: Settings):
        self._settings = settings
        self._functions = {
            "submit_order": partial(exchange.submit_order, agent_id),
            "cancel_order": partial(exchange.cancel_order, agent_id),
            "get_orderbook": exchange.get_orderbook,
            "get_recent_trades": exchange.get_recent_trades,
            "get_own_orders": partial(exchange.get_own_orders, agent_id),
            "get_account": partial(exchange.get_account, agent_id),
            "search_news": partial(search_news, api_key_env=settings.tavily_api_key_env),
            "hold_for_now": self._hold_for_now,
            "hold_until_next_day": self._hold_until_next_day,
        }
        self.schemas = self._schemas()
        self._async_functions = {
            "search_news": partial(asearch_news, api_key_env=settings.tavily_api_key_env),
        }

    def _schemas(self) -> list[dict]:
        ticker = {"type": "string", "description": "Ticker from your assigned markets."}
        note = {
            "type": "string",
            "maxLength": self._settings.private_note_max_chars,
            "description": "Optional private note; replaces your previous note.",
        }
        specifications = [
            ("submit_order", "Submit a funded YES order; integer cents and quantities.", {
                "ticker": ticker,
                "action": {"type": "string", "enum": ["buy", "sell"]},
                "order_type": {"type": "string", "enum": ["limit", "market"]},
                "quantity": {"type": "integer", "minimum": 1},
                "price_cents": {"type": "integer", "minimum": 1, "maximum": 99,
                                "description": "Required for limit orders; omit for market orders."},
            }, ["ticker", "action", "order_type", "quantity"]),
            ("cancel_order", "Cancel your own order's remaining quantity.", {
                "order_id": {"type": "integer", "minimum": 1},
            }, ["order_id"]),
            ("get_orderbook", "Read five aggregated internal price levels per side.",
             {"ticker": ticker}, ["ticker"]),
            ("get_recent_trades", "Read the last five anonymous internal trades.",
             {"ticker": ticker}, ["ticker"]),
            ("get_own_orders", "Read every order you have placed, newest first, including "
                               "filled and canceled ones.", {
                "ticker": {"type": "string", "description": "Optional: only this market."},
            }, []),
            ("get_account", "Read your cash, reservations, positions, equity and total PnL.", {}, []),
            ("hold_for_now", f"Hold for {self._settings.hold_for_now_minutes} elapsed minutes. "
                             "Finish the entire hold even across the session close.",
             {"memory_note": note}, []),
            ("hold_until_next_day", "Relinquish the rest of this session; return at the next day's opening.",
             {"memory_note": note}, []),
        ]
        return [
            {"type": "function", "function": {
                "name": name, "description": description,
                "parameters": {"type": "object", "properties": properties,
                               "required": required, "additionalProperties": False},
            }}
            for name, description, properties, required in specifications
        ] + [copy.deepcopy(NEWS_TOOL_SCHEMA)]

    def dispatch(
        self, name: str, arguments: str, *,
        on_dispatch: Callable[[str, dict], dict | None] | None = None,
    ) -> ToolOutcome:
        """Run one tool. ``on_dispatch`` sees the parsed call and may refuse it.

        A refusal is returned undispatched and does not execute the tool.
        """
        prepared = self._prepare(name, arguments, on_dispatch)
        if isinstance(prepared, ToolOutcome):
            return prepared
        return ToolOutcome(self._functions[name](**prepared), dispatched=True)

    async def dispatch_async(
        self, name: str, arguments: str, *,
        on_dispatch: Callable[[str, dict], dict | None] | None = None,
    ) -> ToolOutcome:
        """Charge on the event-loop thread before any network await.

        Local exchange tools still execute synchronously on that same thread.
        """
        prepared = self._prepare(name, arguments, on_dispatch)
        if isinstance(prepared, ToolOutcome):
            return prepared
        if name in self._async_functions:
            result = await self._async_functions[name](**prepared)
        else:
            result = self._functions[name](**prepared)
        return ToolOutcome(result, dispatched=True)

    def _prepare(self, name, arguments, on_dispatch) -> dict | ToolOutcome:
        if not isinstance(name, str) or name not in self._functions:
            return ToolOutcome(error("unknown_tool", "Unknown tool name."))
        function = self._functions[name]
        try:
            values = parse_arguments(arguments)
            # The bound key name is harness configuration, never model input.
            if name == "search_news" and set(values) != {"query"}:
                raise ValueError("search_news requires only query")
            # Bind shape only. Business/value errors returned by an invoked
            # exchange tool still consume its time cost and activity count.
            inspect.signature(function).bind(**values)
        except (ValueError, TypeError, RecursionError) as exc:
            return ToolOutcome(error("invalid_arguments", str(exc)))

        if name in {"hold_for_now", "hold_until_next_day"}:
            try:
                outcome = function(**values)
            except ValueError as exc:
                return ToolOutcome(error("invalid_arguments", str(exc)))
            if on_dispatch is not None:
                refusal = on_dispatch(name, values)
                if refusal is not None:
                    return ToolOutcome(refusal)
            return outcome
        # Unexpected implementation/DB failures propagate; never retry a write.
        if on_dispatch is not None:
            refusal = on_dispatch(name, values)
            if refusal is not None:
                return ToolOutcome(refusal)
        return values

    def _note(self, value: Any) -> str | None:
        if value is not None and (
            not isinstance(value, str) or len(value) > self._settings.private_note_max_chars
        ):
            raise ValueError(
                f"memory_note must be a string of at most {self._settings.private_note_max_chars} characters"
            )
        return value

    def time_cost_seconds(self, name: str) -> int:
        if name in {"submit_order", "cancel_order"}:
            return self._settings.order_tool_minutes * 60
        if name == "search_news":
            return self._settings.news_tool_minutes * 60
        if name == "hold_for_now":
            return self._settings.hold_for_now_minutes * 60
        if name in {"get_orderbook", "get_recent_trades", "get_own_orders", "get_account"}:
            return self._settings.read_tool_minutes * 60
        raise ValueError(f"no fixed time cost for {name}")

    def _hold_for_now(self, memory_note: str | None = None) -> ToolOutcome:
        return ToolOutcome(
            {"ok": True}, dispatched=True,
            wait_seconds=self.time_cost_seconds("hold_for_now"),
            memory_note=self._note(memory_note),
        )

    def _hold_until_next_day(self, memory_note: str | None = None) -> ToolOutcome:
        return ToolOutcome(
            {"ok": True}, dispatched=True,
            next_day=True,
            memory_note=self._note(memory_note),
        )
