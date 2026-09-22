"""Consistent run data, filters, tool dispatch times, and complete book replay."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..storage.runs import open_run, resolve_database
from .legacy import LegacyBookHistory


class RunReader:
    """One connection and segment policy for trace, export, and price analysis."""

    def __init__(self, database=None, *, follow=False, segment="all"):
        if segment not in {"all", "current"}:
            raise ValueError("segment must be all or current")
        self.path = resolve_database(database)
        self.segment = segment
        self._reader = open_run(self.path, follow=follow)
        self._conn = self._reader.__enter__()
        try:
            self.boundaries = {"first_transcript_entry_id": 1, "first_trade_id": 1, "started_at": 0}
            if self._conn.execute("SELECT 1 FROM sqlite_master WHERE name='run_segments'").fetchone():
                latest = self._conn.execute("SELECT * FROM run_segments ORDER BY rowid DESC LIMIT 1").fetchone()
                if latest:
                    self.boundaries = dict(latest)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def close(self):
        self._reader.__exit__(None, None, None)

    def __exit__(self, *exc):
        self.close()

    def transcript(self, *, after=0, filters=None):
        filters = filters or Filters()
        first = self.boundaries["first_transcript_entry_id"] if self.segment == "current" else 1
        rows = self._conn.execute("SELECT * FROM transcript_entries WHERE entry_id>? ORDER BY entry_id",
                                  (max(after, first - 1),))
        for row in rows:
            if filters.matches(row["created_at"], row["agent_id"], None):
                yield row

    def prices(self, tickers=None, agent_id=None, *, start=None, end=None, filter_agent=True):
        markets = [dict(r) for r in self._conn.execute("SELECT * FROM markets ORDER BY cohort_index")]
        wanted = set(tickers or [m["ticker"] for m in markets])
        unknown = wanted - {m["ticker"] for m in markets}
        if unknown:
            raise ValueError("tickers not in this run: " + ", ".join(sorted(unknown)))
        agents = {r[0] for r in self._conn.execute("SELECT agent_id FROM accounts")}
        if agent_id and agent_id not in agents:
            raise ValueError(f"agent not in this run: {agent_id}")
        filters = Filters(start=start, end=end, market=wanted)
        settings = json.loads(self._conn.execute("SELECT config_json FROM run_settings").fetchone()[0])
        first = self.boundaries["first_trade_id"] if self.segment == "current" else 1
        rows = self._conn.execute("""
            SELECT t.*, maker.agent_id AS maker_agent_id, taker.agent_id AS taker_agent_id,
                   CASE WHEN maker.action='buy' THEN maker.agent_id ELSE taker.agent_id END AS buyer,
                   CASE WHEN maker.action='sell' THEN maker.agent_id ELSE taker.agent_id END AS seller
            FROM trades t JOIN orders maker ON maker.order_id=t.maker_order_id
                          JOIN orders taker ON taker.order_id=t.taker_order_id
            WHERE t.trade_id>=? ORDER BY t.executed_at, t.trade_id
        """, (first,))
        trades = [dict(row) for row in rows
                  if filters.matches(row["executed_at"], None, None, row["ticker"])
                  and (not filter_agent or not agent_id or agent_id in (row["buyer"], row["seller"]))]
        return [m for m in markets if m["ticker"] in wanted], trades, settings


def utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def parse_time(value: str | int | float | datetime | None) -> float | None:
    """Accept Unix seconds or an ISO 8601 instant with an explicit timezone."""
    if value is None:
        return None
    if isinstance(value, datetime):
        date = value
    else:
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).replace("Z", "+00:00")
            # Python 3.10 accepts 3/6 fractional digits; API timestamps vary.
            text = re.sub(r"\.(\d+)(?=[+-]\d{2}:\d{2}$)",
                          lambda match: "." + match[1][:6].ljust(6, "0"), text)
            date = datetime.fromisoformat(text)
        else:
            if not math.isfinite(number):
                raise ValueError("time must be finite")
            return number
    if date.tzinfo is None:
        raise ValueError("ISO timestamps must include Z or a UTC offset, e.g. -04:00")
    return date.timestamp()


def _choices(value: str | Iterable[str] | None) -> set[str] | None:
    return {value} if isinstance(value, str) else (set(value) if value is not None else None)


@dataclass(frozen=True)
class Filters:
    agent: str | Iterable[str] | None = None
    tool: str | Iterable[str] | None = None
    start: str | int | float | datetime | None = None
    end: str | int | float | datetime | None = None
    market: str | Iterable[str] | None = None

    def __post_init__(self):
        for name in ("agent", "tool", "market"):
            object.__setattr__(self, name, _choices(getattr(self, name)))
        object.__setattr__(self, "start", parse_time(self.start))
        object.__setattr__(self, "end", parse_time(self.end))
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("start must precede end; intervals are [start, end)")

    def matches(self, timestamp, agent, tool, ticker=None):
        return (
            (self.start is None or timestamp >= self.start)
            and (self.end is None or timestamp < self.end)
            and (self.agent is None or agent in self.agent)
            and (self.tool is None or tool in self.tool)
            and (self.market is None or ticker in self.market)
        )


@dataclass
class _Action:
    action_id: int
    timestamp: int
    sequence: int
    agent_id: str
    name: str
    arguments: dict | None
    time_source: str
    result: dict = field(default_factory=dict)

    def object(self):
        return {
            "time": utc(self.timestamp), "agent_id": self.agent_id,
            "tool_query": {"name": self.name, "arguments": self.arguments},
            "action_id": self.action_id, "sequence": self.sequence,
            "time_source": self.time_source,
        }


class RunDataset(LegacyBookHistory, RunReader):
    """Context-managed export with generators; use list(...) for Python lists.

    Only action metadata, compact mutation records and the current book are
    retained. Full historical snapshots are generated one at a time.
    """

    def __init__(self, database=None, *, log=None, segment="all"):
        super().__init__(database, segment=segment)
        self.log_path = Path(log) if log is not None else self.path.with_name("run.log")
        try:
            tables = {row[0] for row in self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            required = {"transcript_entries", "orders", "trades", "markets", "run_settings", "market_settlements"}
            if not required <= tables:
                raise ValueError("unsupported run schema; missing tables: " + ", ".join(sorted(required - tables)))
            self.recorded_books = "book_events" in tables
            self.warnings: set[str] = set()
            self.book_history_complete = True
            self._market_payloads = {}
            self._market_sources = {}
            self._exported_market_sources = {}
            self._missing_market_tickers = []
            if "market_api_responses" in tables:
                for row in self._conn.execute("SELECT * FROM market_api_responses"):
                    self._market_payloads[row["ticker"]] = json.loads(row["payload_json"])
                    self._market_sources[row["ticker"]] = {
                        "source": "run_first_pull", "captured_at": utc(row["captured_at"]),
                        "source_url": row["source_url"],
                    }
            self._actions = self._read_actions()
            self._orders = {row["order_id"]: dict(row) for row in self._conn.execute("SELECT * FROM orders")}
            self._order_actions = {}
            for action in self._actions:
                if action.name in {"submit_order", "submit_quote"}:
                    for key in ("order", "bid", "ask"):
                        order = action.result.get(key)
                        if isinstance(order, dict) and "order_id" in order:
                            self._order_actions[order["order_id"]] = action
            self._legacy_events = None
            if not self.recorded_books:
                self.warnings.add("Legacy book history is reconstructed from orders, trades, tool results and the optional run.log.")
                self._legacy_events = self._reconstruct_events()
        except BaseException:
            self.close()
            raise

    def _read_actions(self) -> list[_Action]:
        pending: dict[str, dict[str, dict]] = {}
        actions = []
        for row in self._conn.execute("SELECT * FROM transcript_entries ORDER BY entry_id"):
            body = json.loads(row["content_json"])
            agent, timestamp, sequence = row["agent_id"], row["created_at"], row["entry_id"]
            if row["entry_type"] == "model_response":
                pending[agent] = {}
                if not isinstance(body, dict) or body.get("finish_reason") not in {"stop", "tool_calls"}:
                    continue
                message = body.get("message")
                if not isinstance(message, dict):
                    continue
                calls = message.get("tool_calls") or []
                if not isinstance(calls, list):
                    continue
                for call in calls:
                    if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                        continue
                    function = call.get("function")
                    if not isinstance(function, dict):
                        continue
                    args = function.get("arguments")
                    try:
                        args = json.loads(args) if isinstance(args, str) else args
                    except (ValueError, TypeError):
                        args = None
                    pending[agent][call.get("id")] = {
                        "name": function.get("name"), "arguments": args,
                        "model_time": timestamp, "model_sequence": sequence, "action": None,
                    }
            elif row["entry_type"] == "timing":
                event = body.get("event")
                if event == "tool_dispatched":
                    action = _Action(sequence, body["fired_at"], sequence, agent,
                                     body["tool"], body["arguments"], "tool_dispatched")
                    actions.append(action)
                    item = pending.get(agent, {}).get(body.get("tool_call_id"))
                    if item is not None:
                        item["action"] = action
                elif event == "action_scheduled" and pending.get(agent):
                    # Before explicit dispatch records, scheduling occurs on
                    # dispatch of the first remaining (not already refused) call.
                    item = next(iter(pending[agent].values()))
                    if item["action"] is None and isinstance(item["arguments"], dict):
                        action = _Action(sequence, timestamp, sequence, agent,
                                         item["name"], item["arguments"], "action_scheduled")
                        item["action"] = action
                        actions.append(action)
            elif row["entry_type"] == "tool_result":
                if body.get("name") is None:
                    # A malformed response can contain plausible calls, but
                    # the harness rejects the entire response before dispatch.
                    pending.pop(agent, None)
                    continue
                item = pending.get(agent, {}).pop(body.get("tool_call_id"), None)
                result = body.get("result") or {}
                if not result.get("tool_call_executed", result.get("executed", False)):
                    continue
                action = item["action"] if item else None
                if action is None:
                    fired_at = item["model_time"] if item else timestamp
                    source = "model_response" if item else "tool_result"
                    name = body.get("name")
                    if name in {"submit_order", "submit_quote"}:
                        orders = [result[k] for k in ("order", "bid", "ask") if isinstance(result.get(k), dict)]
                        if orders:
                            fired_at = min(order["created_at"] for order in orders)
                            source = "order_created"
                    action = _Action(sequence, fired_at,
                                     item["model_sequence"] if item and source == "model_response" else sequence,
                                     agent, name, item["arguments"] if item else None, source)
                    actions.append(action)
                    if source in {"model_response", "tool_result"}:
                        self.warnings.add("Some legacy dispatch times are estimates; inspect each action's time_source.")
                    if item is None:
                        self.warnings.add("Some legacy fired calls have no matching model arguments.")
                # Do not retain potentially large news text or opaque reasoning.
                action.result = {key: result[key] for key in ("ok", "order", "bid", "ask") if key in result}
        return sorted(actions, key=lambda action: (action.timestamp, action.sequence))

    def _action_ticker(self, action: _Action) -> str | None:
        args = action.arguments or {}
        if action.name == "cancel_order":
            order_id = args.get("order_id")
            return self._orders.get(order_id, {}).get("ticker") if type(order_id) is int else None
        ticker = args.get("ticker")
        return ticker if isinstance(ticker, str) else None

    def actions(self, filters: Filters | None = None) -> Iterator[dict[str, Any]]:
        filters = filters or Filters()
        for action in self._actions:
            ticker = self._action_ticker(action)
            if self.includes_action(action) and filters.matches(action.timestamp, action.agent_id, action.name, ticker):
                yield action.object()

    def markets(self, filters: Filters | None = None, *, fetch_missing: bool = False) -> Iterator[dict]:
        """Unmodified first-pull Kalshi market objects, with provenance in metadata.

        Default: the fixed run cohort. Agent/tool/time filters select markets
        explicitly referenced by matching actions or book changes. Missing
        historical objects are never synthesized from normalized DB columns.
        Optional HTTP backfill returns current data, clearly labeled as such.
        """
        filters = filters or Filters()
        tickers = [row[0] for row in self._conn.execute("SELECT ticker FROM markets ORDER BY cohort_index")]
        if self.segment == "current" or any(value is not None for value in (filters.agent, filters.tool, filters.start, filters.end)):
            referenced = {
                self._action_ticker(action) for action in self._actions
                if self.includes_action(action) and filters.matches(action.timestamp, action.agent_id, action.name, self._action_ticker(action))
            }
            for event in self._events():
                if not self.includes_event(event):
                    continue
                for change in event["changes"]:
                    if filters.matches(event["timestamp"], event["agent_id"], event["tool"], change["ticker"]):
                        referenced.add(change["ticker"])
            tickers = [ticker for ticker in tickers if ticker in referenced]
        if filters.market is not None:
            tickers = [ticker for ticker in tickers if ticker in filters.market]
        missing = [ticker for ticker in tickers if ticker not in self._market_payloads]
        if missing and fetch_missing:
            # Network access is needed only for explicit legacy backfill.
            from ..integrations.kalshi import fetch_market_payloads
            settings = json.loads(self._conn.execute("SELECT config_json FROM run_settings").fetchone()[0])
            base_url = settings["kalshi_base_url"]
            payloads = fetch_market_payloads(missing, base_url=base_url)
            captured_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            for payload in payloads:
                ticker = payload["ticker"]
                self._market_payloads[ticker] = payload
                self._market_sources[ticker] = {
                    "source": "fetched_at_export", "captured_at": captured_at,
                    "source_url": base_url.rstrip("/") + "/markets",
                }
        self._missing_market_tickers = [ticker for ticker in tickers if ticker not in self._market_payloads]
        self._exported_market_sources = {ticker: self._market_sources[ticker] for ticker in tickers
                                         if ticker in self._market_payloads}
        for ticker in tickers:
            if ticker in self._market_payloads:
                yield self._market_payloads[ticker]

    @staticmethod
    def _resting(order: dict, remaining: int) -> dict:
        return {**{key: order[key] for key in ("order_id", "agent_id", "ticker", "action", "price_cents")},
                "remaining_quantity": remaining}

    def _events(self):
        if self.recorded_books:
            for row in self._conn.execute("SELECT * FROM book_events ORDER BY fired_at, event_id"):
                yield {"event_id": row["event_id"], "timestamp": row["fired_at"],
                       "action_id": row["action_entry_id"], "agent_id": row["agent_id"], "tool": row["tool"],
                       "time_source": "tool_dispatched" if row["action_entry_id"] else "exchange_transaction",
                       "changes": json.loads(row["changes_json"])}
            return
        resting = {}
        for event_id, event in enumerate(self._legacy_events, 1):
            changes = []
            if event["kind"] == "submit":
                order = self._orders[event["order_id"]]
                for trade in event["trades"]:
                    maker = resting.get(trade["maker_order_id"])
                    if maker is None or maker["remaining_quantity"] < trade["quantity"]:
                        self.book_history_complete = False
                        self.warnings.add("Legacy same-second ordering or missing cancellations prevent exact replay of some fills.")
                        continue
                    remaining = maker["remaining_quantity"] - trade["quantity"]
                    updated = {**maker, "remaining_quantity": remaining} if remaining else None
                    changes.append({"order_id": maker["order_id"], "ticker": maker["ticker"], "order": updated})
                    if updated:
                        resting[maker["order_id"]] = updated
                    else:
                        resting.pop(maker["order_id"], None)
                remaining = order["original_quantity"] - sum(trade["quantity"] for trade in event["trades"])
                if order["order_type"] == "limit" and remaining > 0:
                    updated = self._resting(order, remaining)
                    resting[order["order_id"]] = updated
                    changes.append({"order_id": order["order_id"], "ticker": order["ticker"], "order": updated})
            else:
                order_ids = event.get("order_ids", [event.get("order_id")])
                for order_id in order_ids:
                    removed = resting.pop(order_id, None)
                    if removed is not None:
                        changes.append({"order_id": order_id, "ticker": removed["ticker"], "order": None})
            if changes:
                yield {**{key: event[key] for key in ("timestamp", "action_id", "agent_id", "tool", "time_source")},
                       "event_id": event_id, "changes": changes}

    def orderbooks(self, filters: Filters | None = None) -> Iterator[dict[str, Any]]:
        """Full per-market books AFTER each committed change, all price levels.

        Filters select the event's initiating agent/tool. All events are replayed
        first, including those before --start and those from other agents.
        """
        filters = filters or Filters()
        books = defaultdict(dict)
        for event in self._events():
            changed = set()
            for change in event["changes"]:
                book = books[change["ticker"]]
                order_id, order = change["order_id"], change["order"]
                before = book.get(order_id)
                if order is None:
                    book.pop(order_id, None)
                else:
                    book[order_id] = order
                if before != order:
                    changed.add(change["ticker"])
            for ticker in sorted(changed):
                if not self.includes_event(event) or not filters.matches(event["timestamp"], event["agent_id"], event["tool"], ticker):
                    continue
                book = books[ticker].values()
                def side(action):
                    orders = [order for order in book if order["action"] == action]
                    orders.sort(key=lambda order: ((-1 if action == "buy" else 1) * order["price_cents"], order["order_id"]))
                    return [{"order_id": order["order_id"], "agent_id": order["agent_id"],
                             "price_cents": order["price_cents"], "quantity": order["remaining_quantity"]}
                            for order in orders]
                yield {"time": utc(event["timestamp"]), "ticker": ticker,
                       "event_id": event["event_id"], "action_id": event["action_id"],
                       "agent_id": event["agent_id"], "tool": event["tool"],
                       "time_source": event["time_source"], "bids": side("buy"), "asks": side("sell")}
        actual = {order_id: order for book in books.values() for order_id, order in book.items()}
        expected = {order_id: self._resting(order, order["remaining_quantity"])
                    for order_id, order in self._orders.items()
                    if order["status"] == "open" and order["order_type"] == "limit"}
        if actual != expected:
            self.book_history_complete = False
            self.warnings.add("Replayed final book differs from the database's resting orders; book history is incomplete.")

    def includes_action(self, action):
        return self.segment == "all" or action.action_id >= self.boundaries["first_transcript_entry_id"]

    def includes_event(self, event):
        if self.segment == "all":
            return True
        if event["action_id"] is not None:
            return event["action_id"] >= self.boundaries["first_transcript_entry_id"]
        return event["timestamp"] >= self.boundaries["started_at"]

    def metadata(self):
        warnings = set(self.warnings)
        if self._missing_market_tickers:
            warnings.add("Raw Kalshi payloads unavailable for: " + ", ".join(self._missing_market_tickers)
                         + ". Use --fetch-missing-markets to request current data; original responses cannot be recovered.")
        if any(source["source"] == "fetched_at_export" for source in self._exported_market_sources.values()):
            warnings.add("Some Kalshi payloads were fetched at export time; they are not historical run-time observations.")
        return {"schema_version": 1, "source_database": str(self.path), "segment": self.segment,
                "time_format": "ISO 8601 UTC", "time_resolution": "one second",
                "time_interval": "[start, end)", "book_history_complete": self.book_history_complete,
                "book_history_source": "recorded_deltas" if self.recorded_books else "legacy_reconstruction",
                "book_filter_semantics": "initiating agent/tool; full market state is preserved",
                "market_payloads": {"sources": self._exported_market_sources,
                                    "missing_tickers": self._missing_market_tickers},
                "warnings": sorted(warnings)}
