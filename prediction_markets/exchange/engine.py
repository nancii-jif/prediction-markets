"""Single-process continuous YES order book backed by SQLite.

All public methods are synchronous. The runner owns one ``Exchange`` instance
and calls it on the event-loop thread; an internal lock serializes admission,
matching, cancellation, and reads without placing an ``await`` inside a
transaction.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from ..storage import database
from . import bookkeeping

log = logging.getLogger(__name__)

MAX_SQLITE_INTEGER = 2**63 - 1
MARKET_BUY_PROTECTION_CENTS = 99
MARKET_SELL_PROTECTION_CENTS = 1


class _Rejected(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


class Exchange:
    """The one local exchange for a run."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        now: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(conn, sqlite3.Connection):
            raise TypeError("conn must be a sqlite3.Connection")
        conn.row_factory = sqlite3.Row
        saved = conn.execute(
            "SELECT config_json FROM run_settings WHERE singleton_id = 1"
        ).fetchone()
        if saved is None:
            raise RuntimeError("database is missing its frozen run settings")
        settings = json.loads(saved["config_json"])

        self._conn = conn
        self._now = now if now is not None else lambda: int(time.time())
        self._initial_cash_cents = int(settings["initial_cash_cents"])
        self._stale_after_seconds = int(settings["market_stale_after_seconds"])
        self._lock = threading.RLock()
        self._action_context = ContextVar("exchange_action", default=None)
        self._book_before: dict[int, dict | None] = {}

    def store_market_cohort(self, rows, **kwargs) -> int:
        """Persist the initial cohort using this exchange's connection."""
        return database.store_market_cohort(rows, conn=self._conn, **kwargs)

    def invalidate_market_observations(self) -> None:
        """A continuation must freshly observe every unsettled market."""
        with self._conn:
            self._conn.execute("UPDATE markets SET last_successful_poll=1 "
                               "WHERE ticker NOT IN (SELECT ticker FROM market_settlements)")

    @contextmanager
    def action_context(self, context: dict):
        """Task-local attribution, safe even when a news request awaits I/O."""
        token = self._action_context.set(context)
        try:
            yield
        finally:
            self._action_context.reset(token)

    @contextmanager
    def _transaction(self, tool="exchange", agent_id=None):
        if self._conn.in_transaction:
            raise RuntimeError("exchange connection already has a transaction")
        self._conn.execute("BEGIN IMMEDIATE")
        self._book_before = {}
        try:
            now = self._timestamp()
            yield
            changes = []
            for order_id, before in self._book_before.items():
                row = self._order_row(order_id)
                after = self._resting_order(row)
                if before != after:
                    changes.append({"order_id": order_id, "ticker": row["ticker"], "order": after})
            if changes:
                context = self._action_context.get() or {}
                self._conn.execute(
                    "INSERT INTO book_events (fired_at, action_entry_id, agent_id, tool, changes_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (context.get("fired_at", now), context.get("action_entry_id"),
                     context.get("agent_id", agent_id), context.get("tool", tool),
                     json.dumps(changes, separators=(",", ":"))),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            self._book_before = {}

    @staticmethod
    def _resting_order(row) -> dict | None:
        if row["order_type"] != "limit" or row["status"] != "open":
            return None
        return {key: row[key] for key in (
            "order_id", "agent_id", "ticker", "action", "price_cents", "remaining_quantity",
        )}

    def _remember_order(self, row) -> None:
        self._book_before.setdefault(row["order_id"], self._resting_order(row))

    def _remember_open_orders(self, where: str = "1", params: tuple = ()) -> None:
        for row in self._conn.execute("SELECT * FROM orders WHERE status = 'open' AND " + where, params):
            self._remember_order(row)

    def _timestamp(self) -> int:
        value = self._now()
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
        ):
            raise RuntimeError("exchange clock must return a positive integer")
        return value

    @staticmethod
    def _require_text(name: str, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise _Rejected("invalid_input", f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _require_positive_integer(name: str, value: Any) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value > MAX_SQLITE_INTEGER
        ):
            raise _Rejected(
                "invalid_input",
                f"{name} must be a positive integer",
            )
        return value

    def _account_row(self, agent_id: Any) -> sqlite3.Row:
        agent_id = self._require_text("agent_id", agent_id)
        row = self._conn.execute(
            "SELECT agent_id, balance_cents FROM accounts WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        if row is None:
            raise _Rejected("unknown_agent", f"unknown agent: {agent_id}")
        return row

    def _market_row(self, ticker: Any) -> sqlite3.Row:
        ticker = self._require_text("ticker", ticker)
        row = self._conn.execute(
            "SELECT m.*, s.ticker AS settled_ticker "
            "FROM markets m LEFT JOIN market_settlements s "
            "ON s.ticker = m.ticker WHERE m.ticker = ?",
            (ticker,),
        ).fetchone()
        if row is None:
            raise _Rejected("unknown_market", f"unknown market: {ticker}")
        return row

    def _require_tradable(self, ticker: str, now: int) -> sqlite3.Row:
        row = self._market_row(ticker)
        reason: str | None = None
        if row["settled_ticker"] is not None:
            reason = "market is settled"
        elif row["status"] != "active":
            reason = f"market status is {row['status']}"
        elif row["close_time"] <= now:
            reason = "market trading cutoff has passed"
        elif now - row["last_successful_poll"] > self._stale_after_seconds:
            reason = "market data is stale"
        if reason is not None:
            raise _Rejected(
                "market_not_tradable",
                f"{ticker} is not accepting new orders: {reason}",
            )
        return row

    @staticmethod
    def _effective_price(
        action: str,
        order_type: str,
        price_cents: int | None,
    ) -> int:
        if order_type == "limit":
            if price_cents is None:
                raise RuntimeError("open limit order is missing its price")
            return price_cents
        return (
            MARKET_BUY_PROTECTION_CENTS
            if action == "buy"
            else MARKET_SELL_PROTECTION_CENTS
        )

    def _reservation_total(
        self,
        agent_id: str,
        *,
        extra: Mapping[str, Any] | None = None,
    ) -> int:
        positions = {
            row["ticker"]: row["signed_qty"]
            for row in self._conn.execute(
                "SELECT ticker, signed_qty FROM positions WHERE agent_id = ?",
                (agent_id,),
            )
        }
        exposures: dict[str, list[int]] = {}
        for row in self._conn.execute(
            "SELECT ticker, action, order_type, price_cents, "
            "remaining_quantity FROM orders "
            "WHERE agent_id = ? AND status = 'open'",
            (agent_id,),
        ):
            effective_price = self._effective_price(
                row["action"], row["order_type"], row["price_cents"]
            )
            buy, sell = exposures.setdefault(row["ticker"], [0, 0])
            if row["action"] == "buy":
                buy += effective_price * row["remaining_quantity"]
            else:
                sell += (
                    bookkeeping.CONTRACT_PAYOUT_CENTS - effective_price
                ) * row["remaining_quantity"]
            exposures[row["ticker"]] = [buy, sell]

        if extra is not None:
            effective_price = self._effective_price(
                extra["action"], extra["order_type"], extra["price_cents"]
            )
            buy, sell = exposures.setdefault(extra["ticker"], [0, 0])
            if extra["action"] == "buy":
                buy += effective_price * extra["quantity"]
            else:
                sell += (
                    bookkeeping.CONTRACT_PAYOUT_CENTS - effective_price
                ) * extra["quantity"]
            exposures[extra["ticker"]] = [buy, sell]

        return sum(
            bookkeeping.reservation_cents(
                positions.get(ticker, 0),
                exposure[0],
                exposure[1],
            )
            for ticker, exposure in exposures.items()
        )

    def _validate_submit(
        self,
        agent_id: Any,
        ticker: Any,
        action: Any,
        order_type: Any,
        quantity: Any,
        price_cents: Any,
    ) -> tuple[str, str, str, str, int, int | None]:
        validated_agent = self._require_text("agent_id", agent_id)
        validated_ticker = self._require_text("ticker", ticker)
        if not isinstance(action, str) or action not in {"buy", "sell"}:
            raise _Rejected("invalid_input", "action must be 'buy' or 'sell'")
        if (
            not isinstance(order_type, str)
            or order_type not in {"limit", "market"}
        ):
            raise _Rejected(
                "invalid_input",
                "order_type must be 'limit' or 'market'",
            )
        validated_quantity = self._require_positive_integer("quantity", quantity)
        if order_type == "limit":
            if (
                isinstance(price_cents, bool)
                or not isinstance(price_cents, int)
                or not 1 <= price_cents <= 99
            ):
                raise _Rejected(
                    "invalid_input",
                    "limit price_cents must be an integer from 1 through 99",
                )
            validated_price: int | None = price_cents
        else:
            if price_cents is not None:
                raise _Rejected(
                    "invalid_input",
                    "market orders must omit price_cents",
                )
            validated_price = None
        return (
            validated_agent,
            validated_ticker,
            action,
            order_type,
            validated_quantity,
            validated_price,
        )

    def submit_order(
        self,
        agent_id: str,
        ticker: str,
        action: str,
        order_type: str,
        quantity: int,
        price_cents: int | None = None,
    ) -> dict[str, Any]:
        """Fund, accept, and immediately match one incoming order."""
        try:
            (
                agent_id,
                ticker,
                action,
                order_type,
                quantity,
                price_cents,
            ) = self._validate_submit(
                agent_id,
                ticker,
                action,
                order_type,
                quantity,
                price_cents,
            )
            with self._lock:
                now = self._timestamp()
                with self._transaction("submit_order", agent_id):
                    account = self._account_row(agent_id)
                    self._require_tradable(ticker, now)
                    proposed = {
                        "ticker": ticker,
                        "action": action,
                        "order_type": order_type,
                        "quantity": quantity,
                        "price_cents": price_cents,
                    }
                    reserved = self._reservation_total(agent_id, extra=proposed)
                    if account["balance_cents"] - reserved < 0:
                        raise _Rejected(
                            "insufficient_funds",
                            "order would make available cash negative",
                        )

                    cursor = self._conn.execute(
                        "INSERT INTO orders ("
                        "agent_id, ticker, action, order_type, price_cents, "
                        "original_quantity, filled_quantity, "
                        "remaining_quantity, status, cancel_reason, created_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, 0, ?, 'open', NULL, ?)",
                        (
                            agent_id,
                            ticker,
                            action,
                            order_type,
                            price_cents,
                            quantity,
                            quantity,
                            now,
                        ),
                    )
                    order_id = int(cursor.lastrowid)
                    self._book_before[order_id] = None
                    trades, affected_agents = self._match(order_id, now)

                    incoming = self._order_row(order_id)
                    if incoming["status"] == "open" and order_type == "market":
                        self._cancel_remainder(
                            order_id,
                            reason="market_remainder",
                        )

                    for affected_agent in affected_agents | {agent_id}:
                        self._assert_funded(affected_agent)
                    order = self._order_dict(self._order_row(order_id))
            log.info("order accepted: %s", order)
            for trade in trades:
                log.info("fill: %s", trade)
            return {"ok": True, "order": order, "trades": trades}
        except _Rejected as exc:
            return _error(exc.code, exc.message)

    def _best_maker(self, incoming: sqlite3.Row) -> sqlite3.Row | None:
        params: list[Any] = [incoming["ticker"]]
        if incoming["action"] == "buy":
            where = "action = 'sell'"
            price_order = "price_cents ASC"
            if incoming["order_type"] == "limit":
                where += " AND price_cents <= ?"
                params.append(incoming["price_cents"])
        else:
            where = "action = 'buy'"
            price_order = "price_cents DESC"
            if incoming["order_type"] == "limit":
                where += " AND price_cents >= ?"
                params.append(incoming["price_cents"])
        return self._conn.execute(
            "SELECT * FROM orders WHERE ticker = ? AND status = 'open' "
            "AND order_type = 'limit' AND "
            f"{where} ORDER BY {price_order}, order_id ASC LIMIT 1",
            params,
        ).fetchone()

    def _match(
        self,
        incoming_order_id: int,
        executed_at: int,
    ) -> tuple[list[dict[str, int]], set[str]]:
        trades: list[dict[str, int]] = []
        affected_agents: set[str] = set()
        while True:
            incoming = self._order_row(incoming_order_id)
            if incoming["status"] != "open":
                break
            maker = self._best_maker(incoming)
            if maker is None:
                break

            quantity = min(
                incoming["remaining_quantity"], maker["remaining_quantity"]
            )
            price_cents = maker["price_cents"]
            if incoming["action"] == "buy":
                buyer, seller = incoming["agent_id"], maker["agent_id"]
            else:
                buyer, seller = maker["agent_id"], incoming["agent_id"]

            # A self-match nets to the same cash and position, but the legs are
            # applied in sequence, so the buy must settle first: reservations
            # always cover buy exposure, which keeps interim cash nonnegative.
            self._apply_fill(buyer, incoming["ticker"], quantity, price_cents)
            self._apply_fill(seller, incoming["ticker"], -quantity, price_cents)
            self._fill_order(maker["order_id"], quantity)
            self._fill_order(incoming_order_id, quantity)
            self._conn.execute(
                "INSERT INTO trades ("
                "ticker, maker_order_id, taker_order_id, price_cents, quantity, "
                "executed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    incoming["ticker"],
                    maker["order_id"],
                    incoming_order_id,
                    price_cents,
                    quantity,
                    executed_at,
                ),
            )
            affected_agents.update((buyer, seller))
            trades.append(
                {
                    "price_cents": price_cents,
                    "quantity": quantity,
                    "executed_at": executed_at,
                }
            )
        return trades, affected_agents

    def _apply_fill(
        self,
        agent_id: str,
        ticker: str,
        signed_quantity: int,
        price_cents: int,
    ) -> None:
        account = self._conn.execute(
            "SELECT balance_cents FROM accounts WHERE agent_id = ?",
            (agent_id,),
        ).fetchone()
        position = self._conn.execute(
            "SELECT signed_qty FROM positions "
            "WHERE agent_id = ? AND ticker = ?",
            (agent_id, ticker),
        ).fetchone()
        if account is None:
            raise RuntimeError(f"trade references unknown agent {agent_id}")
        held = position["signed_qty"] if position is not None else 0
        cash, quantity = bookkeeping.after_fill(
            account["balance_cents"],
            held,
            signed_quantity,
            price_cents,
        )
        if cash < 0:
            raise RuntimeError("funding invariant allowed negative cash")
        self._conn.execute(
            "UPDATE accounts SET balance_cents = ? WHERE agent_id = ?",
            (cash, agent_id),
        )
        if quantity == 0:
            self._conn.execute(
                "DELETE FROM positions WHERE agent_id = ? AND ticker = ?",
                (agent_id, ticker),
            )
        else:
            self._conn.execute(
                "INSERT INTO positions (agent_id, ticker, signed_qty) "
                "VALUES (?, ?, ?) ON CONFLICT(agent_id, ticker) DO UPDATE SET "
                "signed_qty = excluded.signed_qty",
                (agent_id, ticker, quantity),
            )

    def _fill_order(self, order_id: int, quantity: int) -> None:
        row = self._order_row(order_id)
        self._remember_order(row)
        remaining = row["remaining_quantity"] - quantity
        filled = row["filled_quantity"] + quantity
        status = "filled" if remaining == 0 else "open"
        self._conn.execute(
            "UPDATE orders SET filled_quantity = ?, remaining_quantity = ?, "
            "status = ? WHERE order_id = ?",
            (filled, remaining, status, order_id),
        )

    def _cancel_remainder(self, order_id: int, *, reason: str) -> None:
        self._remember_order(self._order_row(order_id))
        self._conn.execute(
            "UPDATE orders SET remaining_quantity = 0, status = 'canceled', "
            "cancel_reason = ? WHERE order_id = ? AND status = 'open'",
            (reason, order_id),
        )

    def _assert_funded(self, agent_id: str) -> None:
        row = self._account_row(agent_id)
        if row["balance_cents"] - self._reservation_total(agent_id) < 0:
            raise RuntimeError("post-trade funding invariant failed")

    def _order_row(self, order_id: int) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing accepted order {order_id}")
        return row

    @staticmethod
    def _order_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "order_id": row["order_id"],
            "agent_id": row["agent_id"],
            "ticker": row["ticker"],
            "action": row["action"],
            "order_type": row["order_type"],
            "price_cents": row["price_cents"],
            "original_quantity": row["original_quantity"],
            "filled_quantity": row["filled_quantity"],
            "remaining_quantity": row["remaining_quantity"],
            "status": row["status"],
            "cancel_reason": row["cancel_reason"],
            "created_at": row["created_at"],
        }

    def cancel_order(self, agent_id: str, order_id: int) -> dict[str, Any]:
        """Cancel only the caller's currently open remainder."""
        try:
            agent_id = self._require_text("agent_id", agent_id)
            order_id = self._require_positive_integer("order_id", order_id)
            with self._lock:
                with self._transaction("cancel_order", agent_id):
                    self._account_row(agent_id)
                    row = self._conn.execute(
                        "SELECT * FROM orders WHERE order_id = ?",
                        (order_id,),
                    ).fetchone()
                    if row is None:
                        raise _Rejected("not_found", f"unknown order: {order_id}")
                    if row["agent_id"] != agent_id:
                        raise _Rejected(
                            "not_owner",
                            "an agent may cancel only its own order",
                        )
                    if row["status"] != "open":
                        raise _Rejected(
                            "order_not_open",
                            "order has no open quantity to cancel",
                        )
                    self._cancel_remainder(order_id, reason="user")
                    order = self._order_dict(self._order_row(order_id))
            log.info("order canceled: %s", order)
            return {"ok": True, "order": order}
        except _Rejected as exc:
            return _error(exc.code, exc.message)

    def submit_quote(
        self,
        agent_id: str,
        ticker: str,
        bid_price_cents: int,
        ask_price_cents: int,
        quantity: int,
    ) -> dict[str, Any]:
        """Post both sides of one market as ordinary limit orders.

        A quote is a real commitment: each leg rests, fills and settles like
        any other order. The two legs are posted together so an agent never
        ends up showing one side it did not intend, and the bid is placed
        first because reservations always cover buy exposure.
        """
        try:
            if (
                isinstance(bid_price_cents, bool)
                or isinstance(ask_price_cents, bool)
                or not isinstance(bid_price_cents, int)
                or not isinstance(ask_price_cents, int)
                or bid_price_cents >= ask_price_cents
            ):
                raise _Rejected(
                    "invalid_input",
                    "bid_price_cents must be an integer strictly below ask_price_cents",
                )
        except _Rejected as exc:
            return _error(exc.code, exc.message)

        bid = self.submit_order(agent_id, ticker, "buy", "limit", quantity, bid_price_cents)
        if not bid["ok"]:
            return bid
        ask = self.submit_order(agent_id, ticker, "sell", "limit", quantity, ask_price_cents)
        if not ask["ok"]:
            # Never leave a one-sided quote resting. Anything the bid already
            # traded stands: a filled order cannot be undone, so a rejected
            # quote can still leave the agent holding a partial position.
            with self._lock, self._transaction("quote_rollback", agent_id):
                self._cancel_remainder(bid["order"]["order_id"], reason="quote_rollback")
            return ask
        return {
            "ok": True,
            "bid": bid["order"],
            "ask": ask["order"],
            "spread_cents": ask_price_cents - bid_price_cents,
            "trades": bid["trades"] + ask["trades"],
        }

    def has_market(self, ticker: Any) -> bool:
        """Whether a ticker is in the fixed cohort, for budget attribution.

        A model can name a market that does not exist, and that call must not
        be charged to, or recorded against, a ticker with no row.
        """
        if not isinstance(ticker, str):
            return False
        with self._lock:
            return self._conn.execute(
                "SELECT 1 FROM markets WHERE ticker = ?", (ticker,)
            ).fetchone() is not None

    def order_ticker(self, agent_id: str, order_id: Any) -> str | None:
        """The market of one of this agent's orders, for budget attribution.

        Returns None when the id is not this agent's order, so the caller can
        charge a pool rather than a market it cannot identify.
        """
        if isinstance(order_id, bool) or not isinstance(order_id, int):
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT ticker FROM orders WHERE order_id = ? AND agent_id = ?",
                (order_id, agent_id),
            ).fetchone()
        return row["ticker"] if row is not None else None

    def _cancel_closed_orders(self, now: int) -> int:
        self._remember_open_orders(
            "ticker IN (SELECT ticker FROM markets WHERE close_time <= ? "
            "OR status IN ('closed', 'determined', 'disputed', 'amended', 'finalized'))", (now,),
        )
        return self._conn.execute(
            "UPDATE orders SET remaining_quantity = 0, status = 'canceled', "
            "cancel_reason = 'market_closed' WHERE status = 'open' AND "
            "ticker IN (SELECT ticker FROM markets WHERE close_time <= ? "
            "OR status IN ('closed', 'determined', 'disputed', 'amended', "
            "'finalized'))",
            (now,),
        ).rowcount

    def close_due_markets(self) -> int:
        """Release resting orders at the cutoff, without redeeming positions."""
        with self._lock, self._transaction("market_closed"):
            return self._cancel_closed_orders(self._timestamp())

    def cancel_all_orders(self) -> int:
        """Runner-only shutdown: release reservations, never redeem positions."""
        with self._lock, self._transaction("run_shutdown"):
            self._remember_open_orders()
            return self._conn.execute(
                "UPDATE orders SET remaining_quantity = 0, status = 'canceled', "
                "cancel_reason = 'run_shutdown' WHERE status = 'open'"
            ).rowcount

    def next_close_time(self) -> int | None:
        """Next known cutoff, so the polling task can cancel orders on time."""
        with self._lock:
            return self._conn.execute(
                "SELECT MIN(close_time) FROM markets WHERE close_time > ? "
                "AND status IN ('active', 'inactive', 'initialized') "
                "AND ticker NOT IN (SELECT ticker FROM market_settlements)",
                (self._timestamp(),),
            ).fetchone()[0]

    def unsettled_tickers(self) -> list[str]:
        with self._lock:
            return [
                row[0] for row in self._conn.execute(
                    "SELECT ticker FROM markets WHERE ticker NOT IN "
                    "(SELECT ticker FROM market_settlements) ORDER BY cohort_index"
                )
            ]

    def all_markets_settled(self) -> bool:
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
            return count > 0 and not self.unsettled_tickers()

    def apply_market_observation(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a fetched definition and any final payout in one transaction.

        This is an internal lifecycle operation, never an agent tool. Only
        finalized Kalshi observations with an exact supported payout redeem
        holdings. The unique settlement marker makes repeats harmless.
        """
        database._validate_market(row, 1, require_expiration=False)
        with self._lock, self._transaction("market_observation"):
            market = self._market_row(row["ticker"])
            if market["event_ticker"] != row["event_ticker"]:
                raise ValueError("a market observation cannot change cohort membership")
            if market["settled_ticker"] is not None:
                return {"ok": True, "settled": True}
            if row["last_successful_poll"] < market["last_successful_poll"]:
                raise ValueError("a market observation cannot move freshness backwards")
            self._conn.execute(
                "UPDATE markets SET title = ?, rules = ?, status = ?, "
                "close_time = ?, expected_expiration_time = ?, "
                "latest_expiration_time = ?, last_successful_poll = ?, "
                "payout_cents = ? WHERE ticker = ?",
                (
                    row["title"], row["rules"], row["status"], row["close_time"],
                    row["expected_expiration_time"], row["latest_expiration_time"],
                    row["last_successful_poll"], row["payout_cents"], row["ticker"],
                ),
            )
            now = self._timestamp()
            settled = row["status"] == "finalized" and row["payout_cents"] is not None
            if settled:
                self._remember_open_orders("ticker = ?", (row["ticker"],))
                self._conn.execute(
                    "UPDATE orders SET remaining_quantity = 0, status = 'canceled', "
                    "cancel_reason = 'market_settled' WHERE ticker = ? "
                    "AND status = 'open'",
                    (row["ticker"],),
                )
                positions = self._conn.execute(
                    "SELECT agent_id, signed_qty FROM positions WHERE ticker = ?",
                    (row["ticker"],),
                ).fetchall()
                for position in positions:
                    credit = bookkeeping.settlement_credit_cents(
                        position["signed_qty"], row["payout_cents"],
                    )
                    self._conn.execute(
                        "UPDATE accounts SET balance_cents = balance_cents + ? "
                        "WHERE agent_id = ?", (credit, position["agent_id"]),
                    )
                self._conn.execute(
                    "DELETE FROM positions WHERE ticker = ?", (row["ticker"],),
                )
                self._conn.execute(
                    "INSERT INTO market_settlements (ticker, payout_cents, settled_at) "
                    "VALUES (?, ?, ?)", (row["ticker"], row["payout_cents"], now),
                )
            self._cancel_closed_orders(now)
        if settled:
            log.info("settled %s at %d cents", row["ticker"], row["payout_cents"])
        elif row["status"] == "finalized":
            log.error("finalized market %s has no supported final payout; pending", row["ticker"])
        return {"ok": True, "settled": settled}

    def assigned_markets(self, agent_id: str) -> list[dict[str, Any]]:
        """The same current cohort for every agent, with only public definitions."""
        def utc(value: int | None) -> str | None:
            if value is None:
                return None
            return datetime.fromtimestamp(value, timezone.utc).isoformat().replace("+00:00", "Z")

        with self._lock:
            self._account_row(agent_id)
            now = self._timestamp()
            rows = self._conn.execute(
                "SELECT m.*, s.ticker AS settled_ticker FROM markets m "
                "LEFT JOIN market_settlements s ON m.ticker = s.ticker "
                "ORDER BY m.cohort_index"
            ).fetchall()
            return [
                {
                    "ticker": row["ticker"], "title": row["title"],
                    "rules": row["rules"], "status": row["status"],
                    "close_time": utc(row["close_time"]),
                    "expected_expiration_time": utc(row["expected_expiration_time"]),
                    "latest_expiration_time": utc(row["latest_expiration_time"]),
                    "settled": row["settled_ticker"] is not None,
                    "payout_cents": row["payout_cents"],
                    "tradable": (
                        row["settled_ticker"] is None and row["status"] == "active"
                        and row["close_time"] > now
                        and now - row["last_successful_poll"] <= self._stale_after_seconds
                    ),
                }
                for row in rows
            ]

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        try:
            ticker = self._require_text("ticker", ticker)
            with self._lock:
                self._market_row(ticker)
                bids = self._book_side(ticker, "buy")
                asks = self._book_side(ticker, "sell")
            return {"ok": True, "ticker": ticker, "bids": bids, "asks": asks}
        except _Rejected as exc:
            return _error(exc.code, exc.message)

    def _book_side(self, ticker: str, action: str) -> list[dict[str, int]]:
        order = "DESC" if action == "buy" else "ASC"
        rows = self._conn.execute(
            "SELECT price_cents, SUM(remaining_quantity) AS quantity "
            "FROM orders WHERE ticker = ? AND action = ? "
            "AND status = 'open' AND order_type = 'limit' "
            f"GROUP BY price_cents ORDER BY price_cents {order} LIMIT 5",
            (ticker, action),
        ).fetchall()
        return [
            {"price_cents": row["price_cents"], "quantity": row["quantity"]}
            for row in rows
        ]

    def get_recent_trades(self, ticker: str) -> dict[str, Any]:
        try:
            ticker = self._require_text("ticker", ticker)
            with self._lock:
                self._market_row(ticker)
                rows = self._conn.execute(
                    "SELECT price_cents, quantity, executed_at FROM trades "
                    "WHERE ticker = ? ORDER BY trade_id DESC LIMIT 5",
                    (ticker,),
                ).fetchall()
            return {
                "ok": True,
                "ticker": ticker,
                "trades": [
                    {
                        "ticker": ticker,
                        "price_cents": row["price_cents"],
                        "quantity": row["quantity"],
                        "executed_at": row["executed_at"],
                    }
                    for row in rows
                ],
            }
        except _Rejected as exc:
            return _error(exc.code, exc.message)

    def get_own_orders(
        self,
        agent_id: str,
        ticker: str | None = None,
    ) -> dict[str, Any]:
        """Every order this agent has placed, open and historical, newest first."""
        try:
            agent_id = self._require_text("agent_id", agent_id)
            if ticker is not None:
                ticker = self._require_text("ticker", ticker)
            with self._lock:
                self._account_row(agent_id)
                if ticker is not None:
                    self._market_row(ticker)
                clauses = ["agent_id = ?"]
                params: list[Any] = [agent_id]
                if ticker is not None:
                    clauses.append("ticker = ?")
                    params.append(ticker)
                rows = self._conn.execute(
                    "SELECT * FROM orders WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY order_id DESC",
                    params,
                ).fetchall()
            return {
                "ok": True,
                "orders": [self._order_dict(row) for row in rows],
            }
        except _Rejected as exc:
            return _error(exc.code, exc.message)

    def get_account(self, agent_id: str) -> dict[str, Any]:
        try:
            agent_id = self._require_text("agent_id", agent_id)
            with self._lock:
                account = self._account_row(agent_id)
                reserved = self._reservation_total(agent_id)
                available = account["balance_cents"] - reserved
                if available < 0:
                    raise RuntimeError("stored orders are not fully funded")

                positions: list[dict[str, int]] = []
                position_value = 0
                for row in self._conn.execute(
                    "SELECT ticker, signed_qty FROM positions "
                    "WHERE agent_id = ? AND signed_qty != 0 ORDER BY ticker",
                    (agent_id,),
                ):
                    mark = self._conn.execute(
                        "SELECT price_cents FROM trades WHERE ticker = ? "
                        "ORDER BY trade_id DESC LIMIT 1",
                        (row["ticker"],),
                    ).fetchone()
                    if mark is None:
                        raise RuntimeError(
                            f"open position in {row['ticker']} has no trade mark"
                        )
                    value = bookkeeping.position_value_cents(
                        row["signed_qty"], mark["price_cents"]
                    )
                    position_value += value
                    positions.append(
                        {
                            "ticker": row["ticker"],
                            "signed_qty": row["signed_qty"],
                            "mark_cents": mark["price_cents"],
                            "value_cents": value,
                        }
                    )

                equity = account["balance_cents"] + position_value
            return {
                "ok": True,
                "agent_id": agent_id,
                # Spelled out because integer cents were repeatedly read as
                # dollars: 100000 cents is one thousand dollars, not 100,000.
                "in_dollars": {
                    name: f"${value / 100:,.2f}"
                    for name, value in (
                        ("cash", account["balance_cents"]),
                        ("available", available),
                        ("equity", equity),
                        ("total_pnl", equity - self._initial_cash_cents),
                    )
                },
                "cash_cents": account["balance_cents"],
                "reserved_cents": reserved,
                "available_cents": available,
                "positions": positions,
                "equity_cents": equity,
                "total_pnl_cents": equity - self._initial_cash_cents,
            }
        except _Rejected as exc:
            return _error(exc.code, exc.message)
