"""Historical book reconstruction for runs predating book mutation records."""

import ast
import re
from collections import defaultdict
from datetime import datetime, timezone


class LegacyBookHistory:
    def _shutdown_time(self) -> int | None:
        """Infer the log timezone from an order's embedded Unix timestamp.

        Session timezone is deliberately not used: Python logs use the host's
        local timezone, which can differ. Never guess if there is no anchor.
        """
        if not self.log_path.is_file():
            return None
        offset = None
        shutdown = None
        prefix = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} ")
        with self.log_path.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                match = prefix.match(line)
                if match is None:
                    continue
                wall = int(datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
                if "order accepted: " in line:
                    try:
                        order = ast.literal_eval(line.split("order accepted: ", 1)[1])
                        difference = wall - order["created_at"]
                        # Logging can cross a second boundary. Timezone offsets
                        # are whole minutes; tolerate a one-second delay.
                        candidate = round(difference / 60) * 60
                        if abs(candidate - difference) <= 1:
                            offset = candidate
                    except (ValueError, SyntaxError, KeyError, TypeError):
                        pass
                if "prediction_markets.runner: stopped:" in line and offset is not None:
                    shutdown = wall - offset
        return shutdown

    def _reconstruct_events(self) -> list[dict]:
        trades = defaultdict(list)
        for row in self._conn.execute("SELECT * FROM trades ORDER BY trade_id"):
            trades[row["taker_order_id"]].append(dict(row))
        events = []
        for order_id, order in self._orders.items():
            action = self._order_actions.get(order_id)
            events.append({
                "kind": "submit", "order_id": order_id,
                "timestamp": action.timestamp if action else order["created_at"],
                "sequence": action.sequence if action else 0, "ordinal": order_id,
                "action_id": action.action_id if action else None,
                "agent_id": order["agent_id"], "tool": action.name if action else "submit_order",
                "time_source": action.time_source if action else "order_created",
                "trades": trades[order_id],
            })
        known_cancels = set()
        for action in self._actions:
            if action.name != "cancel_order" or not action.result.get("ok"):
                continue
            order_id = (action.result.get("order") or action.arguments or {}).get("order_id")
            if order_id not in self._orders:
                continue
            known_cancels.add(order_id)
            events.append({
                "kind": "cancel", "order_id": order_id, "timestamp": action.timestamp,
                "sequence": action.sequence, "ordinal": order_id, "action_id": action.action_id,
                "agent_id": action.agent_id, "tool": "cancel_order", "time_source": action.time_source,
            })
        shutdown = self._shutdown_time()
        settlements = dict(self._conn.execute("SELECT ticker, settled_at FROM market_settlements"))
        automatic = defaultdict(list)
        for order_id, order in self._orders.items():
            reason = order["cancel_reason"]
            if order_id in known_cancels or not reason or order["order_type"] == "market":
                continue
            if reason == "run_shutdown" and shutdown is not None:
                automatic[(shutdown, "run_shutdown", "run_log")].append(order_id)
            elif reason == "market_settled" and order["ticker"] in settlements:
                automatic[(settlements[order["ticker"]], "market_settled", "market_settlement")].append(order_id)
            else:
                self.book_history_complete = False
                self.warnings.add(f"Cancellation time unavailable for order {order_id} ({reason}); no timestamp was invented.")
        for (timestamp, tool, source), orders in automatic.items():
            events.append({"kind": "cancel_many", "order_ids": orders, "timestamp": timestamp,
                           "sequence": 2**63 - 1, "ordinal": 0, "action_id": None,
                           "agent_id": None, "tool": tool, "time_source": source})
        return sorted(events, key=lambda event: (event["timestamp"], event["sequence"], event["ordinal"]))

