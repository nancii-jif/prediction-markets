"""Read-only viewer for a run's agent reasoning and action trace.

Read-only inspection, including live following. Displays recorded messages,
tool results and timing; provider-private reasoning remains opaque.

    python -m prediction_markets trace                      # newest run
    python -m prediction_markets trace --agent agent-03
    python -m prediction_markets trace --follow             # watch a live run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

from ..storage.runs import resolve_database
from .dataset import RunReader, Filters

POLL_SECONDS = 2


def _utc(second: int) -> str:
    return datetime.fromtimestamp(second, timezone.utc).strftime("%H:%M:%S")


def _wrap(text: object, width: int) -> str:
    return textwrap.indent(textwrap.fill(str(text), width - 8), " " * 8)


def _render(row: sqlite3.Row, width: int, full: bool) -> None:
    body = json.loads(row["content_json"])
    clock, sequence = _utc(row["created_at"]), row["sequence_number"]

    if row["entry_type"] == "timing":
        print(f"{clock} [{sequence:>4}] TIMING {body.get('event')}")
        print(_wrap(json.dumps(body, sort_keys=True), width))
        return

    if row["entry_type"] == "model_response":
        message = body.get("message") or {}
        finish = body.get("finish_reason")
        usage = body.get("usage") or {}
        truncated = "  << TRUNCATED, discarded" if finish == "length" else ""
        print(f"\n{clock} [{sequence:>4}] MODEL  finish={finish} "
              f"in={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}{truncated}")
        if body.get("invalid_response"):
            print(_wrap(body["invalid_response"], width))
        if message.get("content"):
            print(_wrap(message["content"], width))
        for call in message.get("tool_calls") or []:
            function = call["function"]
            print(f"        -> {function['name']}({function['arguments']})")
        return

    result = body.get("result") or {}
    status = "ok " if result.get("ok") else "ERR"
    executed = result.get("tool_call_executed", result.get("executed"))
    ran = "ran    " if executed else "skipped"
    print(f"{clock} [{sequence:>4}] TOOL   {body.get('name') or '(rejected)'}  {status}  {ran}")
    encoded = json.dumps(result, sort_keys=True)
    if not full and len(encoded) > 300:
        encoded = f"{encoded[:300]} ...(--full)"
    print(_wrap(encoded, width))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="prediction-markets trace", description=__doc__)
    parser.add_argument("database", nargs="?", help="run.db or run directory; default newest")
    parser.add_argument("--agent", help="only this agent_id")
    parser.add_argument("--full", action="store_true", help="do not truncate tool results")
    parser.add_argument("--follow", action="store_true", help="keep printing new entries")
    parser.add_argument("--width", type=int, default=100)
    parser.add_argument("--segment", choices=("all", "current"), default="all")
    parser.add_argument("--start", help="inclusive ISO timestamp or Unix seconds")
    parser.add_argument("--end", help="exclusive ISO timestamp or Unix seconds")
    args = parser.parse_args(argv)

    path = resolve_database(args.database)
    print(f"# {path}")
    filters = Filters(agent=args.agent, start=args.start, end=args.end)
    reader = RunReader(path, follow=args.follow, segment=args.segment)

    agent = None
    last_entry_id = 0
    try:
        while True:
            # Autocommit means every pass starts a fresh read transaction and
            # therefore sees rows the runner has committed since the last one.
            rows = list(reader.transcript(after=last_entry_id, filters=filters))
            for row in rows:
                if row["agent_id"] != agent:
                    agent = row["agent_id"]
                    print(f"\n{'=' * args.width}\n{agent}\n{'=' * args.width}")
                _render(row, args.width, args.full)
                last_entry_id = max(last_entry_id, row["entry_id"])
            if not args.follow:
                break
            # Following one agent keeps a single heading; following them all
            # regroups per pass, so reset and let the next batch reprint it.
            agent = agent if args.agent else None
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
