"""Export chronological agent actions, complete books, and raw market objects."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Iterator

from .dataset import RunDataset, Filters, parse_time, utc

TraceExport = RunDataset


def _write_objects(path: Path, objects: Iterator[dict], format: str) -> int:
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        if format == "json":
            handle.write("[")
        for obj in objects:
            if format == "json" and count:
                handle.write(",")
            handle.write(json.dumps(obj, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
        if format == "json":
            handle.write("]\n")
    return count


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="prediction-markets export", description=__doc__)
    parser.add_argument("database", nargs="?", help="run.db or run directory; defaults to newest run")
    parser.add_argument("--out", type=Path, help="output directory; defaults to the run's trace_export directory")
    parser.add_argument("--format", choices=("jsonl", "json"), default="jsonl", help="JSON Lines (default) or JSON arrays")
    parser.add_argument("--agent", action="append", help="initiating agent; repeat to select several")
    parser.add_argument("--tool", action="append", help="tool name; repeat to select several")
    parser.add_argument("--market", action="append", help="market ticker; repeat to select several")
    parser.add_argument("--start", help="inclusive ISO timestamp with timezone, or Unix seconds")
    parser.add_argument("--end", help="exclusive ISO timestamp with timezone, or Unix seconds")
    parser.add_argument("--log", type=Path, help="optional legacy run.log; defaults to database's sibling run.log")
    parser.add_argument("--fetch-missing-markets", action="store_true",
                        help="fetch current Kalshi payloads when original run payloads were not saved")
    parser.add_argument("--segment", choices=("all", "current"), default="all",
                        help="include inherited history (default) or only this continuation")
    args = parser.parse_args(argv)
    try:
        filters = Filters(args.agent, args.tool, args.start, args.end, args.market)
        with TraceExport(args.database, log=args.log, segment=args.segment) as run:
            directory = args.out or run.path.parent / "trace_export"
            targets = [directory / (name + "." + args.format) for name in ("actions", "orderbooks")]
            markets_path = directory / "markets.json"
            metadata_path = directory / "metadata.json"
            if any(path.exists() for path in [*targets, markets_path, metadata_path]):
                raise ValueError("export files already exist; choose a different --out directory")
            # Market definitions are small. Fetch before writing any output so
            # network failures never leave an apparently complete export.
            markets = list(run.markets(filters, fetch_missing=args.fetch_missing_markets))
            directory.mkdir(parents=True, exist_ok=True)
            counts = {
                "actions": _write_objects(targets[0], run.actions(filters), args.format),
                "orderbooks": _write_objects(targets[1], run.orderbooks(filters), args.format),
                "markets": _write_objects(markets_path, iter(markets), "json"),
            }
            metadata = {**run.metadata(), "counts": counts,
                        "filters": {"agent": args.agent, "tool": args.tool, "market": args.market,
                                    "start": utc(filters.start) if filters.start is not None else None,
                                    "end": utc(filters.end) if filters.end is not None else None}}
            with metadata_path.open("x", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print(json.dumps({"directory": str(directory), **counts, "warnings": metadata["warnings"]}))
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
