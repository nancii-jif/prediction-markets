"""Plot internal fills against contemporaneous Kalshi trades.

python -m prediction_markets plot RUN_ID --combined
python -m prediction_markets plot --ticker TICKER --agent-id agent-03
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import sys

from .. import config
from ..integrations.kalshi import fetch_trade_history
from ..storage.runs import resolve_database
from .dataset import RunReader, utc as iso, parse_time as timestamp

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")


def resolve_run(run_id: str | None, runs_dir: Path) -> Path:
    return resolve_database(run_id, runs_dir=runs_dir)


def read_run(path: Path, tickers=None, agent_id=None, *, segment="all", start=None, end=None):
    with RunReader(path, segment=segment) as run:
        # Fetch/cache one market history window shared by every agent selection.
        return run.prices(tickers, agent_id, start=start, end=end, filter_agent=False)


def normalize_history(raw: list[dict], ticker: str) -> list[dict]:
    history = []
    for trade in raw:
        try:
            if trade["ticker"] != ticker:
                raise ValueError("wrong ticker")
            trade_time = timestamp(trade["created_time"])
            dollars = trade.get("yes_price_dollars")
            price = Decimal(str(dollars)) * 100 if dollars is not None else Decimal(str(trade["yes_price"]))
            if not price.is_finite() or not 0 <= price <= 100:
                raise ValueError("price outside 0–100 cents")
            history.append({"timestamp": trade_time, "price_cents": float(price),
                            "trade_id": trade["trade_id"]})
        except (KeyError, TypeError, AttributeError, ValueError, InvalidOperation) as exc:
            raise ValueError(f"malformed Kalshi trade for {ticker}: {exc}") from exc
    # Stable ordering preserves API order for exact timestamp ties; there is no
    # public sequence field to recover finer ordering for such ties.
    return sorted(history, key=lambda trade: trade["timestamp"])


def load_history(ticker, start, end, base_url, cache_dir, *, refresh=False, offline=False):
    key = {"schema_version": 1, "ticker": ticker, "start_ts": start, "end_ts": end, "base_url": base_url}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
    path = cache_dir / f"{digest}.json.gz"
    if path.exists() and not refresh:
        with gzip.open(path, "rt") as file:
            saved = json.load(file)
        if saved.get("request") != key:
            raise ValueError(f"cache request mismatch: {path}")
        return normalize_history(saved["trades"], ticker)
    if offline:
        raise ValueError(f"no cached history for {ticker}; run once without --offline")
    raw = fetch_trade_history(ticker, start, end, base_url=base_url)
    normalized = normalize_history(raw, ticker)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt") as file:
        json.dump({"request": key, "fetched_at": iso(datetime.now(timezone.utc).timestamp()), "trades": raw}, file)
    temporary.replace(path)
    return normalized


def pair_prices(trades: list[dict], history: list[dict], max_age_seconds: float) -> list[dict]:
    """Backward as-of join: never pair with a future human trade."""
    timestamps = [t["timestamp"] for t in history]
    points = []
    for trade in trades:
        timestamp = trade["executed_at"]
        index = bisect_right(timestamps, timestamp) - 1
        human = history[index] if index >= 0 else None
        age = timestamp - human["timestamp"] if human else None
        usable = human is not None and age <= max_age_seconds
        points.append({
            "ticker": trade["ticker"], "trade_id": trade["trade_id"], "time": iso(timestamp),
            "agent_price_cents": trade["price_cents"], "quantity": trade["quantity"],
            "buyer": trade["buyer"], "seller": trade["seller"],
            "self_trade": trade["buyer"] == trade["seller"],
            "kalshi_price_cents": human["price_cents"] if usable else None,
            "kalshi_trade_id": human["trade_id"] if human else None,
            "kalshi_time": iso(human["timestamp"]) if human else None,
            "kalshi_age_seconds": age,
            "status": "matched" if usable else ("stale_human_price" if human else "no_prior_human_trade"),
        })
    return points


def render(markets, points, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as exc:
        raise ValueError("install plotting dependencies: pip install -r requirements-analysis.txt") from exc

    fig, ax = plt.subplots(figsize=(13, 8), facecolor="white")
    fig.subplots_adjust(left=.09, right=.61, bottom=.17, top=.85)
    ax.set(xlim=(-2, 102), ylim=(-2, 102), xlabel="Agent exchange price · YES cents (Kalshi)",
           ylabel="Human market price · YES cents", xticks=range(0, 101, 20), yticks=range(0, 101, 20))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#e6e9ee", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.plot([0, 100], [0, 100], "--", color="#78828f", linewidth=1.2, zorder=1)
    handles = [Line2D([0], [0], color="#78828f", linestyle="--", label="y = x · price agreement")]
    groups = defaultdict(list)
    for point in points:
        groups[point["ticker"]].append(point)
    for market in markets:
        ticker = market["ticker"]
        series = groups[ticker]
        matched = [p for p in series if p["status"] == "matched"]
        index = market["cohort_index"] - 1
        color = plt.get_cmap("tab10")(index % 10)
        marker = MARKERS[index % len(MARKERS)]
        label = ticker + (f"  ({len(matched)} {'point' if len(matched) == 1 else 'points'})" if matched else
                          ("  (no matching human history)" if series else "  (no selected fills)"))
        handles.append(Line2D([0], [0], color=color if matched else "#adb3bb", marker=marker,
                              linewidth=1.5, label=label))
        # NaN breaks the path at missing history rather than bridging a gap.
        xs = [p["agent_price_cents"] if p["status"] == "matched" else float("nan") for p in series]
        ys = [p["kalshi_price_cents"] if p["status"] == "matched" else float("nan") for p in series]
        ax.plot(xs, ys, color=color, marker=marker, markersize=7, linewidth=1.5, alpha=.85,
                markeredgecolor="white", markeredgewidth=.5, zorder=3)
        labels = defaultdict(list)
        for number, point in enumerate(series, 1):
            if point["status"] != "matched":
                continue
            xy = (point["agent_price_cents"], point["kalshi_price_cents"])
            labels[xy].append(number)
        for xy, numbers in labels.items():
            ax.annotate(",".join(map(str, numbers)), xy, xytext=(5, 6 + 8 * ((numbers[0] - 1) % 2)),
                        textcoords="offset points", color=color, fontsize=8)
        for first, second in zip(series, series[1:]):
            if first["status"] != "matched" or second["status"] != "matched":
                continue
            a = (first["agent_price_cents"], first["kalshi_price_cents"])
            b = (second["agent_price_cents"], second["kalshi_price_cents"])
            if a != b:
                ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": color,
                            "lw": 1, "shrinkA": 7, "shrinkB": 7}, zorder=2)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.04, 1), frameon=False,
              fontsize=8.5, labelspacing=1.25, borderaxespad=0)
    fig.suptitle("Human market vs agent exchange", x=.09, ha="left", fontsize=19, fontweight="bold", y=.96)
    fig.text(.09, .90, title, fontsize=10, color="#505966")
    matched = [p for p in points if p["status"] == "matched"]
    span = f"{points[0]['time']} → {points[-1]['time']}" if points else "No selected fills"
    fig.text(.09, .085, f"{len(matched)}/{len(points)} fills paired · {span}", fontsize=9, color="#505966")
    fig.text(.09, .045, "Numbers/arrows follow time within each market; overlapping points may coincide.\n"
             "Kalshi: last trade at or before each fill. The diagonal measures price agreement, not outcome calibration.",
             fontsize=8.5, color="#505966")
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="prediction-markets plot", description=__doc__)
    parser.add_argument("database", nargs="?", help="Run ID, directory, or run.db")
    parser.add_argument("--segment", choices=("all", "current"), default="all")
    parser.add_argument("--start", help="inclusive ISO timestamp or Unix seconds")
    parser.add_argument("--end", help="exclusive ISO timestamp or Unix seconds")
    parser.add_argument("--run-id", "--run", help="Run ID, run directory, or run.db; defaults to newest run")
    parser.add_argument("--runs-dir", type=Path, default=config.RUNS_DIR)
    parser.add_argument("--ticker", "--market-ticker", action="append", help="Repeat to select markets within the run")
    parser.add_argument("--agent-id", "--agent", help="Only fills where this agent is buyer or seller; empty means all")
    parser.add_argument("--combined", action="store_true", help="All selected markets on one set of axes")
    parser.add_argument("--exclude-self-trades", action="store_true")
    parser.add_argument("--max-human-age-hours", type=float, default=24,
                        help="Maximum human last-trade age and initial lookback (default: 24)")
    parser.add_argument("--out", type=Path, help="Output directory; default: run/price_comparison[/agent-ID]")
    parser.add_argument("--cache-dir", type=Path, help="Reusable raw historical trade cache")
    parser.add_argument("--offline", action="store_true", help="Use cached history only")
    parser.add_argument("--refresh", action="store_true", help="Refetch historical data")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_human_age_hours) or args.max_human_age_hours <= 0:
        parser.error("--max-human-age-hours must be finite and positive")
    if args.offline and args.refresh:
        parser.error("--offline and --refresh cannot be used together")
    try:
        if args.run_id and args.database:
            raise ValueError("supply a positional run or --run-id, not both")
        path = resolve_run(args.run_id or args.database, args.runs_dir)
        markets, all_trades, settings = read_run(path, args.ticker, args.agent_id,
            segment=args.segment, start=args.start, end=args.end)
        out = args.out or path.parent / "price_comparison"
        if args.ticker and args.out is None:
            names = sorted({m["ticker"] for m in markets})
            selection = names[0] if len(names) == 1 else "selected-" + hashlib.sha256(
                json.dumps(names).encode()).hexdigest()[:12]
            out /= re.sub(r"[^A-Za-z0-9_.-]", "_", selection)
        if args.agent_id and args.out is None:
            out /= args.agent_id
        out.mkdir(parents=True, exist_ok=True)
        cache = args.cache_dir or path.parent / "kalshi_history_cache"
        groups = defaultdict(list)
        for trade in all_trades:
            groups[trade["ticker"]].append(trade)
        points = []
        max_age = args.max_human_age_hours * 3600
        for market in markets:
            ticker = market["ticker"]
            trades = groups[ticker]
            selected = [t for t in trades if (not args.agent_id or args.agent_id in (t["buyer"], t["seller"]))
                        and (not args.exclude_self_trades or t["buyer"] != t["seller"])]
            if not selected:
                continue
            print(f"Matching {ticker}: {len(selected)} fills", file=sys.stderr)
            history = load_history(ticker, math.floor(trades[0]["executed_at"] - max_age),
                                   trades[-1]["executed_at"], settings["kalshi_base_url"], cache,
                                   refresh=args.refresh, offline=args.offline)
            points.extend(pair_prices(selected, history, max_age))
        points.sort(key=lambda p: (p["time"], p["trade_id"]))
        with (out / "points.jsonl").open("w") as file:
            for point in points:
                file.write(json.dumps(point, allow_nan=False) + "\n")
        missing = [m["ticker"] for m in markets if not any(p["ticker"] == m["ticker"] for p in points)]
        unmatched = sum(p["status"] != "matched" for p in points)
        metadata = {"source_database": str(path), "run_id": path.parent.name, "agent_id": args.agent_id or None,
                    "segment": args.segment, "start": args.start, "end": args.end,
                    "markets": [{"ticker": m["ticker"], "title": m["title"]} for m in markets],
                    "agent_price": "executed YES fill price", "human_price": "last Kalshi YES trade at or before fill",
                    "max_human_age_seconds": max_age, "fill_time_resolution_seconds": 1,
                    "include_self_trades": not args.exclude_self_trades, "selected_fills": len(points),
                    "matched_fills": len(points) - unmatched, "unmatched_fills": unmatched,
                    "markets_without_selected_fills": missing}
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        subtitle = f"Run {path.parent.name} · {args.agent_id or 'all agents'}"
        batches = [markets] if args.combined else [[m] for m in markets]
        for batch in batches:
            wanted = {m["ticker"] for m in batch}
            batch_points = [p for p in points if p["ticker"] in wanted]
            name = "combined" if args.combined else re.sub(r"[^A-Za-z0-9_.-]", "_", batch[0]["ticker"])
            render(batch, batch_points, out / f"{name}.png", subtitle)
            render(batch, batch_points, out / f"{name}.svg", subtitle)
        print(f"Saved {len(points) - unmatched}/{len(points)} matched fills to {out}")
        if missing:
            print("No selected fills: " + ", ".join(missing))
        if unmatched:
            print(f"WARNING: {unmatched} fills lack a sufficiently recent prior human trade; see points.jsonl", file=sys.stderr)
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
