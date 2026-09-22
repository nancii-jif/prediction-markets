"""Fresh SQLite storage for the minimal continuous-market runtime.

This module exposes run initialization and fixed-cohort storage. The exchange
owns accounting and order mutations so an entire match commits in one
transaction, including final settlement; participant loops own transcript writes.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .. import config

from .schema import SCHEMA_VERSION, SCHEMA, CONTINUATION_SCHEMA

_MARKET_FIELDS = frozenset(
    {
        "ticker",
        "event_ticker",
        "market_type",
        "title",
        "rules",
        "status",
        "close_time",
        "expected_expiration_time",
        "latest_expiration_time",
        "last_successful_poll",
        "payout_cents",
    }
)

# Compatibility for old direct callers. Runtime connections use register=False.
_conn: sqlite3.Connection | None = None


def _configure_connection(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init(
    path: str | Path,
    *,
    run_settings: config.Settings,
    register: bool = True,
) -> sqlite3.Connection:
    """Create exactly one new run database and initialize its participants.

    The path is claimed with exclusive file creation before SQLite opens it.
    Existing files are never opened, migrated, overwritten, or resumed.
    """
    global _conn

    if not isinstance(run_settings, config.Settings):
        raise TypeError("run_settings must be a validated config.Settings")
    if run_settings.schema_version != SCHEMA_VERSION:
        raise ValueError(
            "run settings and database schema versions do not match: "
            f"{run_settings.schema_version} != {SCHEMA_VERSION}"
        )

    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        db_path.open("xb").close()
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite or resume existing database: {db_path}"
        ) from exc

    if register:
        close()
    conn: sqlite3.Connection | None = None
    try:
        conn = _configure_connection(db_path)
        conn.executescript(SCHEMA)
        conn.executescript(CONTINUATION_SCHEMA)
        created_at = int(time.time())
        with conn:
            conn.execute(
                "INSERT INTO run_settings "
                "(singleton_id, schema_version, created_at, config_json) "
                "VALUES (1, ?, ?, ?)",
                (
                    SCHEMA_VERSION,
                    created_at,
                    config.as_json(run_settings),
                ),
            )
            conn.executemany(
                "INSERT INTO accounts (agent_id, balance_cents) VALUES (?, ?)",
                (
                    (agent_id, run_settings.initial_cash_cents)
                    for agent_id in run_settings.agent_ids
                ),
            )
            conn.executemany(
                "INSERT INTO agent_state (agent_id) VALUES (?)",
                ((agent_id,) for agent_id in run_settings.agent_ids),
            )
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    except Exception:
        if conn is not None:
            conn.close()
        for candidate in (
            db_path,
            Path(f"{db_path}-wal"),
            Path(f"{db_path}-shm"),
        ):
            candidate.unlink(missing_ok=True)
        raise

    if register:
        _conn = conn
    return conn


def fork(path: str | Path, source: str | Path, *, run_settings: config.Settings,
         register: bool = True) -> sqlite3.Connection:
    """Back up a stopped parent into a new DB; never reopen it for writes."""
    from .runs import source_snapshot, settings_from_connection

    global _conn
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with source_snapshot(source) as parent:
        if settings_from_connection(parent) != run_settings:
            raise ValueError("resume must use the parent's entire saved configuration; overrides are not supported")
        db_path.open("xb").close()
        if register:
            close()
        conn = None
        try:
            conn = _configure_connection(db_path)
            parent.backup(conn)
            conn.executescript(CONTINUATION_SCHEMA)
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok" or conn.execute("PRAGMA foreign_key_check").fetchone():
                raise ValueError("parent run failed SQLite integrity checks")
        except BaseException:
            if conn is not None:
                conn.close()
            for candidate in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
                candidate.unlink(missing_ok=True)
            raise
    if register:
        _conn = conn
    return conn


def db() -> sqlite3.Connection:
    """Return the initialized run connection; never create one implicitly."""
    if _conn is None:
        raise RuntimeError("database has not been initialized")
    return _conn


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


def _stored_market_count(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT config_json FROM run_settings WHERE singleton_id = 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("database is missing its frozen run settings")
    values = json.loads(row["config_json"])
    return int(values["market_count"])


def _require_integer(name: str, value: Any, *, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")


def _validate_market(
    row: Mapping[str, Any], index: int, *, require_expiration: bool = True,
) -> None:
    missing = sorted(_MARKET_FIELDS - row.keys())
    unknown = sorted(row.keys() - _MARKET_FIELDS)
    if missing:
        raise ValueError(f"market {index} is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"market {index} has unrecognized fields: {unknown}")

    for name in ("ticker", "event_ticker", "title", "rules", "status"):
        value = row[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"market {index} has an empty {name}")
    if row["market_type"] != "binary":
        raise ValueError(f"market {index} is not binary")
    for name in (
        "close_time",
        "last_successful_poll",
    ):
        _require_integer(f"market {index} {name}", row[name])
    if row["expected_expiration_time"] is not None:
        _require_integer(
            f"market {index} expected_expiration_time",
            row["expected_expiration_time"],
        )
    if row["latest_expiration_time"] is not None:
        _require_integer(
            f"market {index} latest_expiration_time",
            row["latest_expiration_time"],
        )
    if require_expiration and all(
        row[name] is None
        for name in ("expected_expiration_time", "latest_expiration_time")
    ):
        raise ValueError(f"market {index} requires an expected or latest expiration")
    if row["payout_cents"] is not None:
        _require_integer(
            f"market {index} payout_cents",
            row["payout_cents"],
            minimum=0,
        )
        if row["payout_cents"] > 100:
            raise ValueError(f"market {index} payout_cents exceeds 100")


def store_market_cohort(
    rows: Sequence[Mapping[str, Any]], *,
    raw_markets: Mapping[str, Mapping[str, Any]] | None = None,
    source_url: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> int:
    """Atomically store the complete fixed cohort exactly once.

    Selection and qualification happen in live_markets.select_markets.
    Strict normalized fields keep source prices out of agent observations.
    Optional original API objects are archived separately for offline exports.
    """
    conn = conn if conn is not None else db()
    cohort = list(rows)
    expected_count = _stored_market_count(conn)
    if len(cohort) != expected_count:
        raise ValueError(
            f"fixed cohort requires {expected_count} markets, got {len(cohort)}"
        )

    for index, row in enumerate(cohort, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"market {index} must be a mapping")
        _validate_market(row, index)

    tickers = [row["ticker"] for row in cohort]
    events = [row["event_ticker"] for row in cohort]
    if len(set(tickers)) != len(tickers):
        raise ValueError("fixed cohort contains duplicate tickers")
    if len(set(events)) != len(events):
        raise ValueError("fixed cohort must contain distinct events")

    sources = []
    if raw_markets is not None:
        if set(raw_markets) != set(tickers):
            raise ValueError("raw market payloads must match the selected cohort")
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError("raw market payloads require a source_url")
        for row in cohort:
            payload = raw_markets[row["ticker"]]
            if not isinstance(payload, Mapping) or payload.get("ticker") != row["ticker"]:
                raise ValueError("raw market payload ticker does not match the selected market")
            sources.append((row["ticker"], row["last_successful_poll"], source_url,
                            json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, separators=(",", ":"))))

    with conn:
        existing = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
        if existing:
            raise RuntimeError("fixed market cohort has already been stored")
        cursor = conn.executemany(
            "INSERT INTO markets ("
            "ticker, event_ticker, cohort_index, market_type, title, rules, "
            "status, close_time, expected_expiration_time, "
            "latest_expiration_time, last_successful_poll, payout_cents"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    row["ticker"],
                    row["event_ticker"],
                    index,
                    row["market_type"],
                    row["title"],
                    row["rules"],
                    row["status"],
                    row["close_time"],
                    row["expected_expiration_time"],
                    row["latest_expiration_time"],
                    row["last_successful_poll"],
                    row["payout_cents"],
                )
                for index, row in enumerate(cohort, start=1)
            ),
        )
        conn.executemany(
            "INSERT INTO market_api_responses (ticker, captured_at, source_url, payload_json) VALUES (?, ?, ?, ?)",
            sources,
        )
    return cursor.rowcount
