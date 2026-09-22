"""Run paths, safe read snapshots, frozen settings, and continuation lineage."""

from __future__ import annotations

import fcntl
import json
import re
import shutil
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from .. import config
from .schema import SCHEMA_VERSION


def validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise ValueError("run_id must be a 1-128 character safe path segment beginning with a letter or number")
    return run_id


def create_run_directory(run_id: str, *, runs_dir: str | Path | None = None) -> Path:
    root = Path(runs_dir) if runs_dir is not None else config.RUNS_DIR
    path = root / validate_run_id(run_id)
    root.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite or resume existing run: {path}") from exc
    return path


def resolve_database(value: str | Path | None = None, *, runs_dir: Path | None = None) -> Path:
    """Resolve an ID, directory, database path, or the newest run consistently."""
    root = Path(runs_dir) if runs_dir is not None else config.RUNS_DIR
    if value is None:
        paths = sorted(root.glob("*/run.db"))
        if not paths:
            raise ValueError(f"no run databases found in {root}")
        return paths[-1].resolve()
    path = Path(value).expanduser()
    if not path.exists() and len(path.parts) == 1:
        path = root / path
    if path.is_dir():
        path /= "run.db"
    if not path.is_file():
        raise FileNotFoundError(f"run database does not exist: {path}")
    return path.resolve()


resolve_source = resolve_database


@contextmanager
def stopped_run(path: Path):
    """Hold a shared lock when stopped; old runs use their shutdown marker."""
    lock_path = path.parent / "run.lock"
    if lock_path.exists():
        with lock_path.open("rb") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
            else:
                yield True
    else:
        log = path.parent / "run.log"
        yield log.is_file() and "prediction_markets.runner: stopped:" in log.read_text(errors="replace")


@contextmanager
def open_run(source=None, *, follow=False, require_stopped=False):
    """Read without changing archived artifacts or ignoring committed WAL frames.

    Immutable reads require positive evidence of a stopped run. For stopped
    databases with a WAL, recover a private copy. Active/unknown runs use a
    normal SQLite read connection; follow mode starts a fresh read each poll.
    """
    path = resolve_database(source)
    with stopped_run(path) as stopped:
        if require_stopped and not stopped:
            if (path.parent / "run.lock").exists():
                raise ValueError("cannot resume a running experiment; stop it first")
            raise ValueError("legacy resume requires a stopped run with its run.log")
        with tempfile.TemporaryDirectory(prefix="prediction-market-snapshot-") as temporary:
            wal = Path(f"{path}-wal")
            if stopped and wal.exists() and wal.stat().st_size:
                snapshot = Path(temporary) / "run.db"
                shutil.copyfile(path, snapshot)
                shutil.copyfile(wal, Path(f"{snapshot}-wal"))
                conn = sqlite3.connect(snapshot)
            else:
                query = "?mode=ro&immutable=1" if stopped else "?mode=ro"
                conn = sqlite3.connect(path.as_uri() + query, uri=True)
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA query_only=ON")
                if not follow:
                    conn.execute("BEGIN")
                yield conn
            finally:
                conn.close()


def source_snapshot(source):
    return open_run(source, require_stopped=True)


def settings_from_connection(conn: sqlite3.Connection) -> config.Settings:
    row = conn.execute("SELECT schema_version, config_json FROM run_settings WHERE singleton_id=1").fetchone()
    if row is None or row[0] != SCHEMA_VERSION:
        raise ValueError("unsupported or missing saved run schema")
    values = json.loads(row[1])
    agent_ids = values.pop("agent_ids", None)
    settings = config.Settings(**values)
    if agent_ids != list(settings.agent_ids):
        raise ValueError("saved agent IDs do not match the frozen configuration")
    if {row[0] for row in conn.execute("SELECT agent_id FROM accounts")} != set(settings.agent_ids):
        raise ValueError("saved accounts do not match the frozen agent setup")
    if {row[0] for row in conn.execute("SELECT agent_id FROM agent_state")} != set(settings.agent_ids):
        raise ValueError("saved participant state is incomplete")
    if conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] != settings.market_count:
        raise ValueError("parent run has no complete market cohort to resume")
    return settings


def load_settings(source) -> config.Settings:
    with source_snapshot(source) as conn:
        return settings_from_connection(conn)


def begin_segment(conn, run_id: str, source: Path | None) -> dict:
    parent = source.parent.name if source else None
    previous = conn.execute("SELECT run_id, root_run_id FROM run_segments ORDER BY rowid DESC LIMIT 1").fetchone()
    if previous:
        parent, root = previous
    else:
        root = parent or run_id
    first_entry = conn.execute("SELECT COALESCE(MAX(entry_id), 0)+1 FROM transcript_entries").fetchone()[0]
    first_trade = conn.execute("SELECT COALESCE(MAX(trade_id), 0)+1 FROM trades").fetchone()[0]
    with conn:
        conn.execute(
            "INSERT INTO run_segments (run_id, parent_run_id, root_run_id, started_at, "
            "first_transcript_entry_id, first_trade_id) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, parent, root, int(time.time()), first_entry, first_trade),
        )
    return {"parent_run_id": parent, "root_run_id": root, "first_trade_id": first_trade}


def finish_segment(conn, run_id: str, reason: str) -> None:
    with conn:
        conn.execute("UPDATE run_segments SET stopped_at=?, stop_reason=? WHERE run_id=?",
                     (int(time.time()), reason, run_id))
