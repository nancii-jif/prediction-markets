"""Downloaded archives and live WAL readers obey distinct SQLite policies."""

import fcntl
from pathlib import Path
import shutil
import sqlite3

import pytest

from prediction_markets import config
from prediction_markets.storage import database
from prediction_markets.storage.runs import open_run, resolve_database


def test_all_run_identifiers_resolve_to_one_database(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    path = root / "trial" / "run.db"
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.setattr(config, "RUNS_DIR", root)
    for value in (None, "trial", path.parent, path):
        assert resolve_database(value) == path.resolve()


def test_downloaded_wal_database_without_sidecars_is_read_without_changes(tmp_path):
    source = tmp_path / "source.db"
    with sqlite3.connect(source) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE example (value)")
        writer.execute("INSERT INTO example VALUES (7)")
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
    writer.close()
    # Modal downloads only the checkpointed main file, with no local sidecars.
    archive = tmp_path / "download"
    archive.mkdir()
    path = archive / "run.db"
    shutil.copyfile(source, path)
    assert path.read_bytes()[18:20] == bytes([2, 2])
    assert not Path(f"{path}-wal").exists()
    (archive / "run.log").write_text("prediction_markets.runner: stopped: duration_elapsed\n")
    before = {p.name: p.read_bytes() for p in archive.iterdir()}
    with open_run(path) as reader:
        assert reader.execute("SELECT value FROM example").fetchone()[0] == 7
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader.execute("INSERT INTO example VALUES (8)")
    assert {p.name: p.read_bytes() for p in archive.iterdir()} == before


def test_follow_reads_new_wal_frames_even_when_wal_initially_empty(tmp_path):
    path = tmp_path / "run.db"
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE example (value)")
    writer.commit()
    writer.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    with (tmp_path / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with open_run(path, follow=True) as reader:
            assert reader.execute("SELECT COUNT(*) FROM example").fetchone()[0] == 0
            writer.execute("INSERT INTO example VALUES (9)")
            writer.commit()
            assert reader.execute("SELECT value FROM example").fetchone()[0] == 9
        with pytest.raises(ValueError, match="running"):
            with open_run(path, require_stopped=True):
                pass
    writer.close()


def test_snapshot_keeps_a_consistent_view_of_active_run(tmp_path):
    path = tmp_path / "run.db"
    writer = sqlite3.connect(path)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("CREATE TABLE example (value)")
    writer.commit()
    with open_run(path) as reader:
        assert reader.execute("SELECT COUNT(*) FROM example").fetchone()[0] == 0
        writer.execute("INSERT INTO example VALUES (9)")
        writer.commit()
        assert reader.execute("SELECT COUNT(*) FROM example").fetchone()[0] == 0
    with open_run(path) as reader:
        assert reader.execute("SELECT COUNT(*) FROM example").fetchone()[0] == 1
    writer.close()


def test_explicit_connection_does_not_replace_another_run(participant_rig, tmp_path):
    rig = participant_rig
    other = database.init(tmp_path / "other" / "run.db", run_settings=rig.settings, register=False)
    try:
        assert database.db() is rig.conn
        assert rig.conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 10
        assert other.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 0
    finally:
        other.close()
