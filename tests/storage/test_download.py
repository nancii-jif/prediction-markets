"""Modal directory downloads are staged and verified before replacement."""

import json
from pathlib import Path
import sqlite3
import subprocess

import pytest

from prediction_markets.storage import download


def fake_modal(rig, monkeypatch, calls):
    def run(command, *, check):
        assert check and command[1:5] == ["-m", "modal", "volume", "get"]
        target = Path(command[-2])
        assert target.is_dir()  # Modal must receive an existing parent directory.
        calls.append(command)
        target /= command[-3]
        target.mkdir()
        conn = sqlite3.connect(target / "run.db")
        rig.conn.backup(conn)
        conn.close()
        (target / "run.log").write_text("prediction_markets.runner: stopped: duration_elapsed\n")
    monkeypatch.setattr(download.subprocess, "run", run)


def test_download_creates_auxiliary_exports_without_live_requests(participant_rig, tmp_path, monkeypatch):
    calls = []
    fake_modal(participant_rig, monkeypatch, calls)
    root = tmp_path / "downloads"
    path = download.download_run("saved-run", runs_dir=root)
    assert path == root / "saved-run" and len(calls) == 1
    assert (path / "run.db").is_file() and (path / "run.log").is_file()
    assert {p.name for p in (path / "trace_export").iterdir()} == {
        "actions.jsonl", "orderbooks.jsonl", "markets.json", "metadata.json",
    }
    assert json.loads((path / "trace_export" / "metadata.json").read_text())["source_database"] == str(path / "run.db")
    with pytest.raises(FileExistsError):
        download.download_run("saved-run", runs_dir=root)
    assert len(calls) == 1


def test_failed_forced_download_preserves_existing_run(participant_rig, tmp_path, monkeypatch):
    calls = []
    fake_modal(participant_rig, monkeypatch, calls)
    path = download.download_run("saved-run", runs_dir=tmp_path / "downloads")
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(download.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        download.download_run("saved-run", runs_dir=path.parent, force=True)
    assert {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()} == before
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("run_id", ["../outside", "/absolute", "x/y", "..", ""])
def test_download_rejects_paths_before_invoking_modal(tmp_path, run_id):
    with pytest.raises(ValueError, match="safe path segment"):
        download.download_run(run_id, runs_dir=tmp_path)
