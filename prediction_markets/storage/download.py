"""Download complete Modal run directories, then build missing local exports."""

import argparse
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from .. import config
from .runs import open_run, stopped_run, validate_run_id


def download_run(run_id, *, runs_dir=None, volume="prediction-markets-runs", force=False):
    """Stage into an existing directory so Modal preserves the run folder."""
    validate_run_id(run_id)
    root = Path(runs_dir) if runs_dir is not None else config.RUNS_DIR
    root.mkdir(parents=True, exist_ok=True)
    destination = root / run_id
    with ExitStack() as stack:
        if destination.exists():
            if not force:
                raise FileExistsError(f"run already exists: {destination}; use --force to replace it")
            if not destination.is_dir() or not (destination / "run.db").is_file():
                raise ValueError("destination must be a run directory; move the incorrectly downloaded file first")
            if not stack.enter_context(stopped_run(destination / "run.db")):
                raise ValueError("cannot replace a running or unverified local experiment")
        temporary = stack.enter_context(tempfile.TemporaryDirectory(prefix=".download-", dir=root))
        stage = Path(temporary)
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", volume, run_id, str(stage), "--force"], check=True)
        downloaded = stage / run_id
        if not (downloaded / "run.db").is_file() or not (downloaded / "run.log").is_file():
            raise ValueError("download did not contain a run directory with run.db and run.log")
        with open_run(downloaded) as conn:
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("downloaded database failed its integrity check")
        from ..analysis.export import main as export
        if not (downloaded / "trace_export").exists():
            with redirect_stdout(io.StringIO()):
                result = export([str(downloaded)])
            if result != 0:
                raise ValueError("could not generate the downloaded run's trace export")
        metadata_path = downloaded / "trace_export" / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            metadata["source_database"] = str((destination / "run.db").resolve())
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        # Both paths live on the same filesystem; publish only complete artifacts.
        backup = stage / "previous"
        if destination.exists():
            destination.rename(backup)
        try:
            downloaded.rename(destination)
        except BaseException:
            if backup.exists():
                backup.rename(destination)
            raise
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(prog="prediction-markets download", description=__doc__)
    parser.add_argument("run_id")
    parser.add_argument("--out", type=Path, help="parent runs directory (default: ./runs)")
    parser.add_argument("--volume", default="prediction-markets-runs")
    parser.add_argument("--force", action="store_true", help="replace an existing stopped local run")
    args = parser.parse_args(argv)
    try:
        path = download_run(args.run_id, runs_dir=args.out, volume=args.volume, force=args.force)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print(f"Saved run artifacts and trace_export to {path}")
    return 0
