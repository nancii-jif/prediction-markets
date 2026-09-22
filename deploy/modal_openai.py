"""Run the existing exchange and OpenAI agents on Modal CPU, with durable outputs.

Usage: modal run --detach deploy/modal_openai.py --config configs/openai.yaml
Resume: modal run --detach deploy/modal_openai.py --resume-from RUN_ID
Only the Python package is uploaded. Credentials come from a Modal Secret;
configs travel as validated values, never as a mounted project or .env file.
"""

import asyncio
import json
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(Path(__file__).resolve().parents[1] / "requirements.txt"))
    .env({"TIKTOKEN_CACHE_DIR": "/opt/tiktoken-cache"})
    .run_commands("python -c 'import tiktoken; tiktoken.get_encoding(\"o200k_base\")'")
    .add_local_python_source("prediction_markets")
)
app = modal.App("prediction-markets-openai")
outputs = modal.Volume.from_name("prediction-markets-runs", create_if_missing=True)


async def execute_run(settings_values, artifact_root, commit, *, resume_from=None):
    """Copy closed artifacts and await the Volume commit, including on cancellation."""
    import logging
    import shutil
    import tempfile
    from datetime import datetime, timezone
    from uuid import uuid4

    from prediction_markets import config
    from prediction_markets.runtime import runner
    from prediction_markets.storage import runs as resume

    source = None
    if resume_from:
        if settings_values is not None:
            raise ValueError("resume inherits the saved config; do not also pass a config")
        if not isinstance(resume_from, str) or resume_from in {".", ".."} or Path(resume_from).name != resume_from:
            raise ValueError("Modal --resume-from must be a run ID in the runs volume")
        source = resume.resolve_source(Path(artifact_root) / resume_from)
        settings = resume.load_settings(source)
    else:
        settings = config.Settings(**settings_values)
    if settings.model_provider != "openai":
        raise ValueError("the CPU runner requires model_provider: openai")
    if settings.run_duration_seconds > 23 * 3600:
        raise ValueError("run duration must be at most 23 hours within the Modal timeout")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    destination = Path(artifact_root) / run_id
    with tempfile.TemporaryDirectory(prefix="prediction-market-run-") as temporary:
        try:
            options = {"resume_from": str(source)} if source is not None else {}
            summary = await runner.run_live(settings, run_id=run_id, runs_dir=Path(temporary), **options)
        finally:
            source = Path(temporary) / run_id
            if source.exists():
                # run_live/run finish cancellation and close SQLite before returning.
                shutil.copytree(source, destination)
                await commit()
                logging.getLogger(__name__).info("saved run artifacts to %s", destination)
    return {**summary, "run_directory": str(destination)}


@app.function(
    image=image, cpu=1, memory=2048, max_containers=1, retries=0, timeout=86400,
    volumes={"/runs": outputs},
    secrets=[modal.Secret.from_name("prediction-markets-openai", required_keys=["OPENAI_API_KEY"])],
)
def run_experiment(settings_values: dict | None = None, resume_from: str | None = None):
    return asyncio.run(execute_run(settings_values, "/runs", outputs.commit.aio, resume_from=resume_from))


@app.local_entrypoint()
def main(config: str = "", resume_from: str = ""):
    from dataclasses import asdict
    from prediction_markets.config import load

    if config and resume_from:
        raise ValueError("use --config or --resume-from; resume does not accept config overrides")
    # Hand the function call to the detached App and return its handle.
    if resume_from:
        call = run_experiment.spawn(resume_from=resume_from)
    else:
        settings = load(config or "configs/openai.yaml")
        if settings.model_provider != "openai":
            raise ValueError("use an OpenAI configuration for this CPU runner")
        if settings.run_duration_seconds > 23 * 3600:
            raise ValueError("run duration must be at most 23 hours within the Modal timeout")
        call = run_experiment.spawn(asdict(settings))
    print(json.dumps({
        "function_call_id": call.object_id,
        "logs": f"modal app logs {app.app_id}" if app.app_id else "modal app list -e main",
    }, indent=2))
