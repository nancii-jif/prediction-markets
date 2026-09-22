"""Static deployment contract; no Modal account, image build or GPU required."""

import ast
import asyncio
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from prediction_markets import config
from prediction_markets.storage import database as database
from prediction_markets.runtime import runner as runner


def test_deployment_pins_match_local_config_and_one_shared_endpoint():
    settings = config.load(config.CONFIG_DIR / "mvp.yaml")
    tree = ast.parse((config.ROOT / "deploy" / "modal_qwen.py").read_text())
    constants = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
    }
    assert constants["MODEL"] == settings.model_name
    assert constants["REVISION"] == settings.model_revision
    serve = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "serve")
    decorators = {node.func.attr: node for node in serve.decorator_list}
    options = {kw.arg: kw.value for kw in decorators["function"].keywords}
    assert ast.literal_eval(options["gpu"]) == "A100-40GB:1"
    assert ast.literal_eval(options["min_containers"]) == ast.literal_eval(options["max_containers"]) == 1
    assert ast.literal_eval(decorators["concurrent"].keywords[0].value) == settings.max_concurrent_inference
    assert decorators["web_server"].args[0].value == 8000
    command = next(node for node in ast.walk(serve) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "Popen").args[0]
    command = [node.value if isinstance(node, ast.Constant) else constants[node.id] for node in command.elts]
    for flag, value in {
        "--revision": settings.model_revision, "--tokenizer-revision": settings.model_revision,
        "--max-model-len": str(settings.server_context_tokens), "--max-num-seqs": "8",
        "--tensor-parallel-size": "1", "--tool-call-parser": "qwen3_coder",
        "--reasoning-parser": "qwen3", "--default-chat-template-kwargs": '{"enable_thinking":false}',
    }.items():
        assert command[command.index(flag) + 1] == value
    assert "--language-model-only" in command
    assert "--api-key" not in command  # Secret supplies VLLM_API_KEY; no shell/CLI exposure.
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    pip_call = next(node for node in calls if node.func.attr == "uv_pip_install")
    packages = {arg.value for arg in pip_call.args}
    assert "vllm==0.17.1" in packages
    local_requirements = set((config.ROOT / "requirements.txt").read_text().splitlines())
    assert packages - {"vllm==0.17.1"} <= local_requirements
    assert not any(node.func.attr in {"add_local_dir", "add_local_python_source"} for node in calls)


def test_openai_deployment_is_cpu_only_and_uploads_only_the_runtime_package():
    tree = ast.parse((config.ROOT / "deploy" / "modal_openai.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_experiment")
    options = {kw.arg: kw.value for kw in function.decorator_list[0].keywords}
    assert "gpu" not in options and "min_containers" not in options
    assert ast.literal_eval(options["cpu"]) == 1
    assert ast.literal_eval(options["max_containers"]) == 1
    assert ast.literal_eval(options["retries"]) == 0
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    sources = [node for node in calls if node.func.attr == "add_local_python_source"]
    assert len(sources) == 1 and ast.literal_eval(sources[0].args[0]) == "prediction_markets"
    assert not any(node.func.attr in {"add_local_dir", "add_local_file"} for node in calls)


@pytest.mark.parametrize("outcome", ["success", "failure", "canceled"])
def test_modal_run_persists_closed_database_and_logs_even_on_failure(tmp_path, monkeypatch, outcome):
    # Load just the helper, so tests need no Modal dependency or credentials.
    tree = ast.parse((config.ROOT / "deploy" / "modal_openai.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute_run")
    namespace = {"Path": Path, "__name__": "test_modal_openai"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "modal_openai.py", "exec"), namespace)
    settings = config.load(config.ROOT / "configs" / "openai.yaml")
    original_root = config.RUNS_DIR
    commits = []
    outputs = tmp_path / "outputs"

    async def run_live(settings, *, run_id, runs_dir):
        assert config.RUNS_DIR == original_root
        from prediction_markets.storage.runs import create_run_directory
        directory = create_run_directory(run_id, runs_dir=runs_dir)
        (directory / "run.log").write_text("run log")
        database.init(directory / "run.db", run_settings=settings)
        database.close()
        if outcome == "failure":
            raise RuntimeError("scripted inference failure")
        if outcome == "canceled":
            asyncio.current_task().cancel()
            await asyncio.sleep(0)
        return {"run_directory": str(directory), "trade_count": 0}

    async def commit():
        # Exercise an actual suspension during cleanup, including after cancellation.
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="not been initialized"):
            database.db()
        commits.append(True)

    monkeypatch.setattr(runner, "run_live", run_live)
    call = namespace["execute_run"](asdict(settings), outputs, commit)
    if outcome == "failure":
        with pytest.raises(RuntimeError, match="scripted inference failure"):
            asyncio.run(call)
    elif outcome == "canceled":
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(call)
    else:
        result = asyncio.run(call)
        assert Path(result["run_directory"]).parent == outputs
    assert config.RUNS_DIR == original_root and commits == [True]
    directories = list(outputs.iterdir())
    assert len(directories) == 1
    assert (directories[0] / "run.db").is_file()
    assert (directories[0] / "run.log").read_text() == "run log"


def test_modal_resume_loads_parent_config_from_volume_and_saves_a_child(participant_rig, tmp_path, monkeypatch):
    import shutil

    rig = participant_rig
    settings = replace(rig.settings, model_provider="openai", model_revision=None, model_name="gpt-5.6-luna",
                       openai_api="responses", openai_reasoning_effort="medium")
    source = Path(rig.conn.execute("PRAGMA database_list").fetchone()[2])
    with rig.conn:
        rig.conn.execute("UPDATE run_settings SET config_json=?", (config.as_json(settings),))
    database.close()
    volume = tmp_path / "volume"
    parent = volume / "parent"
    parent.mkdir(parents=True)
    shutil.copyfile(source, parent / "run.db")
    (parent / "run.lock").touch()
    before = (parent / "run.db").read_bytes()
    tree = ast.parse((config.ROOT / "deploy" / "modal_openai.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute_run")
    namespace = {"Path": Path, "__name__": "test_modal_openai"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "modal_openai.py", "exec"), namespace)

    async def run_live(loaded, *, run_id, resume_from, runs_dir):
        assert loaded == settings
        assert Path(resume_from) == parent / "run.db"
        from prediction_markets.storage.runs import create_run_directory
        directory = create_run_directory(run_id, runs_dir=runs_dir)
        (directory / "run.log").write_text("resumed")
        database.fork(directory / "run.db", resume_from, run_settings=loaded)
        database.close()
        return {"run_directory": str(directory), "parent_run_id": "parent"}

    commits = []

    async def commit():
        commits.append(True)

    monkeypatch.setattr(runner, "run_live", run_live)
    result = asyncio.run(namespace["execute_run"](None, volume, commit, resume_from="parent"))
    child = Path(result["run_directory"])
    assert child.parent == volume and child != parent
    assert (child / "run.db").is_file() and commits == [True]
    assert (parent / "run.db").read_bytes() == before
    with pytest.raises(ValueError, match="do not also pass a config"):
        asyncio.run(namespace["execute_run"](asdict(settings), volume, commit, resume_from="parent"))
    with pytest.raises(ValueError, match="run ID"):
        asyncio.run(namespace["execute_run"](None, volume, commit, resume_from="../parent"))
