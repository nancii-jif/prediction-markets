"""Run configuration. A loader only — no parameter value is defined here.

Every constant comes from a YAML file in `configs/`, named `YYYY-MM-DD-HHMM`.
The run writes its artifacts to `runs/<same stem>/`, so a config file and the
run it produced always share a name.

Nothing is defaulted: a config must list every required key. A missing key is
an error at load time rather than a silent fallback, so a run can never
quietly differ from the file that claims to describe it.

Reading a constant before `load()` raises, rather than returning a stale or
absent value — see `__getattr__`.
"""

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"

# Every key a config file must provide. Derived values (below) are computed
# from these and must NOT appear in the file.
REQUIRED = (
    "KALSHI_BASE_URL",
    # cadences
    "STREAM_INTERVAL_S", "ROUND_INTERVAL_S", "ARCHIVE_SWEEP_INTERVAL_S", "ROUNDS",
    # live market selection
    "SECTORS", "MARKETS_PER_SECTOR", "EXPIRY_MIN_H", "EXPIRY_MAX_H",
    "RETIRE_GRACE_H", "VOLUME_FLOOR", "UNCERTAIN_LO", "UNCERTAIN_HI",
    # internal exchange
    "INITIAL_CASH_CENTS", "MAX_POSITION", "MAX_LOT", "FEE_CENTS",
    "PRICE_MIN", "PRICE_MAX",
    # agents
    "QUOTE_SIDE", "TEMPERATURE", "FAMILIES", "AGENTS_PER_FAMILY", "MAX_CONCURRENCY",
    "MAX_DAILY_CALLS", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES", "LLM_API_RETRIES",
    "PROMPT_VERSION",
    # live metrics; null disables tracking entirely
    "WANDB_PROJECT",
    # determinism
    "RUN_SEED",
    # free text, for the human
    "DESCRIPTION",
)

# Computed by load(); rejected if a config file tries to set them.
DERIVED = ("AGENTS", "RUN_NAME", "RUN_DIR", "DATA_DIR", "DB_PATH")

_VALUES: dict = {}
_SOURCE: Path | None = None


def __getattr__(name: str):
    """Serve constants out of the loaded config.

    Python calls this only when normal module lookup fails, so functions and
    module globals above resolve as usual and only parameter reads land here.
    """
    if name.startswith("_"):
        raise AttributeError(name)
    if not _VALUES:
        raise RuntimeError(
            f"config.{name} was read before any run config was loaded. "
            f"Call config.load(<path>) first "
            f"(e.g. --config configs/2026-09-02-1417.yaml)."
        )
    try:
        return _VALUES[name]
    except KeyError:
        raise AttributeError(
            f"{name!r} is not in the run config loaded from {_SOURCE}"
        ) from None


def _derive(values: dict, name: str) -> dict:
    """Values computed from the file rather than stated in it."""
    run_dir = RUNS_DIR / name
    return {
        "AGENTS": tuple(
            {
                "agent_id": f"{family}-{i}",
                "model": model,
                "persona": None,
                "temperature": values["TEMPERATURE"],
            }
            for family, model in values["FAMILIES"]
            for i in range(1, values["AGENTS_PER_FAMILY"] + 1)
        ),
        "RUN_NAME": name,
        "RUN_DIR": run_dir,
        # The run directory is the data directory: one run, one folder.
        "DATA_DIR": run_dir,
        "DB_PATH": run_dir / "run.db",
    }


def load_credentials() -> bool:
    """Load API keys from .env into the environment, if one is present.

    Credentials live outside the run config on purpose: a config file is
    copied into the run directory verbatim, and a key must never end up there.
    """
    from dotenv import load_dotenv

    for candidate in (ROOT / ".env", Path(__file__).resolve().parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            return True
    return False


def load(path, name: str | None = None) -> dict:
    """Load a run config. The run name defaults to the file's stem."""
    global _VALUES, _SOURCE

    load_credentials()

    path = Path(path)
    if not path.exists():
        for candidate in (CONFIG_DIR / path.name, CONFIG_DIR / f"{path.name}.yaml"):
            if candidate.exists():
                path = candidate
                break
        else:
            raise FileNotFoundError(f"no config at {path} or in {CONFIG_DIR}")

    values = yaml.safe_load(path.read_text())

    missing = [key for key in REQUIRED if key not in values]
    if missing:
        raise ValueError(f"{path.name} is missing required keys: {missing}")
    stated_derived = [key for key in DERIVED if key in values]
    if stated_derived:
        raise ValueError(
            f"{path.name} sets derived keys {stated_derived}; these are computed "
            f"from the config and the run name, and must not be stated"
        )
    unknown = [k for k in values if k not in REQUIRED and not k.startswith("_")]
    if unknown:
        raise ValueError(
            f"{path.name} has unrecognised keys {unknown} (prefix a key with _ "
            f"to keep it as a note)"
        )

    # `null` means no cap. Checked here because YAML reads a bare `None` as the
    # string "None", which would otherwise survive load and only surface as a
    # TypeError deep in the first round's validation.
    cap = values["MAX_POSITION"]
    if cap is not None and not isinstance(cap, int):
        raise ValueError(
            f"{path.name}: MAX_POSITION must be an integer or null (for no "
            f"cap), got {cap!r}"
        )

    values.update(_derive(values, name or path.stem))
    _VALUES, _SOURCE = values, path
    return values


def is_loaded() -> bool:
    return bool(_VALUES)


def source() -> Path | None:
    return _SOURCE


def expected_calls() -> int:
    """Upper bound on LLM calls: every agent quotes every live market, every round.

    A bounded run (ROUNDS set) is costed at what it will actually spend. Only
    an open-ended run is extrapolated to a day, since that is the only case
    where a daily ceiling is the meaningful limit.
    """
    live = len(_VALUES["SECTORS"]) * _VALUES["MARKETS_PER_SECTOR"]
    rounds = _VALUES["ROUNDS"]
    if rounds is None:
        rounds = 86_400 // _VALUES["ROUND_INTERVAL_S"]
    return len(_VALUES["AGENTS"]) * live * rounds


def snapshot() -> dict:
    """The resolved configuration as plain data, for the run_config row and the
    frozen copy in the run directory."""
    values = dict(_VALUES)
    for key in ("RUN_DIR", "DATA_DIR", "DB_PATH"):
        values[key] = str(values[key])
    values["AGENTS"] = [dict(a) for a in values["AGENTS"]]
    values["SECTORS"] = list(values["SECTORS"])
    values["FAMILIES"] = [list(f) for f in values["FAMILIES"]]
    values["_source"] = str(_SOURCE)
    values["_expected_calls"] = expected_calls()
    return values


def as_json() -> str:
    return json.dumps(snapshot(), indent=2, sort_keys=True, default=str)


def freeze() -> Path:
    """Write the resolved config into the run directory, beside its artifacts."""
    run_dir = _VALUES["RUN_DIR"]
    run_dir.mkdir(parents=True, exist_ok=True)
    frozen = run_dir / "config.yaml"
    frozen.write_text(yaml.safe_dump(snapshot(), sort_keys=True, default_flow_style=False))
    return frozen
