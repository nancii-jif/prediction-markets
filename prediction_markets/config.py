"""Strict configuration for the minimal continuous-market runtime."""

from __future__ import annotations

import json
import re
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
RUNS_DIR = ROOT / "runs"


@dataclass(frozen=True)
class Settings:
    schema_version: int
    kalshi_base_url: str
    market_count: int
    allowed_market_categories: tuple[str, ...]
    volume_24h_floor: int
    uncertainty_price_range_cents: tuple[int, int]
    expected_expiration_min_hours: int
    expected_expiration_max_hours: int
    market_poll_interval_seconds: int
    market_stale_after_seconds: int
    participant_count: int
    initial_cash_cents: int
    model_name: str
    model_revision: str | None
    model_base_url_env: str
    model_api_key_env: str
    max_concurrent_inference: int
    request_timeout_seconds: int
    working_token_budget: int
    max_output_tokens: int
    temperature: float | None
    top_p: float | None
    top_k: int
    sampling_seed: int
    server_context_tokens: int
    private_note_max_chars: int
    trading_timezone: str
    trading_day_start: str
    trading_day_end: str
    order_tool_minutes: int
    read_tool_minutes: int
    news_tool_minutes: int
    hold_for_now_minutes: int
    inference_retry_seconds: int
    tavily_api_key_env: str
    run_duration_seconds: int
    enable_thinking: bool
    model_provider: str = "vllm"
    openai_reasoning_effort: str | None = None
    openai_api: str = "chat_completions"

    def __post_init__(self) -> None:
        # Loading from YAML is not the only construction path used by tests or
        # future callers, so the frozen value object enforces its own contract.
        _validate(asdict(self))
        # YAML lists must not leave mutable members inside frozen settings.
        object.__setattr__(
            self, "allowed_market_categories", tuple(self.allowed_market_categories),
        )
        object.__setattr__(
            self, "uncertainty_price_range_cents", tuple(self.uncertainty_price_range_cents),
        )

    @property
    def agent_ids(self) -> tuple[str, ...]:
        return tuple(
            f"agent-{number:02d}"
            for number in range(1, self.participant_count + 1)
        )

    def seed_for(self, agent_id: str) -> int:
        """A distinct but reproducible sampling seed per participant.

        One run-level seed makes a whole run repeatable; the per-agent offset
        stops eight participants with near-identical briefs from sampling the
        same tokens and collapsing into one behaviour.
        """
        return self.sampling_seed + self.agent_ids.index(agent_id)

    def sampling(self, agent_id: str) -> dict[str, Any]:
        """Only sampling parameters supported by the configured provider."""
        if self.model_provider == "openai":
            return {
                name: value for name, value in (
                    ("temperature", self.temperature), ("top_p", self.top_p),
                ) if value is not None
            }
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed_for(agent_id),
        }

    def snapshot(self) -> dict[str, Any]:
        values = asdict(self)
        values["allowed_market_categories"] = list(self.allowed_market_categories)
        values["uncertainty_price_range_cents"] = list(self.uncertainty_price_range_cents)
        values["agent_ids"] = list(self.agent_ids)
        return values


_SETTINGS_FIELDS = frozenset(Settings.__dataclass_fields__)
_REQUIRED_FIELDS = frozenset(
    name for name, field in Settings.__dataclass_fields__.items()
    if field.default is MISSING
)
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_settings: Settings | None = None
_source: Path | None = None


def _positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _bounded_number(name: str, value: Any, low: float, high: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not low <= value <= high
    ):
        raise ValueError(f"{name} must be a number from {low} to {high}, got {value!r}")


def _validate(values: Mapping[str, Any]) -> None:
    provider = values["model_provider"]
    if provider not in ("vllm", "openai"):
        raise ValueError("model_provider must be vllm or openai")
    if values["openai_api"] not in ("chat_completions", "responses"):
        raise ValueError("openai_api must be chat_completions or responses")
    if provider != "openai" and values["openai_api"] != "chat_completions":
        raise ValueError("openai_api: responses only applies to model_provider: openai")
    effort = values["openai_reasoning_effort"]
    if effort not in (None, "none", "minimal", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("openai_reasoning_effort must be null or a supported reasoning effort")
    if provider == "vllm" and effort is not None:
        raise ValueError("openai_reasoning_effort only applies to model_provider: openai")
    if provider == "openai" and values["enable_thinking"]:
        raise ValueError("enable_thinking is vLLM-only; use openai_reasoning_effort for OpenAI")
    price_range = values["uncertainty_price_range_cents"]
    if (
        not isinstance(price_range, (list, tuple))
        or len(price_range) != 2
        or any(type(bound) is not int or not 0 <= bound <= 100 for bound in price_range)
        or price_range[0] > price_range[1]
    ):
        raise ValueError(
            "uncertainty_price_range_cents must be [lo, hi] integer cents with 0 <= lo <= hi <= 100"
        )

    categories = values["allowed_market_categories"]
    if (
        not isinstance(categories, (list, tuple))
        or not categories
        or any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in categories
        )
        or len(set(categories)) != len(categories)
    ):
        raise ValueError(
            "allowed_market_categories must be a nonempty list of unique category names"
        )

    for name in (
        "schema_version",
        "market_count",
        "volume_24h_floor",
        "expected_expiration_min_hours",
        "expected_expiration_max_hours",
        "market_poll_interval_seconds",
        "market_stale_after_seconds",
        "participant_count",
        "initial_cash_cents",
        "max_concurrent_inference",
        "request_timeout_seconds",
        "working_token_budget",
        "max_output_tokens",
        "server_context_tokens",
        "private_note_max_chars",
        "order_tool_minutes",
        "read_tool_minutes",
        "news_tool_minutes",
        "hold_for_now_minutes",
        "inference_retry_seconds",
        "run_duration_seconds",
    ):
        _positive_integer(name, values[name])

    for name in (
        "kalshi_base_url",
        "model_name",
        "model_base_url_env",
        "model_api_key_env",
        "tavily_api_key_env",
    ):
        if not isinstance(values[name], str) or not values[name].strip():
            raise ValueError(f"{name} must be a non-empty string")

    for name in ("model_base_url_env", "model_api_key_env", "tavily_api_key_env"):
        if _ENV_NAME.fullmatch(values[name]) is None:
            raise ValueError(f"{name} must name an environment variable")

    # Freeze requested sampling parameters. OpenAI may omit temperature/top_p;
    # top_k and per-agent seeds are sent only to vLLM. A seed of 0 is valid.
    for name, upper in (("temperature", 2), ("top_p", 1)):
        if provider == "openai" and values[name] is None:
            continue
        _bounded_number(name, values[name], 0, upper)
    if values["top_p"] == 0:
        raise ValueError("top_p must be greater than 0")
    top_k = values["top_k"]
    if isinstance(top_k, bool) or not isinstance(top_k, int) or (top_k < 1 and top_k != -1):
        raise ValueError("top_k must be a positive integer, or -1 to disable it")
    seed = values["sampling_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("sampling_seed must be a nonnegative integer")

    if not isinstance(values["enable_thinking"], bool):
        raise ValueError("enable_thinking must be true or false")
    if values["schema_version"] != 1:
        raise ValueError("this MVP requires schema_version: 1")
    if provider == "openai" and values["model_revision"] is not None:
        raise ValueError("OpenAI model_revision must be null; put the API model or snapshot ID in model_name")
    if provider == "vllm" and (
        not isinstance(values["model_revision"], str)
        or re.fullmatch(r"[0-9a-f]{40}", values["model_revision"]) is None
    ):
        raise ValueError("model_revision must be a pinned 40-character commit hash")
    if (
        values["expected_expiration_min_hours"]
        > values["expected_expiration_max_hours"]
    ):
        raise ValueError("expected-expiration minimum cannot exceed maximum")
    try:
        ZoneInfo(values["trading_timezone"])
    except (TypeError, ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError("trading_timezone must be an IANA timezone, e.g. America/New_York") from exc
    for name in ("trading_day_start", "trading_day_end"):
        value = values[name]
        if not isinstance(value, str) or re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value) is None:
            raise ValueError(f"{name} must be a quoted HH:MM local time")
    if values["trading_day_start"] >= values["trading_day_end"]:
        raise ValueError("trading_day_start must precede trading_day_end on the same day")
    # Necessary, not sufficient: include the fixed prefix using the provider's
    # token counter and recheck the configured context guard at startup.
    if (
        values["working_token_budget"] + values["max_output_tokens"]
        > values["server_context_tokens"]
    ):
        raise ValueError("working and output token budgets exceed the server context")
    if values["max_concurrent_inference"] > values["participant_count"]:
        raise ValueError("max_concurrent_inference cannot exceed participant_count")


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate
    for resolved in (
        CONFIG_DIR / candidate.name,
        CONFIG_DIR / f"{candidate.name}.yaml",
    ):
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"no config at {candidate} or in {CONFIG_DIR}")


def load(path: str | Path, *, remember: bool = True) -> Settings:
    """Load and validate one complete MVP configuration."""
    global _settings, _source

    resolved = _resolve_path(path)
    values = yaml.safe_load(resolved.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"{resolved.name} must contain a YAML mapping")

    missing = sorted(_REQUIRED_FIELDS - values.keys())
    unknown = sorted(values.keys() - _SETTINGS_FIELDS)
    if missing:
        raise ValueError(f"{resolved.name} is missing required keys: {missing}")
    if unknown:
        raise ValueError(f"{resolved.name} has unrecognized keys: {unknown}")

    loaded = Settings(**values)
    if remember:
        _settings, _source = loaded, resolved
    return loaded


def current() -> Settings:
    if _settings is None:
        raise RuntimeError("configuration has not been loaded")
    return _settings


def is_loaded() -> bool:
    return _settings is not None


def source() -> Path | None:
    return _source


def as_json(settings: Settings | None = None) -> str:
    selected = settings if settings is not None else current()
    return json.dumps(selected.snapshot(), sort_keys=True, separators=(",", ":"))


def create_run_directory(run_id: str) -> Path:
    """Compatibility helper; path ownership lives in storage.runs."""
    from .storage.runs import create_run_directory as create
    return create(run_id)


def load_credentials() -> bool:
    """Load only the root .env; never load archived/legacy provider credentials."""
    from dotenv import load_dotenv

    candidate = ROOT / ".env"
    if not candidate.exists():
        return False
    load_dotenv(candidate)
    return True
