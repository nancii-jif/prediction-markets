"""Step-one contracts for the minimal runtime configuration and database.

These tests intentionally stop at durable run setup.  Accounting mutations,
order matching, market polling, and agent loops belong to later steps.
"""

from dataclasses import FrozenInstanceError, asdict, replace
import json
import sqlite3

import pytest
import yaml

from prediction_markets import config
from prediction_markets.storage import database as database


def _settings():
    return config.load(config.CONFIG_DIR / "mvp.yaml")


def _market(index: int, *, event_ticker: str | None = None) -> dict:
    """A minimal fixed-cohort row, using Unix seconds for UTC timestamps."""
    return {
        "ticker": f"MKT-{index:02d}",
        "event_ticker": (
            event_ticker if event_ticker is not None else f"EVENT-{index:02d}"
        ),
        "title": f"Market {index}",
        "rules": f"Market {index} resolves according to its source.",
        "market_type": "binary",
        "status": "active",
        "close_time": 2_000_000_000 + index,
        "expected_expiration_time": 2_001_000_000 + index,
        "latest_expiration_time": 2_002_000_000 + index,
        "last_successful_poll": 1_999_000_000,
        "payout_cents": None,
    }


@pytest.fixture
def settings():
    return _settings()


@pytest.fixture
def fresh_database(tmp_path, settings):
    path = tmp_path / "run.db"
    conn = database.init(path, run_settings=settings)
    try:
        yield conn, path
    finally:
        database.close()


def test_mvp_config_loads_as_frozen_settings():
    settings = _settings()

    assert isinstance(settings, config.Settings)
    assert settings.market_count == 10
    assert settings.allowed_market_categories == (
        "Elections", "Politics", "Entertainment", "Commodities", "Climate and Weather",
        "Economics", "Mentions", "Financials", "Science and Technology",
    )
    assert settings.volume_24h_floor == 500
    assert settings.uncertainty_price_range_cents == (10, 90)
    inputs = yaml.safe_load((config.CONFIG_DIR / "mvp.yaml").read_text())
    assert settings.expected_expiration_min_hours == inputs["expected_expiration_min_hours"]
    assert settings.expected_expiration_max_hours == inputs["expected_expiration_max_hours"]
    assert settings.market_poll_interval_seconds == 300
    assert settings.market_stale_after_seconds == 600
    assert settings.participant_count == 8
    assert settings.initial_cash_cents == 100_000
    assert settings.trading_timezone == "America/New_York"
    assert (settings.trading_day_start, settings.trading_day_end) == ("08:00", "16:00")
    assert (settings.order_tool_minutes, settings.read_tool_minutes, settings.news_tool_minutes) == (2, 5, 10)
    assert settings.hold_for_now_minutes == 60
    assert settings.agent_ids == tuple(f"agent-{i:02d}" for i in range(1, 9))

    with pytest.raises(FrozenInstanceError):
        settings.market_count = 9


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"market_count": True},
        {"allowed_market_categories": []},
        {"allowed_market_categories": "Politics"},
        {"allowed_market_categories": ["Politics", "Politics"]},
        {"allowed_market_categories": [None]},
        {"allowed_market_categories": [["Politics"]]},
        {"allowed_market_categories": [" Politics "]},
        {"expected_expiration_min_hours": 0},
        {"expected_expiration_max_hours": True},
        {"hold_for_now_minutes": 0},
        {"hold_for_now_minutes": True},
        {"hold_for_now_minutes": 1.5},
        {"order_tool_minutes": 0},
        {"read_tool_minutes": -1},
        {"news_tool_minutes": False},
        {"inference_retry_seconds": 0},
        {"trading_timezone": "american/new york"},
        {"trading_day_start": "8:00"},
        {"trading_day_end": "25:00"},
        {"trading_day_end": "08:00"},
        {"working_token_budget": 19_457},
        {"max_concurrent_inference": 9},
        {"model_api_key_env": "not an env name"},
        {"model_revision": "main"},
        {"model_revision": None},
    ],
)
def test_settings_validate_every_construction_path(settings, changes):
    with pytest.raises(ValueError):
        replace(settings, **changes)


def test_expiration_minimum_cannot_exceed_the_maximum(settings):
    # Derived, so tuning the horizon in mvp.yaml never invalidates this case.
    with pytest.raises(ValueError, match="minimum cannot exceed maximum"):
        replace(settings, expected_expiration_min_hours=settings.expected_expiration_max_hours + 1)


def test_category_list_is_copied_and_frozen_in_settings(settings):
    categories = ["Politics"]
    frozen = replace(settings, allowed_market_categories=categories)
    categories.append("Sports")
    assert frozen.allowed_market_categories == ("Politics",)
    assert json.loads(config.as_json(frozen))["allowed_market_categories"] == ["Politics"]


@pytest.mark.parametrize("bounds", [
    None, [], [10], [10, 90, 95], "10,90", {"lo": 10, "hi": 90},
    [-1, 90], [10, 101], [90, 10], [True, 90], [10, False],
    [10.0, 90], [10, "90"], [float("nan"), 90], [10, float("inf")],
])
def test_invalid_uncertainty_ranges_are_rejected_in_settings_and_yaml(settings, tmp_path, bounds):
    with pytest.raises(ValueError, match="uncertainty_price_range_cents"):
        replace(settings, uncertainty_price_range_cents=bounds)
    values = asdict(settings)
    values["uncertainty_price_range_cents"] = bounds
    path = tmp_path / "bad-range.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="uncertainty_price_range_cents"):
        config.load(path)


@pytest.mark.parametrize("bounds", [[0, 100], [10, 90], [50, 50]])
def test_uncertainty_range_is_frozen_and_serializes_as_a_yaml_pair(settings, tmp_path, bounds):
    original = bounds.copy()
    frozen = replace(settings, uncertainty_price_range_cents=bounds)
    bounds[0] = 99
    assert frozen.uncertainty_price_range_cents == tuple(original)
    assert json.loads(config.as_json(frozen))["uncertainty_price_range_cents"] == original
    values = frozen.snapshot()
    values.pop("agent_ids")
    path = tmp_path / "range.yaml"
    path.write_text(yaml.safe_dump(values))
    assert config.load(path).uncertainty_price_range_cents == tuple(original)


def test_actual_run_config_loads_with_uncertainty_bounds():
    # The rest of the suite uses a fixed test configuration; validate the
    # user-editable run file too, including cross-field constraints.
    settings = config.load(config.ROOT / "configs" / "mvp.yaml")
    assert 0 <= settings.uncertainty_price_range_cents[0] <= settings.uncertainty_price_range_cents[1] <= 100


@pytest.mark.parametrize("bad_key", ["market_count", "unexpected_option"])
def test_config_rejects_missing_and_unknown_keys(tmp_path, settings, bad_key):
    values = asdict(settings)
    if bad_key == "market_count":
        values.pop(bad_key)
        message = "missing required keys"
    else:
        values[bad_key] = 1
        message = "unrecognized keys"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(values))

    with pytest.raises(ValueError, match=message):
        config.load(path)


def test_config_rejects_old_day_keys_instead_of_reinterpreting_units(tmp_path, settings):
    values = asdict(settings)
    for bound in ("min", "max"):
        values[f"expected_expiration_{bound}_days"] = values.pop(f"expected_expiration_{bound}_hours")
    path = tmp_path / "old-days.yaml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="missing required keys.*hours"):
        config.load(path)


def test_run_directory_is_claimed_once_without_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    run_dir = config.create_run_directory("trial-001")
    marker = run_dir / "keep.txt"
    marker.write_text("preserve me")

    with pytest.raises(FileExistsError, match="overwrite or resume"):
        config.create_run_directory("trial-001")
    with pytest.raises(ValueError, match="safe path segment"):
        config.create_run_directory("../escape")

    assert marker.read_text() == "preserve me"


def test_credentials_load_only_root_dotenv_and_preserve_exported_values(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setenv("MODEL_API_KEY", "already-exported")
    # Ensure pytest restores the original environment even when dotenv adds it.
    monkeypatch.setenv("MODEL_BASE_URL", "temporary")
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    legacy_dir = tmp_path / "archive" / "auction" / "prediction_markets"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / ".env").write_text("MODEL_API_KEY=do-not-load\n")
    assert config.load_credentials() is False
    (tmp_path / ".env").write_text("MODEL_API_KEY=from-file\nMODEL_BASE_URL=https://model.example/v1\n")
    assert config.load_credentials() is True
    import os
    assert os.environ["MODEL_API_KEY"] == "already-exported"
    assert os.environ["MODEL_BASE_URL"] == "https://model.example/v1"


def test_fresh_database_saves_settings_and_initializes_eight_accounts(
    fresh_database, settings, monkeypatch
):
    conn, _ = fresh_database

    accounts = conn.execute(
        "SELECT agent_id, balance_cents FROM accounts ORDER BY agent_id"
    ).fetchall()
    assert [(row["agent_id"], row["balance_cents"]) for row in accounts] == [
        (agent_id, 100_000) for agent_id in settings.agent_ids
    ]

    states = conn.execute(
        "SELECT agent_id, private_note, activity_day_started_at, "
        "shared_tool_calls, scheduled_wake_at "
        "FROM agent_state ORDER BY agent_id"
    ).fetchall()
    assert [
        (
            row["agent_id"],
            row["private_note"],
            row["activity_day_started_at"],
            row["shared_tool_calls"],
            row["scheduled_wake_at"],
        )
        for row in states
    ] == [(agent_id, "", None, 0, None) for agent_id in settings.agent_ids]

    saved = conn.execute(
        "SELECT schema_version, config_json FROM run_settings"
    ).fetchone()
    assert saved is not None
    assert saved["schema_version"] == 1
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 1

    secret = "sentinel-secret-must-not-be-persisted"
    monkeypatch.setenv(settings.model_api_key_env, secret)
    serialized = config.as_json(settings)
    assert secret not in serialized
    assert settings.model_api_key_env in serialized

    snapshot = json.loads(saved["config_json"])
    assert snapshot["market_count"] == 10
    assert snapshot["expected_expiration_min_hours"] == settings.expected_expiration_min_hours
    assert snapshot["expected_expiration_max_hours"] == settings.expected_expiration_max_hours
    assert "expected_expiration_min_days" not in snapshot
    assert "expected_expiration_max_days" not in snapshot
    assert snapshot["allowed_market_categories"] == list(settings.allowed_market_categories)
    assert snapshot["participant_count"] == 8
    assert snapshot["initial_cash_cents"] == 100_000
    assert snapshot["model_api_key_env"] == settings.model_api_key_env


def test_schema_contains_only_the_mvp_runtime_entities(fresh_database):
    conn, _ = fresh_database
    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }

    assert tables == {
        "run_settings",
        "markets",
        "accounts",
        "agent_state",
        "agent_market_calls",
        "positions",
        "orders",
        "trades",
        "market_settlements",
        "transcript_entries",
        "book_events",
        "market_api_responses",
        "run_segments",
        "agent_checkpoints",
    }
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_account_constraints_reject_duplicate_agents_and_negative_cash(
    fresh_database, settings
):
    conn, _ = fresh_database

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO accounts (agent_id, balance_cents) VALUES (?, ?)",
            (settings.agent_ids[0], 100_000),
        )

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE accounts SET balance_cents = -1 WHERE agent_id = ?",
            (settings.agent_ids[0],),
        )


def test_init_refuses_to_overwrite_or_resume_an_existing_database(
    tmp_path, settings
):
    path = tmp_path / "run.db"
    conn = database.init(path, run_settings=settings)
    conn.execute(
        "UPDATE accounts SET balance_cents = 12345 WHERE agent_id = ?",
        (settings.agent_ids[0],),
    )
    conn.commit()
    database.close()
    before = path.read_bytes()

    with pytest.raises(FileExistsError):
        database.init(path, run_settings=settings)

    assert path.read_bytes() == before
    with sqlite3.connect(path) as check:
        assert check.execute(
            "SELECT balance_cents FROM accounts WHERE agent_id = ?",
            (settings.agent_ids[0],),
        ).fetchone()[0] == 12_345


@pytest.mark.parametrize(
    "rows",
    [
        [_market(i) for i in range(1, 10)],
        [_market(i, event_ticker="SAME-EVENT") for i in range(1, 11)],
        [
            *[_market(i) for i in range(1, 10)],
            _market(1, event_ticker="EVENT-10"),
        ],
        [
            *[_market(i) for i in range(1, 10)],
            {**_market(10), "ticker": ""},
        ],
        [
            *[_market(i) for i in range(1, 10)],
            _market(10, event_ticker=""),
        ],
    ],
    ids=(
        "nine-markets",
        "duplicate-events",
        "duplicate-tickers",
        "empty-ticker",
        "empty-event",
    ),
)
def test_invalid_cohort_is_rejected_atomically(fresh_database, rows):
    conn, _ = fresh_database

    with pytest.raises(ValueError):
        database.store_market_cohort(rows)

    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 0


def test_fixed_ten_market_cohort_is_stored_once(fresh_database):
    conn, _ = fresh_database
    cohort = [_market(i) for i in range(1, 11)]

    assert database.store_market_cohort(cohort) == 10
    saved = conn.execute(
        "SELECT ticker, event_ticker FROM markets ORDER BY cohort_index"
    ).fetchall()
    assert [(row["ticker"], row["event_ticker"]) for row in saved] == [
        (row["ticker"], row["event_ticker"]) for row in cohort
    ]

    with pytest.raises(RuntimeError, match="cohort"):
        database.store_market_cohort([_market(i + 20) for i in range(1, 11)])

    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 10


def test_initial_cohort_requires_at_least_one_expiration(fresh_database):
    conn, _ = fresh_database
    cohort = [_market(i) for i in range(1, 11)]
    cohort[0].update(expected_expiration_time=None, latest_expiration_time=None)
    with pytest.raises(ValueError, match="requires an expected or latest expiration"):
        database.store_market_cohort(cohort)
    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 0


def test_sql_failure_rolls_back_the_entire_cohort(fresh_database):
    conn, _ = fresh_database
    cohort = [_market(i) for i in range(1, 11)]
    cohort[5]["status"] = "not-a-kalshi-status"

    with pytest.raises(sqlite3.IntegrityError):
        database.store_market_cohort(cohort)

    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 0


def test_schema_enforces_foreign_keys(fresh_database, settings):
    conn, _ = fresh_database

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO positions (agent_id, ticker, signed_qty) "
            "VALUES (?, ?, ?)",
            (settings.agent_ids[0], "UNKNOWN-MARKET", 1),
        )


@pytest.mark.parametrize(
    "price_cents, filled_quantity, remaining_quantity, status",
    [
        (100, 0, 1, "open"),
        (50, 0, 0, "open"),
        (50, 2, 0, "filled"),
    ],
)
def test_schema_rejects_invalid_order_state(
    fresh_database,
    settings,
    price_cents,
    filled_quantity,
    remaining_quantity,
    status,
):
    conn, _ = fresh_database
    database.store_market_cohort([_market(i) for i in range(1, 11)])

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO orders ("
            "agent_id, ticker, action, order_type, price_cents, "
            "original_quantity, filled_quantity, remaining_quantity, status, "
            "created_at) VALUES (?, ?, 'buy', 'limit', ?, 1, ?, ?, ?, ?)",
            (
                settings.agent_ids[0],
                "MKT-01",
                price_cents,
                filled_quantity,
                remaining_quantity,
                status,
                2_000_000_000,
            ),
        )


def test_schema_requires_json_transcript_content(fresh_database, settings):
    conn, _ = fresh_database

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO transcript_entries ("
            "agent_id, sequence_number, entry_type, content_json, created_at"
            ") VALUES (?, 1, 'model_response', ?, ?)",
            (settings.agent_ids[0], "not-json", 2_000_000_000),
        )


def test_settlement_marker_is_unique_per_market(fresh_database):
    conn, _ = fresh_database
    database.store_market_cohort([_market(i) for i in range(1, 11)])

    conn.execute(
        "INSERT INTO market_settlements (ticker, payout_cents, settled_at) "
        "VALUES (?, ?, ?)",
        ("MKT-01", 100, 2_100_000_000),
    )
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO market_settlements (ticker, payout_cents, settled_at) "
            "VALUES (?, ?, ?)",
            ("MKT-01", 0, 2_100_000_001),
        )


def test_sampling_is_pinned_in_configuration_not_left_to_the_server(settings):
    # Track the configured values rather than pinning a tuning choice.
    inputs = yaml.safe_load((config.CONFIG_DIR / "mvp.yaml").read_text())
    assert settings.temperature == inputs["temperature"]
    assert settings.top_p == inputs["top_p"]
    assert settings.top_k == inputs["top_k"]
    assert settings.sampling_seed >= 0
    assert settings.sampling("agent-03") == {
        "temperature": settings.temperature, "top_p": settings.top_p,
        "top_k": settings.top_k, "seed": settings.sampling_seed + 2,
    }
    # Frozen settings record what a run sampled at, so it can be reproduced.
    saved = json.loads(config.as_json(settings))
    assert saved["temperature"] == settings.temperature and saved["sampling_seed"] == settings.sampling_seed


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": -0.1}, {"temperature": 2.5}, {"temperature": "hot"},
        {"temperature": True}, {"top_p": 0}, {"top_p": 1.5}, {"top_k": 0},
        {"top_k": -2}, {"top_k": 1.5}, {"sampling_seed": -1}, {"sampling_seed": 1.5},
    ],
)
def test_invalid_sampling_parameters_are_rejected(settings, changes):
    with pytest.raises(ValueError):
        replace(settings, **changes)
