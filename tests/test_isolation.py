"""The core invariant: no agent-visible surface may carry Kalshi price data.

Two independent checks, because either one alone can be defeated. The import
check stops a module from reaching the API; the brief check stops price data
that is already legitimately in the database from being rendered into a
prompt.
"""

import re
import time
from pathlib import Path

import pytest

from prediction_markets import agents, config, database

PACKAGE = Path(__file__).resolve().parent.parent / "prediction_markets"

# The snapshot columns that are Kalshi price data, plus anything derived.
FORBIDDEN_FIELDS = (
    "yes_bid", "yes_ask", "bid_size", "ask_size", "last", "volume_24h",
    "spread", "mid",
)


def test_only_stream_imports_kalshi_client():
    offenders = []
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name in ("kalshi_client.py", "stream.py"):
            continue
        source = path.read_text()
        if re.search(r"^\s*(from\s+\.\s+import.*\bkalshi_client\b|"
                     r"import\s+.*\bkalshi_client\b|"
                     r"from\s+\.kalshi_client\b)", source, re.MULTILINE):
            offenders.append(path.name)
    assert offenders == [], (
        f"{offenders} import kalshi_client; stream.py must be the sole caller"
    )


def test_tools_module_exposes_no_kalshi_data():
    source = (PACKAGE / "tools.py").read_text()
    assert "kalshi" not in source.lower().replace("kalshi_client", "").replace(
        "no kalshi price data", ""
    ) or "import" not in source


@pytest.fixture
def seeded(tmp_path):
    """A market with a full, distinctive Kalshi snapshot behind it."""
    database.init(tmp_path / "test.db")
    now = int(time.time())
    database.upsert_markets([{
        "ticker": "T1", "event_ticker": "EV-T1",
        "title": "Will the example resolve yes?",
        "rules": "Resolves YES if the example resolves yes.",
        "sector": "Politics", "open_ts": now - 3600,
        "close_ts": now + 48 * 3600, "first_seen": now,
    }])
    # Deliberately unusual values, so finding them in a brief is unambiguous.
    database.write_snapshots([{
        "ticker": "T1", "captured_at": now, "yes_bid": 0.37, "yes_ask": 0.41,
        "bid_size": 1234.0, "ask_size": 4321.0, "last": 0.39,
        "volume_24h": 98765.0, "status": "active", "result": "",
        "settlement_value": None,
    }])
    database.register_agent("a1", config.INITIAL_CASH_CENTS)
    yield now
    database.close()


def test_brief_contains_no_kalshi_price_data(seeded):
    brief = agents.render_brief("a1", "T1", round_id=1)

    for field in FORBIDDEN_FIELDS:
        assert field not in brief.lower(), f"brief names the field {field}"

    # The actual numbers, in every form they could plausibly be rendered.
    for value in ("0.37", "0.41", "37", "41", "39", "0.39",
                  "1234", "4321", "98765"):
        assert value not in brief, f"brief leaks the snapshot value {value}"


def test_brief_does_contain_what_agents_are_supposed_to_see(seeded):
    brief = agents.render_brief("a1", "T1", round_id=1)
    assert "Will the example resolve yes?" in brief
    assert "Resolves YES if the example resolves yes." in brief
    assert "Closes in" in brief
    assert str(config.INITIAL_CASH_CENTS) in brief
