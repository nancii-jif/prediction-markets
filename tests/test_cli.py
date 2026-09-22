"""Unified commands and old imports reach the same implementations."""

import importlib

import pytest

from prediction_markets import cli


@pytest.mark.parametrize("command,module,arguments,expected", [
    ("run", "runtime.runner", ["--config", "configs/openai.yaml"], ["--config", "configs/openai.yaml"]),
    ("resume", "runtime.runner", ["parent"], ["--resume-from", "parent"]),
    ("trace", "analysis.trace", ["parent", "--agent", "agent-01"], ["parent", "--agent", "agent-01"]),
    ("export", "analysis.export", ["parent"], ["parent"]),
    ("plot", "analysis.prices", ["parent", "--combined"], ["parent", "--combined"]),
    ("download", "storage.download", ["parent"], ["parent"]),
])
def test_commands_dispatch_without_changing_arguments(monkeypatch, command, module, arguments, expected):
    implementation = importlib.import_module("prediction_markets." + module)
    calls = []
    monkeypatch.setattr(implementation, "main", lambda args: calls.append(args) or 0)
    assert cli.main([command, *arguments]) == 0
    assert calls == [expected]


@pytest.mark.parametrize("old,new", [
    ("runner", "runtime.runner"), ("trace", "analysis.trace"),
    ("export_trace", "analysis.export"), ("plot_prices", "analysis.prices"),
    ("ledger", "exchange.bookkeeping"), ("stream", "markets.update"),
    ("news", "integrations.tavily"), ("database", "storage.database"),
])
def test_legacy_imports_share_the_canonical_module(old, new):
    assert importlib.import_module("prediction_markets." + old) is importlib.import_module("prediction_markets." + new)
