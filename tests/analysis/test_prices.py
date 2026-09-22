"""Historical alignment and run/agent filters, without network calls."""

import json
from pathlib import Path

import pytest

from prediction_markets.integrations import kalshi as kalshi_client
from prediction_markets.analysis import prices as plot_prices


def human(time, price="0.4000", ticker="MKT-01", trade_id="h1"):
    return {"ticker": ticker, "created_time": time, "yes_price_dollars": price, "trade_id": trade_id}


def fill(time, trade_id=1):
    return {"ticker": "MKT-01", "executed_at": time, "trade_id": trade_id,
            "price_cents": 50, "quantity": 10, "buyer": "agent-01", "seller": "agent-02"}


def test_asof_match_never_looks_ahead_and_handles_zero_and_staleness():
    raw = [human("2026-09-20T10:00:01.1Z", "0.9900", trade_id="future"),
           human("2026-09-20T10:00:00.29082Z", "0.0000", trade_id="prior")]
    history = plot_prices.normalize_history(raw, "MKT-01")
    t = plot_prices.timestamp("2026-09-20T10:00:01Z")
    points = plot_prices.pair_prices([fill(t-1), fill(t), fill(t+120)], history, 60)
    assert points[0]["status"] == "no_prior_human_trade"
    assert points[1]["kalshi_trade_id"] == "prior"
    assert points[1]["kalshi_price_cents"] == 0
    assert points[1]["kalshi_age_seconds"] == pytest.approx(.70918, abs=1e-6)
    assert points[2]["status"] == "stale_human_price"
    assert points[2]["kalshi_price_cents"] is None


def test_exact_timestamp_match_and_subcent_prices():
    history = plot_prices.normalize_history([human("2026-09-20T10:00:00Z", ".3719")], "MKT-01")
    point = plot_prices.pair_prices([fill(history[0]["timestamp"])], history, 60)[0]
    assert point["kalshi_price_cents"] == 37.19
    assert point["kalshi_age_seconds"] == 0


@pytest.mark.parametrize("raw", [human("2026-09-20T10:00:00"), human("2026-09-20T10:00:00Z", "NaN"),
                                 human("2026-09-20T10:00:00Z", "1.1"), human("2026-09-20T10:00:00Z", ticker="OTHER")])
def test_malformed_history_is_not_silently_plotted(raw):
    with pytest.raises(ValueError, match="malformed"):
        plot_prices.normalize_history([raw], "MKT-01")


def test_history_crosses_archive_cutoff_and_paginates(monkeypatch):
    calls = []
    cutoff = int(plot_prices.timestamp("2026-09-20T10:00:00Z"))

    def get(path, params, *, base_url):
        calls.append((path, params))
        if path == "/historical/cutoff":
            return {"trades_created_ts": "2026-09-20T10:00:00Z"}
        if path == "/historical/trades":
            return {"trades": [human("2026-09-20T09:59:59Z", trade_id="old")], "cursor": ""}
        if "cursor" not in params:
            return {"trades": [human("2026-09-20T10:00:01Z", trade_id="new")], "cursor": "next"}
        return {"trades": [human("2026-09-20T10:00:00Z", trade_id="boundary")], "cursor": ""}

    monkeypatch.setattr(kalshi_client, "_get", get)
    trades = kalshi_client.get_trade_history("MKT-01", cutoff-10, cutoff+10, base_url="https://example.test")
    assert {t["trade_id"] for t in trades} == {"old", "boundary", "new"}
    assert [p for p, _ in calls] == ["/historical/cutoff", "/historical/trades", "/markets/trades", "/markets/trades"]
    assert calls[1][1]["max_ts"] >= cutoff
    assert calls[2][1]["min_ts"] <= cutoff
    assert calls[3][1]["cursor"] == "next"


def test_cached_history_is_reused_offline_and_request_scoped(tmp_path, monkeypatch):
    calls = []

    def fetch(*args, **kwargs):
        calls.append(args)
        return [human("2026-09-20T10:00:00Z")]

    monkeypatch.setattr(plot_prices, "fetch_trade_history", fetch)
    first = plot_prices.load_history("MKT-01", 1, 2, "https://example.test", tmp_path)
    assert plot_prices.load_history("MKT-01", 1, 2, "https://example.test", tmp_path, offline=True) == first
    assert len(calls) == 1
    with pytest.raises(ValueError, match="no cached history"):
        plot_prices.load_history("MKT-01", 1, 3, "https://example.test", tmp_path, offline=True)


def test_cli_filters_buyer_seller_self_trades_and_empty_markets(participant_rig, tmp_path, monkeypatch):
    rig = participant_rig
    rig.exchange.submit_order("agent-01", "MKT-01", "sell", "limit", 10, 40)
    rig.exchange.submit_order("agent-02", "MKT-01", "buy", "limit", 10, 45)
    rig.exchange.submit_order("agent-02", "MKT-01", "sell", "limit", 2, 50)
    rig.exchange.submit_order("agent-02", "MKT-01", "buy", "limit", 2, 50)
    # The rig has its own pytest temp directory: use SQLite's actual path.
    path = Path(rig.conn.execute("PRAGMA database_list").fetchone()[2])
    monkeypatch.setattr(plot_prices, "fetch_trade_history", lambda ticker, *a, **kw:
                        [human(plot_prices.iso(rig.clock.now-10), ticker=ticker)])
    rendered = []
    monkeypatch.setattr(plot_prices, "render", lambda markets, points, path, title:
                        rendered.append((markets, points, path)))
    before = rig.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    out = tmp_path / "plots"
    common = ["--run-id", str(path), "--out", str(out)]
    assert plot_prices.main(common + ["--agent-id", "agent-01", "--combined"]) == 0
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["selected_fills"] == 1  # agent-01 was the seller, not the taker.
    assert len(meta["markets_without_selected_fills"]) == 9
    assert len(rendered[0][0]) == 10
    assert plot_prices.main(common + ["--agent-id", "agent-02", "--ticker", "MKT-01", "--offline"]) == 0
    assert len((out / "points.jsonl").read_text().splitlines()) == 2
    assert plot_prices.main(common + ["--agent-id", "agent-02", "--exclude-self-trades", "--offline"]) == 0
    assert len((out / "points.jsonl").read_text().splitlines()) == 1
    assert rig.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == before
    assert plot_prices.main(common + ["--ticker", "MISSING"]) == 1
    assert plot_prices.main(common + ["--agent-id", "MISSING"]) == 1


def test_real_render_with_missing_data_and_no_fill_market(tmp_path):
    pytest.importorskip("matplotlib")
    market = {"ticker": "MKT-01", "cohort_index": 1}
    history = plot_prices.normalize_history([human("2026-09-20T10:00:00Z")], "MKT-01")
    t = history[0]["timestamp"]
    points = plot_prices.pair_prices([fill(t-1), fill(t), fill(t+1)], history, 60)
    for suffix in ["png", "svg"]:
        path = tmp_path / f"plot.{suffix}"
        plot_prices.render([market, {"ticker": "EMPTY", "cohort_index": 2}], points, path, "Test")
        assert path.stat().st_size > 1000
