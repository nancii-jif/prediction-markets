"""Public request construction, pagination, and retained network retries."""

import io
import json
import urllib.error
import urllib.parse

import pytest

from prediction_markets.integrations import kalshi as kalshi_client


def test_discovery_paginates_once_without_close_filters(monkeypatch):
    calls = []

    def get(path, params, *, base_url):
        calls.append((path, params, base_url))
        if path == "/series":
            return {"series": [{"ticker": "SERIES", "category": "Politics"}]}
        ticker = "B" if "cursor" in params else "A"
        return {"events": [{
            "event_ticker": f"EVENT-{ticker}", "series_ticker": "SERIES",
            "category": "World",  # Stale event label must not override series.
            "markets": [{"ticker": ticker, "event_ticker": f"EVENT-{ticker}"}],
        }], "cursor": "" if ticker == "B" else "next"}

    monkeypatch.setattr(kalshi_client, "_get", get)
    assert kalshi_client.get_market_candidates(base_url="https://example.test/v2") == [
        {"ticker": "A", "event_ticker": "EVENT-A", "categories": ["Politics"],
         "_raw_market": {"ticker": "A", "event_ticker": "EVENT-A"}},
        {"ticker": "B", "event_ticker": "EVENT-B", "categories": ["Politics"],
         "_raw_market": {"ticker": "B", "event_ticker": "EVENT-B"}},
    ]
    assert calls == [
        ("/series", {}, "https://example.test/v2"),
        ("/events", {"status": "open", "limit": 200, "with_nested_markets": "true"}, "https://example.test/v2"),
        ("/events", {"status": "open", "limit": 200, "with_nested_markets": "true", "cursor": "next"}, "https://example.test/v2"),
    ]


@pytest.mark.parametrize(("series", "expected"), [
    ([], []),
    ([{"ticker": "OTHER", "category": "Politics"}], []),
    ([{"ticker": "SERIES"}], [None]),
    ([{"ticker": "SERIES", "category": "Sports"}], ["Sports"]),
    ([{"ticker": "SERIES", "category": "Sports", "categories": ["Mentions"]}], ["Sports", "Mentions"]),
    ([{"ticker": "SERIES", "category": "Financials", "categories": ["Crypto"]}], ["Financials", "Crypto"]),
    ([{"ticker": "SERIES", "category": "Politics", "categories": "Politics"}], []),
    ([{"ticker": "SERIES", "category": "Politics", "categories": None}], ["Politics"]),
])
def test_series_metadata_is_authoritative_without_event_or_market_fallback(monkeypatch, series, expected):
    def get(path, params, *, base_url):
        if path == "/series":
            return {"series": series}
        return {"events": [{
            "event_ticker": "E", "series_ticker": "SERIES", "category": "Politics",
            "markets": [{"ticker": "M", "event_ticker": "E", "categories": ["Politics"]}],
        }]}

    monkeypatch.setattr(kalshi_client, "_get", get)
    assert kalshi_client.get_market_candidates(base_url="https://example.test")[0]["categories"] == expected


@pytest.mark.parametrize("event", [
    None, {},
    {"event_ticker": "E", "series_ticker": "S", "markets": None},
    {"event_ticker": "E", "series_ticker": "S", "markets": [None]},
    {"event_ticker": "E", "series_ticker": "S", "markets": [{"event_ticker": "OTHER"}]},
])
def test_malformed_event_join_fails_startup(monkeypatch, event):
    monkeypatch.setattr(kalshi_client, "_get", lambda path, *args, **kwargs:
                        {"series": []} if path == "/series" else {"events": [event]})
    with pytest.raises(kalshi_client.KalshiError, match="malformed"):
        kalshi_client.get_market_candidates(base_url="https://example.test")


@pytest.mark.parametrize("series", [
    [None], [{}], [{"ticker": []}],
    [{"ticker": "S", "category": "Politics"}, {"ticker": "S", "category": "Sports"}],
])
def test_malformed_or_duplicate_series_fails_startup(monkeypatch, series):
    monkeypatch.setattr(kalshi_client, "_get", lambda *args, **kwargs: {"series": series})
    with pytest.raises(kalshi_client.KalshiError, match="malformed|duplicate"):
        kalshi_client.get_market_candidates(base_url="https://example.test")


def test_repeating_cursor_fails_instead_of_selecting_partial_universe(monkeypatch):
    monkeypatch.setattr(
        kalshi_client, "_get",
        lambda *args, **kwargs: {"markets": [{"ticker": "A"}], "cursor": "stuck"},
    )
    with pytest.raises(kalshi_client.KalshiError, match="cursor repeated"):
        kalshi_client.get_markets_by_tickers(["A"], base_url="https://example.test")


@pytest.mark.parametrize("payload", [None, [], {}, {"markets": {}}, {"markets": [], "cursor": ["bad"]}])
def test_malformed_response_is_a_reportable_fetch_failure(monkeypatch, payload):
    monkeypatch.setattr(kalshi_client, "_get", lambda *args, **kwargs: payload)
    with pytest.raises(kalshi_client.KalshiError, match="malformed"):
        kalshi_client.get_markets_by_tickers(["A"], base_url="https://example.test")


def test_targeted_poll_has_no_status_filter_and_empty_cohort_makes_no_request(monkeypatch):
    calls = []

    def get(path, params, *, base_url):
        calls.append((path, params, base_url))
        return {"markets": [{"ticker": ticker} for ticker in params["tickers"].split(",")]}

    monkeypatch.setattr(kalshi_client, "_get", get)
    assert kalshi_client.get_markets_by_tickers([], base_url="https://example.test") == []
    assert calls == []
    assert kalshi_client.get_markets_by_tickers(["A", "B"], base_url="https://example.test") == [
        {"ticker": "A"}, {"ticker": "B"},
    ]
    assert calls == [("/markets", {"tickers": "A,B", "limit": 1000}, "https://example.test")]


def test_request_uses_passed_base_url_and_retries_transient_error(monkeypatch):
    calls = []
    sleeps = []

    def open_request(request, *, timeout):
        calls.append((request, timeout))
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "slow down", {"Retry-After": "2"}, None)
        return io.BytesIO(json.dumps({"markets": []}).encode())

    monkeypatch.setattr(kalshi_client.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(kalshi_client.time, "sleep", sleeps.append)
    assert kalshi_client.get_markets_by_tickers(["A"], base_url="https://example.test/v2/") == []
    assert sleeps == [2.0]
    assert len(calls) == 2
    request, timeout = calls[0]
    parts = urllib.parse.urlsplit(request.full_url)
    assert parts.netloc == "example.test"
    assert parts.path == "/v2/markets"
    assert request.get_method() == "GET"
    assert not request.has_header("Authorization")
    assert timeout == 30
    assert urllib.parse.parse_qs(parts.query) == {
        "tickers": ["A"], "limit": ["1000"],
    }


def test_permanent_http_error_is_reported_without_retry(monkeypatch):
    def open_request(request, *, timeout):
        raise urllib.error.HTTPError(request.full_url, 400, "bad request", {}, None)

    monkeypatch.setattr(kalshi_client.urllib.request, "urlopen", open_request)
    monkeypatch.setattr(kalshi_client.time, "sleep", lambda _: pytest.fail("unexpected retry"))
    with pytest.raises(kalshi_client.KalshiError, match="HTTP 400"):
        kalshi_client.get_market_candidates(base_url="https://example.test")
