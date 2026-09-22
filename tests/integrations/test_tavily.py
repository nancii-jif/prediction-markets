"""Real Tavily SDK with offline HTTP, elapsed costs, and activity tracking."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
import requests

from prediction_markets import config
from prediction_markets.integrations import tavily as news
from prediction_markets.agents.tools import ParticipantTools


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "private-news-test-key")
    state = SimpleNamespace(
        calls=[], clients=[], status=200, effect=None,
        body={"results": [{
            "title": "News headline", "url": "https://example.com/news",
            "content": "Source evidence", "score": 0.9,
            "published_date": "2026-09-17", "raw_content": "unneeded full article",
        }], "answer": "unneeded synthesis"},
    )

    def record(client, url, payload, timeout):
        assert str(url).endswith("/search")
        assert client.headers["Authorization"] == "Bearer private-news-test-key"
        assert "private-news-test-key" not in payload
        assert timeout == news.SEARCH_TIMEOUT_SECONDS
        state.calls.append(json.loads(payload))
        state.clients.append(client)

    async def apost(client, url, *, content, timeout, **kwargs):
        record(client, url, content, timeout)
        if state.effect is not None:
            await state.effect()
        return httpx.Response(state.status, json=state.body,
                              request=httpx.Request("POST", "https://api.tavily.com/search"))

    def post(client, url, *, data, timeout, **kwargs):
        record(client, url, data, timeout)
        response = requests.Response()
        response.status_code = state.status
        response._content = json.dumps(state.body).encode()
        return response

    monkeypatch.setattr(httpx.AsyncClient, "post", apost)
    monkeypatch.setattr(requests.Session, "post", post)
    return state


@pytest.mark.parametrize("asynchronous", [False, True])
def test_model_independent_search_uses_real_sdk_and_returns_source_evidence(backend, asynchronous):
    result = (asyncio.run(news.asearch_news("  central bank news  ")) if asynchronous
              else news.search_news("  central bank news  "))
    assert result == {"ok": True, "query": "central bank news", "results": [{
        "title": "News headline", "url": "https://example.com/news", "content": "Source evidence",
        "score": 0.9, "published_date": "2026-09-17",
    }]}
    assert backend.calls == [{
        "query": "central bank news", "search_depth": "advanced", "topic": "news",
        "max_results": 5, "include_answer": False, "include_raw_content": False,
        "include_images": False,
        "exclude_domains": ["kalshi.com", "polymarket.us", "polymarket.com"],
    }]
    if asynchronous:
        assert backend.clients[0].is_closed


@pytest.mark.parametrize("asynchronous", [False, True])
def test_excluded_domains_and_subdomains_never_reach_agents_even_if_tavily_returns_them(backend, asynchronous):
    blocked = [
        "https://kalshi.com/markets/example", "https://news.kalshi.com/p/example",
        "https://API.ELECTIONS.KALSHI.COM.:443/trade-api/v2/markets",
        "https://polymarket.us/event/example", "https://www.polymarket.us/example",
        "https://polymarket.com/event/example", "https://news.polymarket.com/example",
    ]
    allowed = [
        {"title": "Economic release", "url": "https://www.bls.gov/news.release/empsit.htm",
         "content": "Official economic data."},
        {"title": "Independent site", "url": "https://notkalshi.com/news"},
        {"title": "Different domain", "url": "https://kalshi.com.example.org/news"},
    ]
    backend.body = {
        "results": [{"url": url, "content": "blocked evidence"} for url in blocked] + allowed,
        "answer": "blocked evidence", "exclude_domains": blocked,
    }
    # Agent-directed site searches still use the same backend exclusions.
    query = "site:kalshi.com latest forecast"
    result = asyncio.run(news.asearch_news(query)) if asynchronous else news.search_news(query)
    assert result == {"ok": True, "query": query, "results": allowed}
    assert "blocked evidence" not in json.dumps(result)
    assert "exclude_domains" not in result
    assert backend.calls[0]["query"] == query
    assert len(backend.calls) == 1  # No hidden retries to refill removed results.


@pytest.mark.parametrize("asynchronous", [False, True])
def test_only_blocked_or_unverifiable_sources_returns_normal_empty_results(backend, asynchronous):
    backend.body = {"results": [
        {"url": "https://news.kalshi.com/article"}, {"url": "https://polymarket.us/"},
        {}, {"url": None}, {"url": 123}, {"url": "not a URL"},
        {"url": "https://[invalid"}, {"url": "javascript:alert(1)"},
    ]}
    result = asyncio.run(news.asearch_news("forecast")) if asynchronous else news.search_news("forecast")
    assert result == {"ok": True, "query": "forecast", "results": []}
    assert len(backend.calls) == 1


@pytest.mark.parametrize("query", ["", "  ", None, 42, True, [], {}, "x" * 401])
def test_invalid_queries_never_reach_tavily(backend, query):
    assert news.search_news(query)["error"]["code"] == "invalid_arguments"
    assert asyncio.run(news.asearch_news(query))["error"]["code"] == "invalid_arguments"
    assert backend.calls == []


def test_missing_key_and_configurable_environment_name(backend, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY")
    assert news.search_news("news")["error"]["code"] == "news_not_configured"
    assert asyncio.run(news.asearch_news("news"))["error"]["code"] == "news_not_configured"
    assert backend.calls == []
    monkeypatch.setenv("CUSTOM_NEWS_KEY", "private-news-test-key")
    assert news.search_news("news", api_key_env="CUSTOM_NEWS_KEY")["ok"]


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_provider_errors_are_sanitized_and_never_retried(backend, status):
    backend.status = status
    backend.body = {"detail": {"error": "private-news-test-key"}}
    results = [news.search_news("news"), asyncio.run(news.asearch_news("news"))]
    assert all(result["error"]["code"] == "news_search_failed" for result in results)
    assert "private-news-test-key" not in json.dumps(results)
    assert len(backend.calls) == 2
    assert backend.clients[-1].is_closed


@pytest.mark.parametrize("body", [{"results": []}, {"results": None}, {"results": ["bad"]}])
def test_empty_or_malformed_results(backend, body):
    backend.body = body
    result = asyncio.run(news.asearch_news("news"))
    assert result["ok"] is (body == {"results": []})


def test_timeout_closes_the_client_without_retry(backend, monkeypatch):
    monkeypatch.setattr(news, "SEARCH_TIMEOUT_SECONDS", 0.01)

    async def hang():
        await asyncio.Future()

    backend.effect = hang
    result = asyncio.run(news.asearch_news("news"))
    assert result["error"]["code"] == "news_search_failed"
    assert len(backend.calls) == 1 and backend.clients[0].is_closed


@pytest.mark.parametrize("arguments", [
    {}, {"query": "news", "api_key_env": "OTHER_KEY"},
    {"query": "news", "agent_id": "agent-02"}, {"query": "news", "max_calls": 999},
    {"query": "news", "exclude_domains": []}, {"query": "news", "include_domains": ["kalshi.com"]},
])
def test_agents_cannot_override_credentials_identity_or_limits(participant_rig, backend, arguments):
    rig = participant_rig
    tools = ParticipantTools(rig.exchange, "agent-01", rig.settings)
    for outcome in (tools.dispatch("search_news", json.dumps(arguments)),
                    asyncio.run(tools.dispatch_async("search_news", json.dumps(arguments)))):
        assert outcome.dispatched is False
        assert outcome.result["error"]["code"] == "invalid_arguments"
    assert backend.calls == []


def test_domain_policy_is_not_exposed_in_agent_prompt_or_schema(participant_rig, backend):
    rig = participant_rig
    backend.body = {"results": [{"url": "https://news.kalshi.com/article", "content": "market odds"}]}
    participant, model = rig.make(
        rig.response(rig.call("search_news", {"query": "film reviews"})),
        rig.response(rig.call("get_account")),
    )
    with pytest.raises(rig.finished):
        asyncio.run(participant.run())
    assert rig.state()["news_tool_calls"] == 1
    assert {k: v for k, v in results(rig)[0].items() if k not in {"next_action_at", "session_closes_at"}} == {"ok": True, "query": "film reviews", "results": [], "tool_call_executed": True}
    for visible in (json.dumps(model.requests), json.dumps(rig.entries())):
        assert "exclude_domains" not in visible and "polymarket" not in visible.lower()
        assert "news.kalshi.com" not in visible and "market odds" not in visible


def results(rig, agent_id="agent-01"):
    return [row["content"]["result"] for row in rig.entries(agent_id)
            if row["entry_type"] == "tool_result"]


def test_news_searches_continue_past_former_count_cap_with_elapsed_costs(participant_rig, backend):
    rig = participant_rig
    initial = rig.clock.now
    participant, model = rig.make(
        *(rig.response(rig.call("search_news", {"query": f"news {index}"})) for index in range(12)),
        rig.response(rig.call("get_account")),
    )
    with pytest.raises(rig.finished):
        asyncio.run(participant.run())
    assert len(backend.calls) == 12
    assert rig.state()["news_tool_calls"] == 12
    assert rig.clock.sleeps == [600] * 12
    assert rig.clock.now == initial + 12 * 600
    assert rig.state()["shared_tool_calls"] == 1 and rig.market_calls() == {}
    assert all(row["tool_call_executed"] and row["ok"] for row in results(rig))
    for request in model.requests:
        assert "news_tool_calls_remaining" not in json.loads(request["messages"][-1]["content"])
        assert "news budget" not in json.dumps(request).lower()
    assert "private-news-test-key" not in "\n".join(rig.conn.iterdump())


def test_failed_search_costs_time_and_later_turn_can_search_again(participant_rig, backend):
    rig = participant_rig
    backend.status, backend.body = 429, {"detail": {"error": "quota exceeded"}}
    participant, _ = rig.make(
        rig.response(rig.call("search_news", {"query": "news"}),
                     rig.call("search_news", {"query": "second"})),
        rig.response(rig.call("search_news", {"query": "third"})),
    )
    with pytest.raises(rig.finished):
        asyncio.run(participant.run())
    assert len(backend.calls) == 2 and rig.state()["news_tool_calls"] == 2
    assert rig.clock.sleeps == [600]
    assert [row["tool_call_executed"] for row in results(rig)] == [True, False, True]


def test_news_activity_counter_resets_at_next_session_without_limiting_searches(participant_rig, backend):
    rig = participant_rig
    participant, model = rig.make(
        rig.response(rig.call("search_news", {"query": "news"})),
        rig.response(rig.call("hold_for_now")),
        rig.response(rig.call("search_news", {"query": "again"})),
        rig.response(rig.call("hold_until_next_day")),
        rig.response(rig.call("search_news", {"query": "next day"})),
        settings=rig.tight(hold_for_now_minutes=1), sleep=rig.advance_sleep,
    )
    with pytest.raises(rig.finished):
        asyncio.run(participant.run())
    assert [call["query"] for call in backend.calls] == ["news", "again", "next day"]
    assert rig.clock.sleeps[:3] == [600, 60, 600]
    assert sum(rig.clock.sleeps) == 86400 and rig.state()["news_tool_calls"] == 1
    assert "news_tool_calls_remaining" not in json.loads(model.requests[4]["messages"][-1]["content"])


def test_news_search_delays_next_order_by_its_time_cost(participant_rig, backend):
    rig = participant_rig
    participant, _ = rig.make(
        rig.response(rig.call("search_news", {"query": "news"})),
        rig.response(rig.call("submit_order", {"ticker": "MKT-01", "action": "buy",
                     "order_type": "limit", "quantity": 1, "price_cents": 40})),
    )
    with pytest.raises(rig.finished):
        asyncio.run(participant.run())
    assert rig.state()["news_tool_calls"] == 1
    assert rig.clock.sleeps == [600]
    assert rig.market_calls() == {"MKT-01": 1}


def test_news_can_be_the_first_action_without_posting_quotes(participant_rig, backend):
    rig = participant_rig
    participant, _ = rig.make(
        rig.response(rig.call("search_news", {"query": "news"})),
    )
    with pytest.raises(rig.finished):
        asyncio.run(participant.run())
    assert results(rig)[0]["ok"] is True
    assert results(rig)[0]["tool_call_executed"] is True
    assert rig.state()["news_tool_calls"] == 1
    assert len(backend.calls) == 1 and backend.calls[0]["query"] == "news"
    assert rig.state()["shared_tool_calls"] == 0 and rig.market_calls() == {}
    assert rig.conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_pending_news_does_not_block_other_agents_and_cancellation_preserves_charge(participant_rig, backend):
    rig = participant_rig

    async def scenario():
        started = asyncio.Event()

        async def hang():
            started.set()
            await asyncio.Future()

        backend.effect = hang
        first, _ = rig.make(rig.response(rig.call("search_news", {"query": "news"})))
        second, _ = rig.make(rig.response(rig.call("get_account")), agent_id="agent-02")
        task = asyncio.create_task(first.run())
        try:
            await asyncio.wait_for(started.wait(), 1)
            with pytest.raises(rig.finished):
                await asyncio.wait_for(second.run(), 1)
            assert rig.state()["news_tool_calls"] == 1
            assert rig.state("agent-02")["news_tool_calls"] == 0
            assert rig.state("agent-02")["shared_tool_calls"] == 1
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())
    assert backend.clients[0].is_closed and len(backend.calls) == 1
    assert rig.state()["news_tool_calls"] == 1
    assert results(rig) == []


def test_yaml_news_settings_are_loaded_and_snapshot_contains_only_key_name(tmp_path, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "private-news-test-key")
    settings = config.load(config.CONFIG_DIR / "mvp.yaml")
    assert "news_tool_calls_per_day" not in settings.snapshot()
    assert settings.news_tool_minutes == 10
    assert settings.tavily_api_key_env == "TAVILY_API_KEY"
    assert "private-news-test-key" not in config.as_json(settings)
    with pytest.raises(ValueError, match="tavily_api_key_env"):
        replace(settings, tavily_api_key_env="not an env name")
