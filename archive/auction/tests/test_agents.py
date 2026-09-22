"""Response parsing, the quote frame, and brief rendering."""

import time

import pytest

from prediction_markets import agents, config, database


@pytest.fixture
def db(tmp_path):
    database.init(tmp_path / "test.db")
    now = int(time.time())
    database.upsert_markets([{
        "ticker": "T1", "event_ticker": "EV-T1", "title": "Example market",
        "rules": "Example rules.",
        "sector": "Politics", "open_ts": now - 3600,
        "close_ts": now + 48 * 3600, "first_seen": now,
    }])
    database.register_agent("a1", config.INITIAL_CASH_CENTS)
    yield now
    database.close()


@pytest.fixture
def no_frame(monkeypatch):
    monkeypatch.setattr(config, "QUOTE_SIDE", "no")


# --- parsing ---------------------------------------------------------------


def test_valid_json_parses():
    quote = agents.parse_quote(
        '{"bid_cents": 40, "bid_size": 10, "ask_cents": 60, "ask_size": 5}'
    )
    assert quote == {"bid_cents": 40, "bid_size": 10, "ask_cents": 60, "ask_size": 5}


def test_json_embedded_in_prose_parses():
    quote = agents.parse_quote(
        'Here is my quote:\n```json\n'
        '{"bid_cents": 30, "bid_size": 2, "ask_cents": 70, "ask_size": 3}\n```\n'
    )
    assert quote["bid_cents"] == 30 and quote["ask_size"] == 3


@pytest.mark.parametrize("payload, fragment", [
    ('{"bid_cents": 60, "bid_size": 1, "ask_cents": 40, "ask_size": 1}', "bid_cents <"),
    ('{"bid_cents": 50, "bid_size": 1, "ask_cents": 50, "ask_size": 1}', "bid_cents <"),
    ('{"bid_cents": 0, "bid_size": 1, "ask_cents": 50, "ask_size": 1}', "bid_cents <"),
    ('{"bid_cents": 40, "bid_size": 1, "ask_cents": 100, "ask_size": 1}', "bid_cents <"),
    ('{"bid_cents": 40, "bid_size": 0, "ask_cents": 60, "ask_size": 1}', "bid_size"),
    ('{"bid_cents": 40, "bid_size": 999, "ask_cents": 60, "ask_size": 1}', "bid_size"),
    ('{"bid_cents": 40.5, "bid_size": 1, "ask_cents": 60, "ask_size": 1}', "whole number"),
    ('{"bid_cents": 40, "ask_cents": 60, "ask_size": 1}', "missing"),
    ('not json at all', "no JSON object"),
    ('', "empty response"),
])
def test_invalid_responses_are_rejected(payload, fragment):
    with pytest.raises(agents.QuoteError) as exc:
        agents.parse_quote(payload)
    assert fragment in str(exc.value)


# --- the quote frame -------------------------------------------------------


def test_no_frame_maps_a_quote_into_yes_space():
    """Buying NO at 30 is selling YES at 70; the sides swap as well as invert."""
    quote = agents.parse_quote(
        '{"bid_cents": 30, "bid_size": 7, "ask_cents": 45, "ask_size": 3}',
        quote_side="no",
    )
    assert quote == {
        "bid_cents": 55, "bid_size": 3,   # was the NO ask of 45
        "ask_cents": 70, "ask_size": 7,   # was the NO bid of 30
    }
    assert quote["bid_cents"] < quote["ask_cents"]


def test_no_frame_round_trips_through_the_brief(db, no_frame):
    """A NO quote becomes the right YES order, and comes back as the same NO
    numbers when the agent is shown its own history."""
    raw = '{"bid_cents": 30, "bid_size": 7, "ask_cents": 45, "ask_size": 3}'
    quote = agents.parse_quote(raw, quote_side="no")

    database.insert_quotes([{
        "round_id": 1, "agent": "a1", "ticker": "T1",
        "bid_cents": quote["bid_cents"], "bid_size": quote["bid_size"],
        "ask_cents": quote["ask_cents"], "ask_size": quote["ask_size"],
        "status": "accepted", "reject_reason": None,
    }])

    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "round 1: 30c x7 at 45c x3" in brief


def test_no_frame_shows_clears_and_positions_in_no_terms(db, no_frame):
    database.record_auction(
        {"ticker": "T1", "round_id": 1, "ts": 0, "clear_cents": 70,
         "cleared_qty": 5, "tie_lo": 70, "tie_hi": 70},
        [{"round_id": 1, "agent": "a1", "ticker": "T1",
          "signed_qty": -5, "price_cents": 70}],
    )
    database.set_position("a1", "T1", -5, -350)

    brief = agents.render_brief("a1", "T1", round_id=2)
    # A YES clear of 70 is a NO clear of 30, and short 5 YES is long 5 NO.
    assert "cleared at 30c" in brief
    assert "+5 contracts" in brief


def test_yes_frame_is_the_identity():
    payload = '{"bid_cents": 30, "bid_size": 7, "ask_cents": 45, "ask_size": 3}'
    assert agents.parse_quote(payload, "yes") == {
        "bid_cents": 30, "bid_size": 7, "ask_cents": 45, "ask_size": 3,
    }


# --- prompt and roster -----------------------------------------------------


def test_prompt_states_the_contract_for_the_configured_frame(monkeypatch):
    monkeypatch.setattr(config, "QUOTE_SIDE", "yes")
    assert "quoting the YES contract" in agents.build_prompt()
    monkeypatch.setattr(config, "QUOTE_SIDE", "no")
    assert "quoting the NO contract" in agents.build_prompt()


def test_prompt_never_mentions_v1_tools():
    prompt = agents.build_prompt()
    assert "web_search" not in prompt and "get_news" not in prompt


def test_roster_instantiates_one_agent_per_spec():
    roster = agents.build_roster()
    assert len(roster) == len(config.AGENTS)
    assert [a.agent_id for a in roster] == [s["agent_id"] for s in config.AGENTS]
    assert all(a.tools == [] for a in roster)
    # v0 shares a single prompt across the roster, which is also what makes
    # the cached static prefix identical for every call.
    assert len({a.prompt for a in roster}) == 1


# --- the brief -------------------------------------------------------------


def test_first_round_brief_says_there_is_no_history(db):
    brief = agents.render_brief("a1", "T1", round_id=1)
    assert "No rounds have been held yet" in brief
    assert "You have not quoted this market before." in brief
    assert "Position in this market: flat" in brief


def test_brief_reports_no_trade_rounds(db):
    database.record_auction(
        {"ticker": "T1", "round_id": 1, "ts": 0, "clear_cents": None,
         "cleared_qty": 0, "tie_lo": None, "tie_hi": None}, [],
    )
    assert "round 1: no trade" in agents.render_brief("a1", "T1", round_id=2)


def _no_trade_round(round_id=1, quotes=()):
    """A round that printed nothing, with `quotes` as (agent, bid, bid_size,
    ask, ask_size, status) tuples behind it."""
    database.insert_quotes([{
        "round_id": round_id, "agent": agent, "ticker": "T1",
        "bid_cents": bid, "bid_size": bid_size,
        "ask_cents": ask, "ask_size": ask_size,
        "status": status, "reject_reason": None if status == "accepted" else "margin",
    } for agent, bid, bid_size, ask, ask_size, status in quotes])
    database.record_auction(
        {"ticker": "T1", "round_id": round_id, "ts": 0, "clear_cents": None,
         "cleared_qty": 0, "tie_lo": None, "tie_hi": None}, [],
    )


def test_no_trade_round_shows_the_top_of_book(db):
    database.register_agent("a2", config.INITIAL_CASH_CENTS)
    _no_trade_round(quotes=[
        ("a1", 30, 4, 70, 6, "accepted"),
        ("a2", 40, 5, 55, 7, "accepted"),
    ])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "round 1: no trade, best bid 40c x5, best ask 55c x7" in brief


def test_top_of_book_sums_size_across_agents_at_the_best_level(db):
    """Two agents on the same price are one level, not two — which is also
    what stops a level from identifying the agent behind it."""
    for agent in ("a2", "a3"):
        database.register_agent(agent, config.INITIAL_CASH_CENTS)
    _no_trade_round(quotes=[
        ("a1", 40, 5, 55, 7, "accepted"),
        ("a2", 40, 3, 55, 2, "accepted"),
        ("a3", 35, 9, 60, 9, "accepted"),
    ])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "no trade, best bid 40c x8, best ask 55c x9" in brief


def test_top_of_book_excludes_rejected_quotes(db):
    database.register_agent("a2", config.INITIAL_CASH_CENTS)
    _no_trade_round(quotes=[
        ("a1", 30, 4, 70, 6, "accepted"),
        ("a2", 45, 5, 50, 5, "rejected"),
    ])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "no trade, best bid 30c x4, best ask 70c x6" in brief


def test_no_trade_round_without_quotes_shows_no_book(db):
    _no_trade_round()
    assert "round 1: no trade\n" in agents.render_brief("a1", "T1", round_id=2)


def test_a_cleared_round_shows_no_book(db):
    database.insert_quotes([{
        "round_id": 1, "agent": "a1", "ticker": "T1", "bid_cents": 40,
        "bid_size": 5, "ask_cents": 60, "ask_size": 5,
        "status": "accepted", "reject_reason": None,
    }])
    database.record_auction(
        {"ticker": "T1", "round_id": 1, "ts": 0, "clear_cents": 50,
         "cleared_qty": 5, "tie_lo": 50, "tie_hi": 50}, [],
    )
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "round 1: cleared at 50c, volume x5" in brief
    assert "best bid" not in brief


def _cleared_round(clear_cents, cleared_qty, quotes, round_id=1):
    """A round that printed, with `quotes` as (agent, bid, bid_size, ask,
    ask_size) tuples behind it."""
    database.insert_quotes([{
        "round_id": round_id, "agent": agent, "ticker": "T1",
        "bid_cents": bid, "bid_size": bid_size,
        "ask_cents": ask, "ask_size": ask_size,
        "status": "accepted", "reject_reason": None,
    } for agent, bid, bid_size, ask, ask_size in quotes])
    database.record_auction(
        {"ticker": "T1", "round_id": round_id, "ts": 0,
         "clear_cents": clear_cents, "cleared_qty": cleared_qty,
         "tie_lo": clear_cents, "tie_hi": clear_cents}, [],
    )


def test_cleared_round_reports_the_unfilled_bid_side(db):
    """10 contracts bid at or above 48c, 4 offered at or below: 6 could not
    trade, and the starved side is the one the exchange reports."""
    database.register_agent("a2", config.INITIAL_CASH_CENTS)
    _cleared_round(48, 4, [("a1", 50, 10, 60, 10), ("a2", 30, 2, 45, 4)])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "round 1: cleared at 48c, volume x4, unfilled bid x6" in brief


def test_cleared_round_reports_the_unfilled_ask_side(db):
    database.register_agent("a2", config.INITIAL_CASH_CENTS)
    _cleared_round(48, 2, [("a1", 50, 2, 60, 10), ("a2", 30, 2, 45, 9)])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "round 1: cleared at 48c, volume x2, unfilled ask x7" in brief


def test_a_fully_matched_clear_reports_zero_unfilled(db):
    database.register_agent("a2", config.INITIAL_CASH_CENTS)
    _cleared_round(48, 5, [("a1", 50, 5, 60, 10), ("a2", 30, 2, 45, 5)])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "round 1: cleared at 48c, volume x5, unfilled 0" in brief


def test_no_frame_flips_the_unfilled_side(db, no_frame):
    """Excess demand for YES is excess supply of NO."""
    database.register_agent("a2", config.INITIAL_CASH_CENTS)
    _cleared_round(48, 4, [("a1", 50, 10, 60, 10), ("a2", 30, 2, 45, 4)])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "cleared at 52c, volume x4, unfilled ask x6" in brief


def test_unfilled_ignores_rejected_quotes(db):
    database.register_agent("a2", config.INITIAL_CASH_CENTS)
    _cleared_round(48, 4, [("a1", 50, 10, 60, 10), ("a2", 30, 2, 45, 4)])
    database.insert_quotes([{
        "round_id": 1, "agent": "a3", "ticker": "T1", "bid_cents": 55,
        "bid_size": 40, "ask_cents": 60, "ask_size": 40,
        "status": "rejected", "reject_reason": "worse-side margin",
    }])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "unfilled bid x6" in brief


def test_no_frame_shows_the_book_in_no_terms(db, no_frame):
    """A YES book of 40/55 is a NO book of 45/60: the sides invert and swap,
    and each price keeps the size that was quoted at it."""
    _no_trade_round(quotes=[("a1", 40, 5, 55, 7, "accepted")])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "no trade, best bid 45c x7, best ask 60c x5" in brief


def test_brief_reports_a_rejected_quote_with_its_reason(db):
    database.insert_quotes([{
        "round_id": 1, "agent": "a1", "ticker": "T1", "bid_cents": 40,
        "bid_size": 10, "ask_cents": 60, "ask_size": 10,
        "status": "rejected", "reject_reason": "worse-side margin",
    }])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "quote rejected (worse-side margin)" in brief


def test_brief_never_shows_another_agents_quote(db):
    database.register_agent("a2", config.INITIAL_CASH_CENTS)
    database.insert_quotes([{
        "round_id": 1, "agent": "a2", "ticker": "T1", "bid_cents": 44,
        "bid_size": 11, "ask_cents": 66, "ask_size": 11,
        "status": "accepted", "reject_reason": None,
    }])
    brief = agents.render_brief("a1", "T1", round_id=2)
    assert "a2" not in brief
    assert "44" not in brief and "66" not in brief
