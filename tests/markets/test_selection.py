"""One-time selection and definition normalization, with no network or DB."""

from dataclasses import replace
from datetime import datetime, timezone
from decimal import localcontext

import pytest

from prediction_markets import config
from prediction_markets.storage import database as database
from prediction_markets.markets import selection as live_markets


NOW = 1_800_000_000
DAY = 86_400
HOUR = 3_600


def iso(second):
    return datetime.fromtimestamp(second, timezone.utc).isoformat().replace("+00:00", "Z")


def candidate(ticker="M0", *, volume="1000.00", **changes):
    return {
        "ticker": ticker,
        "event_ticker": f"EVENT-{ticker}",
        "categories": ["Politics"],
        "market_type": "binary",
        "title": f"Will {ticker} occur?",
        "rules_primary": "  Full primary rules.\nNo truncation.  ",
        "rules_secondary": "Additional settlement terms.",
        "status": "active",
        "open_time": iso(NOW - DAY),
        "close_time": iso(NOW + 70 * HOUR),
        "expected_expiration_time": iso(NOW + 100 * HOUR),
        "latest_expiration_time": iso(NOW + 200 * HOUR),
        "volume_24h_fp": volume,
        "yes_bid_dollars": "0.44",
        "yes_ask_dollars": "0.46",
        "last_price_dollars": "0.45",
        **changes,
    }


@pytest.fixture
def settings():
    # Keep synthetic timestamps independent of edits to the run configuration.
    return replace(
        config.load(config.CONFIG_DIR / "mvp.yaml"),
        expected_expiration_min_hours=72, expected_expiration_max_hours=168,
    )


def select(candidates, settings, **changes):
    return live_markets.select_markets(
        candidates, settings, startup_at=NOW,
        observed_at=changes.get("observed_at", NOW),
    )


def test_selects_ten_distinct_events_with_deterministic_volume_tiebreak(settings):
    candidates = [candidate(f"M{i:02d}") for i in range(12)]
    candidates.extend([
        candidate("TOP", volume="9999", event_ticker="EVENT-M00"),
        candidate("SECOND", volume="9000", event_ticker="EVENT-M00"),
    ])
    rows = select(reversed(candidates), settings)

    assert [row["ticker"] for row in rows] == ["TOP"] + [f"M{i:02d}" for i in range(1, 10)]
    assert len({row["event_ticker"] for row in rows}) == 10
    assert all(set(row) == database._MARKET_FIELDS for row in rows)
    assert all(row["last_successful_poll"] == NOW for row in rows)
    assert all(row["payout_cents"] is None for row in rows)


def test_insufficient_distinct_events_reports_actual_eligible_count(settings):
    candidates = [candidate(f"A{i}", event_ticker="SAME") for i in range(30)]
    candidates.extend([candidate("B"), candidate("C", volume="499.99")])
    with pytest.raises(ValueError, match="only 2 eligible markets across distinct events; need 10"):
        select(candidates, settings)


@pytest.mark.parametrize("category", [
    "Elections", "Politics", "Entertainment", "Commodities", "Climate and Weather",
    "Economics", "Mentions", "Financials", "Science and Technology",
])
def test_each_requested_category_is_eligible(settings, category):
    rows = select([candidate(categories=[category])], replace(settings, market_count=1))
    assert len(rows) == 1
    assert "categories" not in rows[0]


@pytest.mark.parametrize("categories", [
    ["Sports"], ["Crypto"], ["World"], ["Companies"], ["Social"],
    ["Mentions", "Sports"], ["Financials", "Crypto"], ["Politics", "Unknown"],
    None, [], "Politics", {}, [None], [["Politics"]], [""],
])
def test_disallowed_or_unknown_categories_cannot_win_on_volume(settings, categories):
    rows = select([
        candidate("EXCLUDED", categories=categories, volume="999999"), candidate("ALLOWED"),
    ], replace(settings, market_count=1))
    assert [row["ticker"] for row in rows] == ["ALLOWED"]


def test_missing_category_never_relaxes_filter(settings):
    payload = candidate()
    del payload["categories"]
    with pytest.raises(ValueError, match="only 0 eligible"):
        select([payload], replace(settings, market_count=1))


def test_allowlist_is_configurable_and_accepts_allowed_cross_listings(settings):
    settings = replace(settings, market_count=1, allowed_market_categories=("Elections", "Politics"))
    rows = select([
        candidate("MENTIONS", categories=["Mentions"], volume="99999"),
        candidate("POLITICS", categories=["Elections", "Politics"]),
    ], settings)
    assert [row["ticker"] for row in rows] == ["POLITICS"]


def test_rotation_visits_sectors_in_configuration_order_before_repeating(settings):
    settings = replace(
        settings, market_count=4,
        allowed_market_categories=("Elections", "Politics", "Economics"),
    )
    rows = select([
        candidate("POL-HIGH", categories=["Politics"], volume="9000"),
        candidate("POL-MID", categories=["Politics"], volume="8000"),
        candidate("POL-LOW", categories=["Politics"], volume="7000"),
        candidate("ELECT", categories=["Elections"], volume="510"),
        candidate("ECON", categories=["Economics"], volume="520"),
    ], settings)

    # One per sector in configuration order, then back to the head of the list.
    assert [row["ticker"] for row in rows] == ["ELECT", "POL-HIGH", "ECON", "POL-MID"]


def test_one_high_volume_sector_cannot_claim_the_whole_cohort(settings):
    settings = replace(
        settings, market_count=3,
        allowed_market_categories=("Climate and Weather", "Economics"),
    )
    candidates = [
        candidate(f"WX{index}", categories=["Climate and Weather"], volume=str(90_000 - index))
        for index in range(5)
    ]
    candidates.append(candidate("ECON", categories=["Economics"], volume="600"))
    rows = select(candidates, settings)

    # Ranking by volume alone would have returned WX0, WX1, WX2.
    assert [row["ticker"] for row in rows] == ["WX0", "ECON", "WX1"]


def test_exhausted_sectors_drop_out_while_later_passes_continue(settings):
    settings = replace(
        settings, market_count=4, allowed_market_categories=("Elections", "Politics"),
    )
    rows = select([
        candidate("ELECT", categories=["Elections"]),
        *[
            candidate(f"POL{index}", categories=["Politics"], volume=str(5_000 - index))
            for index in range(3)
        ],
    ], settings)

    assert [row["ticker"] for row in rows] == ["ELECT", "POL0", "POL1", "POL2"]


def test_duplicate_event_is_skipped_without_costing_the_sector_its_turn(settings):
    settings = replace(
        settings, market_count=2, allowed_market_categories=("Elections", "Politics"),
    )
    rows = select([
        candidate("ELECT", categories=["Elections"]),
        candidate("POL-DUP", categories=["Politics"], volume="9000",
                  event_ticker="EVENT-ELECT"),
        candidate("POL-OK", categories=["Politics"], volume="8000"),
    ], settings)

    assert [row["ticker"] for row in rows] == ["ELECT", "POL-OK"]


def test_cross_listed_market_waits_in_its_primary_sector_only(settings):
    settings = replace(
        settings, market_count=2, allowed_market_categories=("Elections", "Politics"),
    )
    rows = select([
        candidate("CROSS", categories=["Elections", "Politics"], volume="9000"),
        candidate("POL", categories=["Politics"], volume="510"),
    ], settings)

    # CROSS takes the Elections turn, so Politics still receives its own market.
    assert [row["ticker"] for row in rows] == ["CROSS", "POL"]


def test_resolution_window_is_inclusive_hours_anchored_to_startup(settings):
    settings = replace(settings, market_count=2)
    candidates = [
        candidate("MIN", expected_expiration_time=iso(NOW + 72 * HOUR)),
        candidate("MAX", expected_expiration_time=iso(NOW + 168 * HOUR)),
        candidate("EARLY", expected_expiration_time=iso(NOW + 72 * HOUR - 1)),
        candidate("LATE", expected_expiration_time=iso(NOW + 168 * HOUR + 1)),
        candidate("DAYS", expected_expiration_time=iso(NOW + 100 * DAY),
                  latest_expiration_time=iso(NOW + 101 * DAY)),
    ]
    assert [row["ticker"] for row in select(candidates, settings, observed_at=NOW + DAY)] == ["MAX", "MIN"]


@pytest.mark.parametrize("changes", [
    {"market_type": "scalar"},
    {"status": "initialized"},
    {"status": "inactive"},
    {"status": "closed"},
    {"status": "determined"},
    {"status": "finalized"},
    {"status": "unknown"},
    {"event_ticker": ""},
    {"title": ""},
    {"rules_primary": "", "rules_secondary": ""},
    {"open_time": iso(NOW + 1)},
    {"open_time": None},
    {"open_time": "invalid"},
    {"open_time": "2027-01-15T00:00:00"},
    {"open_time": "2027-01-15T00:00:00.1Z"},
    {"close_time": iso(NOW)},
    {"close_time": iso(NOW - 1)},
    {"expected_expiration_time": None},
    {"expected_expiration_time": "invalid"},
    {"latest_expiration_time": "invalid"},
    {"latest_expiration_time": iso(NOW + 72 * HOUR - 1)},
    {"volume_24h_fp": "499.999"},
    {"notional_value_dollars": "2.00"},
])
def test_ineligible_or_malformed_candidate_is_skipped(settings, changes):
    settings = replace(settings, market_count=1)
    rows = select([candidate("BAD", **changes), candidate("GOOD")], settings)
    assert [row["ticker"] for row in rows] == ["GOOD"]


def test_future_close_is_checked_at_observation_not_startup(settings):
    settings = replace(settings, market_count=1)
    rows = select([
        candidate("CLOSED", close_time=iso(NOW + 10)), candidate("OPEN"),
    ], settings, observed_at=NOW + 10)
    assert [row["ticker"] for row in rows] == ["OPEN"]


def test_open_time_is_inclusive_and_checked_at_observation(settings):
    settings = replace(settings, market_count=1)
    payload = candidate(open_time=iso(NOW + 10))
    with pytest.raises(ValueError, match="only 0 eligible"):
        select([payload], settings, observed_at=NOW + 9)
    assert select([payload], settings, observed_at=NOW + 10)[0]["ticker"] == "M0"


def test_missing_open_time_cannot_be_replaced_by_active_status(settings):
    payload = candidate()
    del payload["open_time"]
    with pytest.raises(ValueError, match="only 0 eligible"):
        select([payload], replace(settings, market_count=1))


@pytest.mark.parametrize("latest", [None, iso(NOW + 100 * HOUR)])
def test_expected_expiration_can_equal_latest_or_latest_can_be_unknown(settings, latest):
    payload = candidate(latest_expiration_time=latest)
    assert select([payload], replace(settings, market_count=1))[0]["ticker"] == "M0"


@pytest.mark.parametrize("expected_hours,latest_hours", [
    (100, 200), (200, 100), (100, 100), (100, None), (None, 100),
])
def test_earlier_available_expiration_selects_without_overwriting_source_fields(
    settings, expected_hours, latest_hours,
):
    expected = None if expected_hours is None else iso(NOW + expected_hours * HOUR)
    latest = None if latest_hours is None else iso(NOW + latest_hours * HOUR)
    row = select([candidate(expected_expiration_time=expected, latest_expiration_time=latest)],
                 replace(settings, market_count=1))[0]
    assert row["expected_expiration_time"] == (None if expected_hours is None else NOW + expected_hours * HOUR)
    assert row["latest_expiration_time"] == (None if latest_hours is None else NOW + latest_hours * HOUR)
    assert set(row) == database._MARKET_FIELDS


@pytest.mark.parametrize("earlier_seconds,eligible", [
    (72 * HOUR - 1, False), (72 * HOUR, True),
    (168 * HOUR, True), (168 * HOUR + 1, False),
])
@pytest.mark.parametrize("earlier_field", ["expected_expiration_time", "latest_expiration_time"])
def test_earlier_expiration_controls_both_inclusive_boundaries(settings, earlier_seconds, eligible, earlier_field):
    payload = candidate(
        expected_expiration_time=iso(NOW + 200 * HOUR),
        latest_expiration_time=iso(NOW + 200 * HOUR),
    )
    payload[earlier_field] = iso(NOW + earlier_seconds)
    settings = replace(settings, market_count=1)
    if eligible:
        assert select([payload], settings)[0]["ticker"] == "M0"
    else:
        with pytest.raises(ValueError, match="only 0 eligible"):
            select([payload], settings)


def test_earlier_expiration_respects_a_different_configured_hour_window(settings):
    settings = replace(settings, market_count=1, expected_expiration_min_hours=12,
                       expected_expiration_max_hours=72)
    rows = select([
        candidate("IN_WINDOW", latest_expiration_time=iso(NOW + 24 * HOUR)),
        candidate("TOO_EARLY", latest_expiration_time=iso(NOW + 11 * HOUR)),
        candidate("TOO_LATE", latest_expiration_time=iso(NOW + 73 * HOUR)),
    ], settings)
    assert [row["ticker"] for row in rows] == ["IN_WINDOW"]


def test_missing_both_expirations_does_not_fall_back_to_close(settings):
    missing = candidate(close_time=iso(NOW + 100 * HOUR))
    del missing["expected_expiration_time"]
    del missing["latest_expiration_time"]
    with pytest.raises(ValueError, match="only 0 eligible"):
        select([missing], replace(settings, market_count=1))


def test_bid_ask_do_not_determine_eligibility_and_source_statistics_are_not_kept(settings):
    payload = candidate(yes_bid_dollars="0.00", yes_ask_dollars="1.00")
    row = select([payload], replace(settings, market_count=1))[0]
    assert set(row) == database._MARKET_FIELDS
    assert "volume_24h_fp" not in row
    assert "yes_bid_dollars" not in row


@pytest.mark.parametrize("price, eligible", [
    ("0.10", True), ("0.90", True), ("0.435", True), (0.5, True),
    ("0.099999999999999999999999999999999", False),
    ("0.900000000000000000000000000000001", False),
    ("0.00", False), ("1.00", False), ("1.01", False),
    (None, False), (True, False), ("", False), ("NaN", False),
    ("Infinity", False), ("-0.1", False), ("bad", False), ([], False), ({}, False),
])
def test_initial_trade_price_filter_is_inclusive_and_never_rounds(settings, price, eligible):
    settings = replace(settings, market_count=1, uncertainty_price_range_cents=(10, 90))
    with localcontext() as context:
        context.prec = 6
        rows = select([
            candidate("CANDIDATE", volume="9000", last_price_dollars=price),
            candidate("FALLBACK"),
        ], settings)
    assert rows[0]["ticker"] == ("CANDIDATE" if eligible else "FALLBACK")


def test_missing_trade_price_never_falls_back_to_quotes_or_legacy_price(settings):
    payload = candidate(last_price=50, yes_bid_dollars="0.49", yes_ask_dollars="0.51")
    del payload["last_price_dollars"]
    with pytest.raises(ValueError, match="only 0 eligible"):
        select([payload], replace(settings, market_count=1))


def test_uncertainty_range_is_configurable_and_can_select_one_exact_price(settings):
    settings = replace(settings, market_count=1, uncertainty_price_range_cents=(50, 50))
    rows = select([
        candidate("LOW", volume="9000", last_price_dollars="0.49"),
        candidate("HIGH", volume="9000", last_price_dollars="0.51"),
        candidate("EXACT", last_price_dollars="0.5000"),
    ], settings)
    assert rows[0]["ticker"] == "EXACT"


def test_uncertainty_filter_applies_before_sector_rotation_and_volume_ranking(settings):
    settings = replace(settings, market_count=2,
                       allowed_market_categories=("Elections", "Politics"))
    rows = select([
        candidate("OUTSIDE", categories=["Elections"], volume="99999", last_price_dollars="0.99"),
        candidate("INSIDE", categories=["Elections"], last_price_dollars="0.10"),
        candidate("POLITICS", volume="9000", last_price_dollars="0.90"),
    ], settings)
    assert [row["ticker"] for row in rows] == ["INSIDE", "POLITICS"]


def test_price_filter_is_never_relaxed_to_fill_the_cohort(settings):
    with pytest.raises(ValueError, match="only 1 eligible markets.*need 2"):
        select([candidate("GOOD"), candidate("OUTSIDE", last_price_dollars="0.99")],
               replace(settings, market_count=2))


def test_volume_floor_is_inclusive_and_fixed_point_takes_precedence(settings):
    settings = replace(settings, market_count=1)
    rows = select([
        candidate("LEGACY_HIGH", volume="499", volume_24h=999999),
        candidate("EXACT_FLOOR", volume="500.00", volume_24h=0),
    ], settings)
    assert [row["ticker"] for row in rows] == ["EXACT_FLOOR"]


def test_legacy_volume_fallback_and_decimal_sort_do_not_round(settings):
    settings = replace(settings, market_count=2)
    legacy = candidate("LEGACY")
    del legacy["volume_24h_fp"]
    legacy["volume_24h"] = 501
    with localcontext() as context:
        context.prec = 6
        rows = select([
            candidate("A", volume="1000.000000000000000000000000000001"),
            candidate("Z", volume="1000.000000000000000000000000000002"),
            legacy,
        ], settings)
    assert [row["ticker"] for row in rows] == ["Z", "A"]
    assert select([legacy], replace(settings, market_count=1))[0]["ticker"] == "LEGACY"


@pytest.mark.parametrize("volume", [None, True, "NaN", "Infinity", "-1", "bad", "1_000", [], {}])
def test_invalid_fixed_point_volume_cannot_use_legacy_fallback(settings, volume):
    with pytest.raises(ValueError, match="only 0 eligible"):
        select([candidate(volume=volume, volume_24h=1000)], replace(settings, market_count=1))


def test_normalization_preserves_rules_and_separately_labeled_times():
    payload = candidate()
    row = live_markets.market_row(payload, NOW)
    assert row["rules"] == payload["rules_primary"] + "\n\n" + payload["rules_secondary"]
    assert row["close_time"] == NOW + 70 * HOUR
    assert row["expected_expiration_time"] == NOW + 100 * HOUR
    assert row["latest_expiration_time"] == NOW + 200 * HOUR


def test_missing_optional_times_remain_unknown():
    payload = candidate()
    del payload["expected_expiration_time"]
    del payload["latest_expiration_time"]
    row = live_markets.market_row(payload, NOW)
    assert row["expected_expiration_time"] is None
    assert row["latest_expiration_time"] is None


def test_timestamp_with_explicit_offset_is_normalized_to_utc():
    row = live_markets.market_row(candidate(close_time="2027-01-15T03:30:00+03:30"), NOW)
    assert row["close_time"] == int(datetime(2027, 1, 15, tzinfo=timezone.utc).timestamp())


@pytest.mark.parametrize("field", ["close_time", "expected_expiration_time", "latest_expiration_time"])
@pytest.mark.parametrize("value", [
    "2027-01-15T00:00:00", "2027-01-15", "2027-01-15T00:00:00.001Z",
    "2027-01-15T00:00:00.0000001Z",
    "2027-02-30T00:00:00Z", "2027-01-15T00:00:00+00:99", "", 1_800_000_000,
])
def test_timestamps_reject_missing_timezone_fractional_or_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        live_markets.market_row(candidate(**{field: value}), NOW)


@pytest.mark.parametrize("field", ["ticker", "event_ticker", "title", "market_type", "status", "close_time"])
def test_required_definition_fields_cannot_be_missing(field):
    payload = candidate()
    del payload[field]
    with pytest.raises(ValueError, match=field):
        live_markets.market_row(payload, NOW)


@pytest.mark.parametrize("time", [0, -1, True, 1.5, "1800000000"])
def test_observation_time_requires_positive_integer_seconds(time):
    with pytest.raises(ValueError, match="observed_at"):
        live_markets.market_row(candidate(), time)


def test_title_uses_yes_sub_title_when_deprecated_title_is_missing():
    payload = candidate(yes_sub_title="The current question")
    del payload["title"]
    assert live_markets.market_row(payload, NOW)["title"] == "The current question"
    payload["title"] = " "
    assert live_markets.market_row(payload, NOW)["title"] == "The current question"


def test_malformed_title_is_rejected_even_with_a_fallback():
    with pytest.raises(ValueError, match="title"):
        live_markets.market_row(candidate(title=123, yes_sub_title="Fallback"), NOW)


def test_duplicate_candidates_do_not_count_as_additional_events(settings):
    with pytest.raises(ValueError, match="only 1 eligible"):
        select([candidate(), candidate()], settings)


@pytest.mark.parametrize("suffix", [".000Z", ".000000+00:00", ".000000000000Z"])
def test_zero_fractional_timestamp_suffixes_preserve_exact_whole_seconds(suffix):
    row = live_markets.market_row(candidate(close_time="2027-01-15T00:00:00" + suffix), NOW)
    assert row["close_time"] == int(datetime(2027, 1, 15, tzinfo=timezone.utc).timestamp())


@pytest.mark.parametrize("notional", ["0.99", "1.0000000000001", "2.00", None, True, "NaN"])
def test_unsupported_notional_is_rejected(notional):
    with pytest.raises(ValueError, match="notional_value_dollars"):
        live_markets.market_row(candidate(notional_value_dollars=notional), NOW)


def test_dollar_notional_is_accepted_without_persisting_it():
    row = live_markets.market_row(candidate(notional_value_dollars="1.0000"), NOW)
    assert "notional_value_dollars" not in row


def test_source_price_never_reaches_the_stored_market_row(settings):
    rows = select([candidate(volume="9999")], replace(settings, market_count=1))
    assert "last_price_dollars" not in rows[0]
