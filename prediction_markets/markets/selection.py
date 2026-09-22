"""Pure normalization and one-time selection of the fixed market cohort."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from ..config import Settings


SECONDS_PER_DAY = 86_400
SECONDS_PER_HOUR = 3_600
_STATUSES = frozenset(
    {
        "initialized", "inactive", "active", "closed", "determined",
        "disputed", "amended", "finalized",
    }
)
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.0+)?"
    r"(?:[Zz]|[+-](?:[01]\d|2[0-3]):[0-5]\d)"
)
_DECIMAL = re.compile(r"[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _positive_second(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer UTC second")
    return value


def _timestamp(value: Any, name: str, *, optional: bool = False) -> int | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"{name} must be an RFC3339 timestamp with whole seconds and a timezone")
    try:
        # The pattern already guarantees every fractional digit is zero.
        # Remove that exact suffix before Python's precision-limited parser.
        normalized = re.sub(r"\.0+", "", value).upper().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        delta = parsed.astimezone(timezone.utc) - _EPOCH
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {name}") from exc
    return _positive_second(delta.days * SECONDS_PER_DAY + delta.seconds, name)


def _text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _decimal(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{name} must be a nonnegative decimal")
    text = str(value)
    if _DECIMAL.fullmatch(text) is None:
        raise ValueError(f"{name} must be a nonnegative decimal")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid {name}") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def market_row(payload: dict, observed_at: int) -> dict:
    """Keep definitions only; payout validation belongs to the polling boundary.

    Kalshi's primary and secondary rules retain their complete source text.
    Optional missing timestamps remain unknown, while malformed ones reject
    the observation rather than preserving misleading older data.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("market payload must be an object")
    observed_at = _positive_second(observed_at, "observed_at")
    market_type = _text(payload, "market_type")
    if market_type != "binary":
        raise ValueError("market_type must be binary")
    if "notional_value_dollars" in payload and _decimal(
        payload["notional_value_dollars"], "notional_value_dollars",
    ) != 1:
        raise ValueError("notional_value_dollars must be exactly 1")
    status = _text(payload, "status")
    if status not in _STATUSES:
        raise ValueError(f"unsupported market status: {status}")
    title = payload.get("title")
    if title is None or (isinstance(title, str) and not title.strip()):
        title = _text(payload, "yes_sub_title")
    elif not isinstance(title, str):
        raise ValueError("title must be a nonempty string")

    rules_parts = []
    for name in ("rules_primary", "rules_secondary"):
        text = payload.get(name, "")
        if not isinstance(text, str):
            raise ValueError(f"{name} must be a string")
        if text.strip():
            rules_parts.append(text)
    if not rules_parts:
        raise ValueError("market rules must be nonempty")

    return {
        "ticker": _text(payload, "ticker"),
        "event_ticker": _text(payload, "event_ticker"),
        "market_type": market_type,
        "title": title,
        "rules": "\n\n".join(rules_parts),
        "status": status,
        "close_time": _timestamp(payload.get("close_time"), "close_time"),
        "expected_expiration_time": _timestamp(
            payload.get("expected_expiration_time"), "expected_expiration_time",
            optional=True,
        ),
        "latest_expiration_time": _timestamp(
            payload.get("latest_expiration_time"), "latest_expiration_time",
            optional=True,
        ),
        "last_successful_poll": observed_at,
        "payout_cents": None,
    }


def select_markets(
    candidates: Iterable[dict], settings: Settings, *, startup_at: int,
    observed_at: int,
) -> list[dict]:
    """Select eligible markets from distinct events, one sector at a time, once.

    Use the earlier available expected/latest expiration for the startup
    horizon; retain both original fields. Trading must still be open at
    observation time, with the initial latest YES trade price inside the
    inclusive uncertainty range. Return no source prices or selection statistics.
    Sectors are visited in ``allowed_market_categories`` order and revisited
    until the cohort is full, so no single high-volume sector can claim it.
    Insufficient eligibility fails instead of weakening the configured filter.
    """
    startup_at = _positive_second(startup_at, "startup_at")
    observed_at = _positive_second(observed_at, "observed_at")
    earliest = startup_at + settings.expected_expiration_min_hours * SECONDS_PER_HOUR
    latest = startup_at + settings.expected_expiration_max_hours * SECONDS_PER_HOUR
    # Compare the original decimal dollar price without rounding fractional
    # cents. String conversion keeps the bounds exact even in a low-precision
    # Decimal context.
    price_lo, price_hi = (
        Decimal(f"{bound}e-2") for bound in settings.uncertainty_price_range_cents
    )
    buckets: dict[str, list[tuple[Decimal, dict]]] = {}
    for payload in candidates:
        # Fail closed on absent/unknown categories, including cross-listings
        # outside the allowlist. Category metadata is selection-only.
        categories = payload.get("categories") if isinstance(payload, Mapping) else None
        if (
            not isinstance(categories, (list, tuple))
            or not categories
            or any(
                not isinstance(category, str)
                or category not in settings.allowed_market_categories
                for category in categories
            )
        ):
            continue
        try:
            row = market_row(payload, observed_at)
            opened = _timestamp(payload.get("open_time"), "open_time")
            # A malformed supplied fixed-point value must not fall back to a
            # potentially different legacy observation.
            volume = _decimal(
                payload.get("volume_24h_fp", payload.get("volume_24h")),
                "volume_24h",
            )
            price = _decimal(payload.get("last_price_dollars"), "last_price_dollars")
        except ValueError:
            continue
        expiration = min(
            (value for value in (
                row["expected_expiration_time"], row["latest_expiration_time"],
            ) if value is not None),
            default=None,
        )
        if (
            row["status"] == "active"
            and opened <= observed_at < row["close_time"]
            and expiration is not None
            and earliest <= expiration <= latest
            and volume >= settings.volume_24h_floor
            and price_lo <= price <= price_hi
        ):
            # categories[0] is Kalshi's primary category. Bucketing by it alone
            # keeps an allowed cross-listing to one turn in the rotation.
            buckets.setdefault(categories[0], []).append((volume, row))

    # Stable sorts avoid negating Decimal values, which can round unusually
    # precise source volumes under the current decimal arithmetic context.
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item[1]["ticker"])
        bucket.sort(key=lambda item: item[0], reverse=True)

    # Configuration order is the rotation order, so the frozen settings record
    # which sector outranks which. Exhausted sectors drop out of later passes.
    rotation = [
        category
        for category in settings.allowed_market_categories
        if category in buckets
    ]
    cursors = dict.fromkeys(rotation, 0)
    selected: list[dict] = []
    tickers: set[str] = set()
    events: set[str] = set()
    while len(selected) < settings.market_count:
        progressed = False
        for category in rotation:
            if len(selected) >= settings.market_count:
                break
            bucket, index = buckets[category], cursors[category]
            while index < len(bucket):
                row = bucket[index][1]
                index += 1
                # Passing over a duplicate event does not cost this sector its
                # turn; it simply takes the next market it is still allowed.
                if row["ticker"] in tickers or row["event_ticker"] in events:
                    continue
                tickers.add(row["ticker"])
                events.add(row["event_ticker"])
                selected.append(row)
                progressed = True
                break
            cursors[category] = index
        if not progressed:
            break
    if len(selected) < settings.market_count:
        raise ValueError(
            f"only {len(selected)} eligible markets across distinct events; "
            f"need {settings.market_count}"
        )
    return selected
