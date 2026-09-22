"""Pure purchase-style accounting for YES-space contracts.

Cash is the participant's balance; reservations are computed separately and
never debited from it. A negative YES position is a fully purchased
NO-equivalent position, so opening it costs ``100 - p``; there is no
short-sale credit and no second collateral balance.
"""

from __future__ import annotations

CONTRACT_PAYOUT_CENTS = 100
MIN_PRICE_CENTS = 1
MAX_PRICE_CENTS = 99


def _integer(name: str, value: int, *, minimum: int | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def after_fill(
    cash_cents: int,
    position_qty: int,
    signed_trade_qty: int,
    price_cents: int,
) -> tuple[int, int]:
    """Return ``(cash, position)`` after one exact integer fill.

    ``signed_trade_qty`` is positive for a YES buy and negative for a YES
    sell. The formula handles adding, closing, and crossing through zero.
    """
    _integer("cash_cents", cash_cents, minimum=0)
    _integer("position_qty", position_qty)
    _integer("signed_trade_qty", signed_trade_qty)
    _integer("price_cents", price_cents)
    if not MIN_PRICE_CENTS <= price_cents <= MAX_PRICE_CENTS:
        raise ValueError("price_cents must be between 1 and 99")

    next_position = position_qty + signed_trade_qty
    short_before = max(-position_qty, 0)
    short_after = max(-next_position, 0)
    next_cash = (
        cash_cents
        - price_cents * signed_trade_qty
        + CONTRACT_PAYOUT_CENTS * (short_before - short_after)
    )
    return next_cash, next_position


def reservation_cents(
    position_qty: int,
    buy_exposure_cents: int,
    sell_exposure_cents: int,
) -> int:
    """Worst additional cash draw for one market's remaining orders.

    Buy exposure is ``sum(price * remaining)``. Sell exposure is
    ``sum((100 - price) * remaining)``. Opposite sides are alternatives in
    the worst case, so they are combined with ``max`` rather than addition.
    """
    _integer("position_qty", position_qty)
    _integer("buy_exposure_cents", buy_exposure_cents, minimum=0)
    _integer("sell_exposure_cents", sell_exposure_cents, minimum=0)

    reserved = max(
        buy_exposure_cents,
        sell_exposure_cents - CONTRACT_PAYOUT_CENTS * position_qty,
    ) - CONTRACT_PAYOUT_CENTS * max(-position_qty, 0)
    if reserved < 0:
        raise RuntimeError("reservation formula produced a negative value")
    return reserved


def position_value_cents(position_qty: int, mark_cents: int) -> int:
    """Mark a purchased YES or NO-equivalent position in integer cents."""
    _integer("position_qty", position_qty)
    _integer("mark_cents", mark_cents)
    if not MIN_PRICE_CENTS <= mark_cents <= MAX_PRICE_CENTS:
        raise ValueError("mark_cents must be between 1 and 99")
    if position_qty >= 0:
        return position_qty * mark_cents
    return -position_qty * (CONTRACT_PAYOUT_CENTS - mark_cents)


def settlement_credit_cents(position_qty: int, payout_cents: int) -> int:
    """Final redemption of purchased YES or NO-equivalent holdings."""
    _integer("position_qty", position_qty)
    _integer("payout_cents", payout_cents)
    if not 0 <= payout_cents <= CONTRACT_PAYOUT_CENTS:
        raise ValueError("payout_cents must be between 0 and 100")
    return (
        max(position_qty, 0) * payout_cents
        + max(-position_qty, 0) * (CONTRACT_PAYOUT_CENTS - payout_cents)
    )
