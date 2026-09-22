"""Which markets the internal exchange trades. Pure DB logic, no Kalshi access.

This module deliberately does NOT import kalshi_client. If the data it needs
looks stale or missing, the fix belongs in `stream.py` — never a direct API
call from here.

It reads Kalshi prices out of `snapshots` to *select* markets. Agents never
see any of it; the only thing that crosses into the agents' world is the fact
of membership itself.
"""

import logging
import time

from . import config, database

log = logging.getLogger(__name__)


def _mid_cents(snapshot) -> float | None:
    """Kalshi mid in cents, or None if either side of the book is empty."""
    bid, ask = snapshot["yes_bid"], snapshot["yes_ask"]
    if bid is None or ask is None:
        return None
    return (bid + ask) * 100.0 / 2.0


def qualifies(market, snapshot, now: int) -> tuple[bool, dict]:
    """Admission screen. Returns (ok, the stats that decided it).

    Every one of these is checked at ADD time only — see maintain() for why
    none of them is ever re-checked on a sitting member.
    """
    mid = _mid_cents(snapshot) if snapshot else None
    close_ts = market["close_ts"]
    stats = {
        "ticker": market["ticker"],
        "event_ticker": market["event_ticker"],
        "close_ts": close_ts,
        "hours_to_close": round((close_ts - now) / 3600, 1) if close_ts else None,
        "volume_24h": snapshot["volume_24h"] if snapshot else None,
        "mid_cents": round(mid, 1) if mid is not None else None,
        "status": snapshot["status"] if snapshot else None,
        "result": snapshot["result"] if snapshot else None,
    }

    if snapshot is None or close_ts is None:
        return False, stats
    if snapshot["result"]:
        return False, stats
    # An allowlist, not a denylist. Kalshi's response vocabulary includes
    # initialized (listed but not open), inactive (deactivated), determined
    # and finalized; only an actively trading market can be quoted, and an
    # allowlist stays correct if that vocabulary grows. Volume alone does not
    # separate these: a market can carry stale 24h volume and not be tradeable.
    if snapshot["status"] != "active":
        return False, stats
    if not (now + config.EXPIRY_MIN_H * 3600 <= close_ts <= now + config.EXPIRY_MAX_H * 3600):
        return False, stats
    volume = snapshot["volume_24h"]
    if volume is None or volume < config.VOLUME_FLOOR:
        return False, stats
    if mid is None or not (config.UNCERTAIN_LO <= mid <= config.UNCERTAIN_HI):
        return False, stats
    return True, stats


def top_k(sector: str, k: int, exclude: set[str] | None = None,
          now: int | None = None, exclude_events: set[str] | None = None):
    """The k best qualifying candidates in a sector, one per event.

    Ranked by 24h volume, ties broken lexicographically by ticker, then walked
    greedily so that no two picks share an event — the highest-volume market
    in an event is the one that represents it.

    Stratifying by event as well as sector matters because a Kalshi event
    typically lists many markets that are strikes on one question
    ("unemployment >= 4.0 / 4.1 / 4.2"). Seating several of them spends
    several seats on one question, and because those strikes are logically
    nested while agents quote each market independently, the exchange can
    print a set that contradicts itself. Deterministic given the DB: the
    ranking is total, so the greedy walk is too.
    """
    now = now or int(time.time())
    exclude = exclude or set()
    taken_events = set(exclude_events or set())

    candidates = [m for m in database.markets_in_sector(sector) if m["ticker"] not in exclude]
    if not candidates:
        return []
    snapshots = database.latest_snapshots([m["ticker"] for m in candidates])

    scored = []
    for market in candidates:
        ok, stats = qualifies(market, snapshots.get(market["ticker"]), now)
        if ok:
            scored.append((-stats["volume_24h"], market["ticker"], stats))
    scored.sort()

    picks = []
    for _, _, stats in scored:
        if len(picks) == k:
            break
        event = stats["event_ticker"]
        # A market with no recorded event cannot be shown to be distinct from
        # anything, so it is not admissible under event stratification.
        if event is None or event in taken_events:
            continue
        taken_events.add(event)
        picks.append(stats)
    return picks


def _admit(ticker: str, sector: str, now: int, stats: dict, reason: str) -> None:
    database.add_live_market(ticker, sector, now)
    log.info(
        "ADD    %-40s sector=%-12s event=%-32s reason=%-7s vol24h=%s mid=%sc "
        "status=%s close_in=%sh",
        ticker, sector, stats["event_ticker"], reason, stats["volume_24h"],
        stats["mid_cents"], stats["status"], stats["hours_to_close"],
    )


def deficit() -> dict[str, int]:
    """Open seats per sector. The single question refill is driven by.

    Retirement and refill are independent processes: nothing tells refill what
    was just retired, it only ever asks how many seats are short. That is what
    makes a seat left open by a failed sweep get filled on a later run without
    needing some unrelated retirement to trigger it, and it makes seeding and
    refilling the same operation.
    """
    counts = {
        sector: config.MARKETS_PER_SECTOR - len(database.active_in_sector(sector))
        for sector in config.SECTORS
    }
    return {sector: n for sector, n in counts.items() if n > 0}


def retire(now: int | None = None) -> list[str]:
    """Retire members that resolved or ran past close + grace.

    Retirement fires on two triggers and on nothing else:

      - a non-empty `result` — the market resolved; or
      - the clock: `now > close_ts + RETIRE_GRACE_H`. The backstop for
        members whose result never arrives (a deactivated market keeps an
        empty result forever) — the clock needs no status vocabulary to catch
        them. The grace period keeps the market quoted through the usual
        close-to-determination gap, and settlement never depends on
        membership anyway: a retired market someone still holds stays in the
        targeted fetch until its result arrives and pays.

    Admission criteria are checked at ADD time only: a member whose Kalshi
    mid drifts to 5c, or whose volume dries up, stays until a trigger fires.
    Two reasons, written down here so nobody later "fixes" it:

      (a) Drifting toward 0/100 IS the market converging — precisely the phase
          the internal exchange most needs to live through. Ejecting
          converging markets would select the resolution dynamics out of the
          data.
      (b) Membership changes are agent-visible events. Band-based retirement
          would broadcast Kalshi price state into the agents' world through
          the back door: admission leaks one static bit (in-band at entry),
          continuous eviction would leak a running signal. (The clock trigger
          leaks nothing: close time is already printed in every brief.)
    """
    now = now or int(time.time())
    retired = []

    active = database.active_live_markets()
    if not active:
        return retired

    snapshots = database.latest_snapshots([r["ticker"] for r in active])
    for row in active:
        snapshot = snapshots.get(row["ticker"])
        resolved = bool(snapshot and snapshot["result"])
        expired = (
            row["close_ts"] is not None
            and now > row["close_ts"] + config.RETIRE_GRACE_H * 3600
        )
        if not resolved and not expired:
            continue
        database.retire_live_market(row["ticker"], now)
        retired.append(row["ticker"])
        log.info(
            "RETIRE %-40s sector=%-12s trigger=%s result=%s "
            "settlement_value=%s status=%s",
            row["ticker"], row["sector"],
            "result" if resolved else "expired",
            snapshot["result"] if snapshot else None,
            snapshot["settlement_value"] if snapshot else None,
            snapshot["status"] if snapshot else None,
        )
    return retired


def refill(now: int | None = None) -> dict:
    """Top every sector back up to MARKETS_PER_SECTOR.

    Knows nothing about what was just retired — it reads the deficit and
    fills it, which is what makes seeding and refilling the same code path.
    Callers are expected to have swept first, so the candidate data this
    selects from is seconds old.
    """
    now = now or int(time.time())
    result = {"added": [], "unfilled": []}

    admitted = database.ever_admitted()
    # Checked against the seats that are live right now, not against every
    # event ever admitted: once a market resolves, another strike on the same
    # event is a legitimate candidate again. The constraint exists to stop two
    # strikes on one question being live *simultaneously*.
    live_events = database.live_event_tickers()

    for sector, short_by in deficit().items():
        # Never readmit: a returning market would carry internal history that
        # agents have already seen, so its "first" round would not be one.
        reason = "seed" if not any(t in admitted for t in
                                   (r["ticker"] for r in database.markets_in_sector(sector))
                                   ) else "refill"
        picks = top_k(sector, short_by, admitted, now, exclude_events=live_events)
        for stats in picks:
            _admit(stats["ticker"], sector, now, stats, reason)
            admitted.add(stats["ticker"])
            live_events.add(stats["event_ticker"])
            result["added"].append(stats["ticker"])

        if len(picks) < short_by:
            missing = short_by - len(picks)
            result["unfilled"].extend([sector] * missing)
            log.warning(
                "%d seat(s) in %s left open: no qualifying never-admitted "
                "candidate from an unrepresented event; will fill on the next "
                "successful sweep", missing, sector,
            )

    return result


def maintain(allow_refill: bool = True) -> dict:
    """One full membership cycle: retire, then refill.

    A convenience wrapper over the two independent processes. `stream` drives
    them separately so it can sweep for fresh candidates in between.
    """
    now = int(time.time())
    retired = retire(now)
    if allow_refill:
        return {"retired": retired, **refill(now)}

    unfilled = [sector for sector, n in deficit().items() for _ in range(n)]
    for sector, n in deficit().items():
        log.warning("%d seat(s) in %s left open: no fresh sweep to draw from",
                    n, sector)
    return {"retired": retired, "added": [], "unfilled": unfilled}
