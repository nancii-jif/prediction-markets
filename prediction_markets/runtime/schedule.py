"""Shared calendar policy; elapsed tool costs never pause outside a session."""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Session:
    opens_at: int
    closes_at: int

    def contains(self, timestamp: float) -> bool:
        return self.opens_at <= timestamp < self.closes_at


class TradingSchedule:
    """One daily session, including weekends, in the configured local timezone."""

    def __init__(self, timezone: str, start: str, end: str):
        self.timezone = ZoneInfo(timezone)
        self.start = time.fromisoformat(start)
        self.end = time.fromisoformat(end)

    def session_at(self, timestamp: float) -> Session:
        day = datetime.fromtimestamp(timestamp, self.timezone).date()
        return Session(
            int(datetime.combine(day, self.start, self.timezone).timestamp()),
            int(datetime.combine(day, self.end, self.timezone).timestamp()),
        )

    def next_day_open(self, timestamp: float) -> int:
        day = datetime.fromtimestamp(timestamp, self.timezone).date() + timedelta(days=1)
        return int(datetime.combine(day, self.start, self.timezone).timestamp())

    def open_at_or_after(self, timestamp: float) -> float:
        session = self.session_at(timestamp)
        if timestamp < session.opens_at:
            return session.opens_at
        if timestamp < session.closes_at:
            return timestamp
        return self.next_day_open(timestamp)
