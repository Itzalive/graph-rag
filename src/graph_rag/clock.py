# graph-rag/src/graph_rag/clock.py
"""The clock fact backends read the current time from. Inject a fixed or moving clock to run them at a chosen time."""
from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

Clock = Callable[[], datetime]


def utc_now() -> datetime:
    return datetime.now(UTC)


def read_clock(clock: Clock) -> datetime:
    """The clock's time; a naive datetime is refused, since stored times are compared as aware UTC."""
    value = clock()
    if value.utcoffset() is None:
        raise ValueError("clock must return an aware datetime")
    return value
