# graph-rag/tests/test_clock.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from graph_rag.clock import read_clock, utc_now


def test_utc_now_is_aware_utc():
    now = utc_now()
    assert now.utcoffset() == timedelta(0)


def test_read_clock_returns_the_clock_value():
    fixed = datetime(2031, 5, 4, 12, 0, tzinfo=UTC)
    assert read_clock(lambda: fixed) is fixed


def test_read_clock_refuses_a_naive_datetime():
    with pytest.raises(ValueError, match="clock must return an aware datetime"):
        read_clock(lambda: datetime(2031, 5, 4, 12, 0))


def test_read_clock_accepts_an_aware_non_utc_time():
    bst = datetime(2031, 5, 4, 13, 0, tzinfo=timezone(timedelta(hours=1)))
    assert read_clock(lambda: bst) == bst
