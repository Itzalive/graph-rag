# graph-rag/tests/test_clock.py
from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.clock import read_clock, utc_now
from graph_rag.facts import FactRecord, FactSearchRequest


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


FIXED = datetime(2040, 1, 15, 12, 0, tzinfo=UTC)   # years away from today: only the clock can decide expiry


def _fact(fact_id: str, valid_until: datetime | None, vector=(1.0, 0.0)) -> FactRecord:
    created = datetime(2039, 12, 1, tzinfo=UTC)
    return FactRecord(fact_id=fact_id, body=fact_id, created_at=created, updated_at=created,
                      valid_until=valid_until, vector_embeddings=vector)


async def _in_memory_with_expiry_cases(clock):
    backend = InMemoryFactBackend(clock=clock)
    await backend.upsert_fact(_fact("open", None))
    await backend.upsert_fact(_fact("at_now", FIXED))                       # valid_until == now: expired
    await backend.upsert_fact(_fact("past", FIXED - timedelta(days=1)))
    await backend.upsert_fact(_fact("future", FIXED + timedelta(days=1)))
    return backend


async def test_in_memory_list_facts_cuts_off_at_the_clock():
    backend = await _in_memory_with_expiry_cases(lambda: FIXED)
    assert sorted(r.fact_id for r in await backend.list_facts()) == ["future", "open"]


async def test_in_memory_search_facts_cuts_off_at_the_clock():
    backend = await _in_memory_with_expiry_cases(lambda: FIXED)
    hits = await backend.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=10))
    assert sorted(h.fact_id for h in hits) == ["future", "open"]


async def test_in_memory_naive_clock_raises_on_read():
    backend = await _in_memory_with_expiry_cases(lambda: datetime(2040, 1, 15, 12, 0))
    with pytest.raises(ValueError, match="clock must return an aware datetime"):
        await backend.list_facts()
    with pytest.raises(ValueError, match="clock must return an aware datetime"):
        await backend.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=10))


def test_in_memory_default_clock_is_utc_now():
    assert InMemoryFactBackend()._clock is utc_now


def _cosmos(clock):
    container = MagicMock()
    container.query_items.return_value = []
    return CosmosFactBackend(container=container, family_id="f", clock=clock), container


def _now_param(container) -> str:
    params = container.query_items.call_args.kwargs["parameters"]
    return next(p["value"] for p in params if p["name"] == "@now")


async def test_cosmos_list_facts_now_param_is_the_clock():
    backend, container = _cosmos(lambda: FIXED)
    await backend.list_facts()
    assert _now_param(container) == FIXED.isoformat()


async def test_cosmos_search_facts_now_param_is_the_clock():
    backend, container = _cosmos(lambda: FIXED)
    await backend.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=10))
    assert _now_param(container) == FIXED.isoformat()


async def test_cosmos_naive_clock_raises_unless_time_filters_are_widened():
    backend, container = _cosmos(lambda: datetime(2040, 1, 15, 12, 0))
    with pytest.raises(ValueError, match="clock must return an aware datetime"):
        await backend.list_facts()
    with pytest.raises(ValueError, match="clock must return an aware datetime"):
        await backend.list_facts(include_expired=True)   # the valid-from filter still reads the clock
    assert await backend.list_facts(include_expired=True, include_future=True) == []   # reads no clock


def test_cosmos_document_without_timestamps_falls_back_to_the_clock():
    backend, _ = _cosmos(lambda: FIXED)
    record = backend._from_document({"id": "x", "fact": "no timestamps"})
    assert record.created_at == FIXED
    assert record.updated_at == FIXED


def test_cosmos_default_clock_is_utc_now():
    assert CosmosFactBackend(container=MagicMock(), family_id="f")._clock is utc_now
