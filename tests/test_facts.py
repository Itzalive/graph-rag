from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.facts import FactRecord, FactSearchRequest, cosine_distance


def _record(
    fact_id: str,
    body: str,
    *,
    deleted: bool = False,
    valid_until: datetime | None = None,
    vector: tuple[float, ...] | None = None,
) -> FactRecord:
    now = datetime.now(UTC)
    return FactRecord(
        fact_id=fact_id,
        body=body,
        created_at=now,
        updated_at=now,
        deleted=deleted,
        valid_until=valid_until,
        vector_embeddings=vector,
    )


# --- InMemoryFactBackend ---


async def test_upsert_and_read_roundtrip():
    backend = InMemoryFactBackend()
    rec = _record("f1", "child likes blueberries")
    await backend.upsert_fact(rec)
    result = await backend.read_fact("f1")
    assert result == rec


async def test_read_missing_returns_none():
    backend = InMemoryFactBackend()
    assert await backend.read_fact("nope") is None


async def test_upsert_overwrites_existing():
    backend = InMemoryFactBackend()
    await backend.upsert_fact(_record("f1", "old body"))
    updated = _record("f1", "new body")
    await backend.upsert_fact(updated)
    assert (await backend.read_fact("f1")).body == "new body"


async def test_list_facts_excludes_deleted():
    backend = InMemoryFactBackend()
    await backend.upsert_fact(_record("keep", "alive"))
    await backend.upsert_fact(_record("gone", "deleted", deleted=True))
    results = await backend.list_facts()
    assert [r.fact_id for r in results] == ["keep"]


async def test_list_facts_excludes_expired_by_default():
    past = datetime.now(UTC) - timedelta(hours=1)
    backend = InMemoryFactBackend()
    await backend.upsert_fact(_record("expired", "old", valid_until=past))
    await backend.upsert_fact(_record("live", "current"))
    results = await backend.list_facts()
    assert [r.fact_id for r in results] == ["live"]


async def test_list_facts_includes_expired_when_requested():
    past = datetime.now(UTC) - timedelta(hours=1)
    backend = InMemoryFactBackend()
    await backend.upsert_fact(_record("expired", "old", valid_until=past))
    results = await backend.list_facts(include_expired=True)
    assert len(results) == 1
    assert results[0].fact_id == "expired"


async def test_list_facts_does_not_include_deleted_even_with_include_expired():
    past = datetime.now(UTC) - timedelta(hours=1)
    backend = InMemoryFactBackend()
    await backend.upsert_fact(_record("gone", "deleted+expired", deleted=True, valid_until=past))
    results = await backend.list_facts(include_expired=True)
    assert results == []


def test_search_request_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="positive"):
        FactSearchRequest(query_vector=(0.1, 0.2), limit=0)


async def test_search_facts_rejects_limit_over_100():
    backend = InMemoryFactBackend()
    with pytest.raises(ValueError, match="100"):
        await backend.search_facts(FactSearchRequest(query_vector=(1.0,), limit=101))


async def test_search_facts_returns_closest_vector_first():
    backend = InMemoryFactBackend()
    await backend.upsert_fact(_record("far", "far fact", vector=(0.0, 1.0)))
    await backend.upsert_fact(_record("near", "near fact", vector=(1.0, 0.0)))
    hits = await backend.search_facts(
        FactSearchRequest(query_vector=(1.0, 0.0), limit=10)
    )
    assert [h.fact_id for h in hits] == ["near", "far"]
    assert hits[0].score < hits[1].score
    assert hits[0].score == pytest.approx(0.0)
    assert hits[1].score == pytest.approx(1.0)


async def test_search_facts_excludes_deleted():
    backend = InMemoryFactBackend()
    await backend.upsert_fact(_record("gone", "deleted", deleted=True, vector=(1.0, 0.0)))
    hits = await backend.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=10))
    assert hits == []


async def test_search_facts_respects_exclude_ids():
    backend = InMemoryFactBackend()
    await backend.upsert_fact(_record("f1", "match", vector=(1.0, 0.0)))
    await backend.upsert_fact(_record("f2", "also match", vector=(1.0, 0.0)))
    hits = await backend.search_facts(
        FactSearchRequest(query_vector=(1.0, 0.0), limit=10, exclude_ids=("f1",))
    )
    assert [h.fact_id for h in hits] == ["f2"]


async def test_search_facts_skips_records_without_embeddings():
    backend = InMemoryFactBackend()
    await backend.upsert_fact(_record("no-vec", "no vector"))
    await backend.upsert_fact(_record("vec", "has vector", vector=(1.0, 0.0)))
    hits = await backend.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=10))
    assert [h.fact_id for h in hits] == ["vec"]


def test_cosine_distance_rejects_mismatched_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        cosine_distance((1.0, 0.0), (1.0,))
