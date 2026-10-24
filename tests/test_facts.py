from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.facts import (
    FactRecord, FactSearchRequest, Holder, cosine_distance, fact_state, hit_from_record, record_from_state,
)


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


T0 = datetime(2026, 10, 18, 9, 0, tzinfo=UTC)


def _full_record() -> FactRecord:
    return FactRecord(
        fact_id="m1", body="Bob must not eat peanuts", created_at=T0, updated_at=T0, confidence=0.95,
        kind="fact", holder=Holder("parent", "parent1"), lifespan="lasting",
        valid_from=T0, valid_until=T0 + timedelta(days=30), validity_basis="ended",
        forgotten_at=None, purged_at=None, learned_at=T0, last_changed_at=T0, last_confirmed_at=T0,
        about=("person-child2",), mentions=("activity-1",), topics=("meals",), seq=3, extras={"constraint": True},
    )


def test_state_round_trips_every_recorded_field():
    record = _full_record()
    assert record_from_state(fact_state(record), now=T0) == record


def test_old_style_record_state_has_no_new_keys():
    state = fact_state(FactRecord(fact_id="o1", body="likes pasta", created_at=T0, updated_at=T0))
    assert set(state) == {"id", "fact", "created_at", "last_updated_at", "deleted"}


def test_old_document_reads_with_seq_zero_and_empty_lists():
    record = record_from_state({"id": "o1", "fact": "likes pasta", "created_at": T0.isoformat()}, now=T0)
    assert (record.seq, record.about, record.mentions, record.topics, record.kind, record.holder) == (
        0, (), (), (), None, None)


def test_extras_value_survives_when_the_typed_field_is_unset_and_loses_when_set():
    via_extras = FactRecord(fact_id="e1", body="b", created_at=T0, updated_at=T0, extras={"kind": "opinion"})
    assert fact_state(via_extras)["kind"] == "opinion"
    typed = FactRecord(fact_id="e1", body="b", created_at=T0, updated_at=T0, kind="fact", extras={"kind": "opinion"})
    assert fact_state(typed)["kind"] == "fact"


def test_holder_without_ref_has_no_ref_key():
    state = fact_state(FactRecord(fact_id="h1", body="b", created_at=T0, updated_at=T0, holder=Holder("sender")))
    assert state["holder"] == {"kind": "sender"}


def test_search_hit_carries_recorded_fields():
    hit = hit_from_record(_full_record(), 0.25)
    assert (hit.kind, hit.holder, hit.lifespan, hit.topics, hit.about, hit.mentions, hit.seq, hit.score) == (
        "fact", Holder("parent", "parent1"), "lasting", ("meals",), ("person-child2",), ("activity-1",), 3, 0.25)
    assert hit.extras == {"constraint": True}


def test_about_and_mentions_are_two_stored_lists_and_entities_is_gone():
    state = fact_state(_full_record())
    assert (state["about"], state["mentions"]) == (["person-child2"], ["activity-1"])
    assert "entities" not in state


def test_a_document_still_holding_entities_reads_it_as_an_extra():   # no compatibility layer (spec §7)
    record = record_from_state({"id": "o1", "fact": "b", "entities": ["x"]}, now=T0)
    assert (record.about, record.mentions, record.extras) == ((), (), {"entities": ["x"]})
