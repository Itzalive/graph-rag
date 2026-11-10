"""by_kind (claude-assistant-xz59.13.10 S3a spec §3.1), run against the recorded backends."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.facts import FactRecord, Situation
from cosmos_fake import FakeContainer          # tests/ is on sys.path
from recording_events import RecordingEvents

NOW = datetime(2030, 3, 1, 12, 0, tzinfo=UTC)   # far from the real date: a read of the wall clock is caught
BST = timezone(timedelta(hours=1))
EMAIL = Situation(("email",))


@dataclass
class Bucket:
    backend: object
    events: RecordingEvents
    container: FakeContainer | None


@pytest.fixture(params=["memory", "cosmos"])
def bucket(request) -> Bucket:
    events = RecordingEvents()
    if request.param == "memory":
        return Bucket(InMemoryFactBackend(clock=lambda: NOW, events=events), events, None)
    container = FakeContainer()
    return Bucket(CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW, events=events),
                  events, container)


def _mem(fact_id: str, *, kind: str | None = "instruction", situation: Situation | None = EMAIL,
         **fields) -> FactRecord:
    return FactRecord(fact_id=fact_id, body=f"memory {fact_id}", created_at=NOW, updated_at=NOW, kind=kind,
                      situation=situation, vector_embeddings=(1.0, 0.0), **fields)


async def _seed(bucket: Bucket, *records: FactRecord) -> None:
    for record in records:
        await bucket.backend.upsert_fact(record)


async def _ids(bucket: Bucket, *args, **kwargs) -> list[str]:
    return [r.fact_id for r in await bucket.backend.by_kind(*args, **kwargs)]


async def test_current_memories_of_the_kinds_and_constraints_are_returned(bucket):   # AC-S3a-14
    await _seed(
        bucket,
        _mem("i-situated"),
        _mem("i-unsituated", situation=None),
        _mem("k-fact", kind="fact", constraint=True, situation=None),
        _mem("f-plain", kind="fact", situation=None),
        _mem("o-opinion", kind="opinion", situation=None),
        _mem("n-no-kind", kind=None, situation=None),
    )
    assert await _ids(bucket, ("instruction",), constraints=True) == ["k-fact", "i-situated", "i-unsituated"]
    assert await _ids(bucket, (), constraints=True) == ["k-fact"]
    assert await _ids(bucket, ("fact", "opinion")) == ["k-fact", "f-plain", "o-opinion"]   # a constraint sorts first
    assert bucket.events.events == []


async def test_without_constraints_a_constraint_of_another_kind_is_left_out(bucket):
    await _seed(bucket, _mem("i-1"), _mem("k-fact", kind="fact", constraint=True, situation=None))
    assert await _ids(bucket, ("instruction",)) == ["i-1"]
    assert await _ids(bucket, ("instruction",), constraints=False) == ["i-1"]


async def test_ended_and_future_memories_are_left_out(bucket):   # AC-S3a-14: current only
    await _seed(
        bucket,
        _mem("live"),
        _mem("deleted", deleted=True),
        _mem("forgotten", forgotten_at=NOW),
        _mem("purged", purged_at=NOW),
        _mem("expired", valid_until=NOW - timedelta(days=1)),
        _mem("future", valid_from=NOW + timedelta(days=1)),
        _mem("k-expired", kind="fact", constraint=True, valid_until=NOW - timedelta(days=1)),
    )
    assert await _ids(bucket, ("instruction",), constraints=True) == ["live"]


async def test_order_is_constraint_first_then_learned_at_as_stored_then_id(bucket):   # §3.1: situational's order
    await _seed(
        bucket,
        _mem("a", kind="fact", constraint=True, learned_at=datetime(2026, 6, 1, 10, 0, tzinfo=UTC)),
        _mem("b"),
        _mem("c", kind="fact", constraint=True),
        _mem("d", learned_at=datetime(2026, 6, 1, 9, 30, tzinfo=UTC)),
        _mem("e", learned_at=datetime(2026, 6, 1, 10, 0, tzinfo=BST)),   # sorts after d as stored
        _mem("f"),
    )
    assert await _ids(bucket, ("instruction",), constraints=True) == ["c", "a", "b", "f", "d", "e"]


async def test_a_limit_keeps_the_first_rows_and_counts_once(bucket):   # AC-S3a-14: more than limit
    await _seed(bucket, *[_mem(f"i{i}") for i in range(7)])
    assert await _ids(bucket, ("instruction",), limit=5) == ["i0", "i1", "i2", "i3", "i4"]
    assert bucket.events.events == [("situational.limit_reached", {"query": "by_kind", "limit": 5})]
    bucket.events.events.clear()
    assert len(await _ids(bucket, ("instruction",), limit=7)) == 7
    assert bucket.events.events == []


async def test_no_returned_record_carries_vectors(bucket):
    await _seed(bucket, _mem("i1"))
    [record] = await bucket.backend.by_kind(("instruction",))
    assert record.vector_embeddings is None and record.situation == EMAIL


@pytest.mark.parametrize("args, kwargs, error", [
    ((("instruction",),), {"limit": 0}, ValueError),
    ((("instruction",),), {"limit": 101}, ValueError),
    (([f"k{i}" for i in range(21)],), {}, ValueError),
    (((),), {}, ValueError),                                  # no kinds and no constraints could return nothing
    (((),), {"constraints": False}, ValueError),
    (("instruction",), {}, TypeError),                        # a string reads as its characters
], ids=["limit-0", "limit-101", "kinds-21", "nothing", "nothing-explicit", "string"])
async def test_by_kind_bounds_are_checked_before_any_read(bucket, args, kwargs, error):
    if bucket.container is not None:
        bucket.container.query_items = _no_read
    with pytest.raises(error):
        await bucket.backend.by_kind(*args, **kwargs)


async def test_twenty_kinds_pass_and_repeats_count_once(bucket):   # the thresholds
    await _seed(bucket, _mem("i1"))
    kinds = ["instruction", *[f"k{i}" for i in range(19)], "instruction", "k0"]
    assert await _ids(bucket, kinds, limit=1) == ["i1"]
    assert await _ids(bucket, (), constraints=True, limit=100) == []


def _no_read(*args, **kwargs):
    raise AssertionError("a refused call must not read")


class _Spy(FakeContainer):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[tuple[str, dict, list]] = []

    def query_items(self, query, parameters=None, partition_key=None):
        self.queries.append((query, {p["name"]: p["value"] for p in parameters or []}, partition_key))
        return super().query_items(query, parameters=parameters, partition_key=partition_key)


async def test_cosmos_by_kind_reads_named_fields_in_the_indexed_order():   # §3.1: one query, named fields
    container = _Spy()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    await backend.by_kind(("instruction", "instruction"), constraints=True, limit=7)
    [(query, params, pk)] = container.queries
    select = query.split(" FROM c ", 1)[0]
    assert select.startswith("SELECT TOP @n c.id, c.fact, ") and "vector_embeddings" not in select
    assert query.endswith(" ORDER BY c.constraint DESC, c.learned_at ASC, c.id ASC")
    assert "(ARRAY_CONTAINS(@kinds, c.kind) OR c.constraint = true)" in query
    assert (params["@n"], params["@kinds"], pk) == (8, ["instruction"], ["fam1", "shared", "FACTS"])
    await backend.by_kind(("fact",))
    plain, _, _ = container.queries[-1]
    assert "ARRAY_CONTAINS(@kinds, c.kind)" in plain and "c.constraint = true" not in plain


async def test_cosmos_by_kind_sends_kinds_as_a_parameter_never_in_the_query():
    container = _Spy()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    await backend.by_kind(("instr'uction",), constraints=True)
    [(query, params, _)] = container.queries
    assert "instr'uction" not in query and params["@kinds"] == ["instr'uction"]
