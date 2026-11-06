"""situational (claude-assistant-xz59.13.8 §2.1), run against both recorded backends."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.events import NO_EVENTS
from graph_rag.facts import FactRecord, Situation
from cosmos_fake import FakeContainer          # tests/ is on sys.path
from recording_events import RecordingEvents

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
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


def _mem(fact_id: str, *, situation: Situation | None = EMAIL, kind: str = "instruction", **fields) -> FactRecord:
    return FactRecord(fact_id=fact_id, body=f"memory {fact_id}", created_at=NOW, updated_at=NOW, kind=kind,
                      situation=situation, vector_embeddings=(1.0, 0.0), **fields)


async def _seed(bucket: Bucket, *records: FactRecord) -> None:
    for record in records:
        await bucket.backend.upsert_fact(record)


async def _ids(bucket: Bucket, *args, **kwargs) -> list[str]:
    return [r.fact_id for r in await bucket.backend.situational(*args, **kwargs)]


async def test_only_current_memories_whose_situation_applies_are_returned(bucket):   # AC-R1
    await _seed(
        bucket,
        _mem("s-email"),
        _mem("s-chat", situation=Situation(("chat",))),
        _mem("s-type", situation=Situation(("email",), types=("GENERAL",))),
        _mem("s-type-other", situation=Situation(("email",), types=("ORDERS",))),
        _mem("s-entity", situation=Situation(("email",), entities=("ent-1",))),
        _mem("s-entity-other", situation=Situation(("email",), entities=("ent-2",))),
        _mem("s-topic", situation=Situation(("email",), topics=("meals",))),
        _mem("s-deleted", deleted=True),
        _mem("s-forgotten", forgotten_at=NOW),
        _mem("s-purged", purged_at=NOW),
        _mem("s-expired", valid_until=NOW - timedelta(days=1)),
        _mem("s-future", valid_from=NOW + timedelta(days=1)),
        _mem("u-instruction", situation=None),
        _mem("u-fact", situation=None, kind="fact"),
    )
    assert await _ids(bucket, "email", ["GENERAL"], ["ent-1", "ent-9"]) == ["s-email", "s-entity", "s-type"]
    assert await _ids(bucket, "email", ["GENERAL"], ["ent-1", "ent-9"], unsituated_kinds=("instruction",)) == [
        "s-email", "s-entity", "s-type", "u-instruction"]
    assert await _ids(bucket, "email", [], []) == ["s-email"]   # typed and entity situations need a shared value
    assert await _ids(bucket, "chat", [], [], unsituated_kinds=("fact", "instruction")) == [
        "s-chat", "u-fact", "u-instruction"]
    assert bucket.events.events == []


async def test_order_is_constraint_first_then_learned_at_as_stored_then_id(bucket):   # AC-R1
    await _seed(
        bucket,
        _mem("a", kind="fact", constraint=True, learned_at=datetime(2026, 6, 1, 10, 0, tzinfo=UTC)),
        _mem("b"),
        _mem("c", kind="fact", constraint=True),
        _mem("d", learned_at=datetime(2026, 6, 1, 9, 30, tzinfo=UTC)),
        # 09:00 UTC, earlier than d, but stored as "2026-06-01T10:00:00+01:00", which sorts after d's string
        _mem("e", learned_at=datetime(2026, 6, 1, 10, 0, tzinfo=BST)),
        _mem("f"),
    )
    assert await _ids(bucket, "email", [], []) == ["c", "a", "b", "f", "d", "e"]


async def test_a_limit_keeps_the_first_rows_and_counts_once_per_read(bucket):   # AC-R1
    await _seed(bucket, *[_mem(f"s{i}") for i in range(7)], *[_mem(f"u{i}", situation=None) for i in range(3)])
    assert await _ids(bucket, "email", [], [], unsituated_kinds=("instruction",), limit=5) == [
        "s0", "s1", "s2", "s3", "s4", "u0", "u1", "u2"]
    assert bucket.events.events == [("situational.limit_reached", {"query": "situated", "limit": 5})]
    bucket.events.events.clear()
    assert await _ids(bucket, "email", [], [], unsituated_kinds=("instruction",), limit=2) == ["s0", "s1", "u0", "u1"]
    assert bucket.events.events == [("situational.limit_reached", {"query": "situated", "limit": 2}),
                                    ("situational.limit_reached", {"query": "unsituated", "limit": 2})]


@pytest.mark.parametrize("matches", [4, 5, 6])
async def test_a_read_at_its_limit_raises_no_event(bucket, matches):
    await _seed(bucket, *[_mem(f"s{i}") for i in range(matches)])
    assert len(await _ids(bucket, "email", [], [], limit=5)) == min(matches, 5)
    assert bucket.events.events == ([] if matches <= 5 else
                                    [("situational.limit_reached", {"query": "situated", "limit": 5})])


async def test_no_returned_record_carries_vectors(bucket):   # AC-R1
    await _seed(bucket, _mem("s1"), _mem("u1", situation=None))
    records = await bucket.backend.situational("email", [], [], unsituated_kinds=("instruction",))
    assert [r.fact_id for r in records] == ["s1", "u1"]
    assert all(r.vector_embeddings is None for r in records)
    assert records[0].situation == EMAIL


async def test_unsituated_reads_take_current_memories_only(bucket):   # AC-R1
    await _seed(
        bucket,
        _mem("u-live", situation=None),
        _mem("u-deleted", situation=None, deleted=True),
        _mem("u-forgotten", situation=None, forgotten_at=NOW),
        _mem("u-purged", situation=None, purged_at=NOW),
        _mem("u-expired", situation=None, valid_until=NOW - timedelta(days=1)),
        _mem("u-future", situation=None, valid_from=NOW + timedelta(days=1)),
    )
    assert await _ids(bucket, "email", [], [], unsituated_kinds=("instruction",)) == ["u-live"]


@pytest.mark.parametrize("kind", ["memory", "cosmos"])
async def test_now_is_the_backends_clock(kind):   # §2.1: "now" is the backend's clock
    later = datetime(2031, 1, 1, tzinfo=UTC)
    backend = (InMemoryFactBackend(clock=lambda: later) if kind == "memory" else
               CosmosFactBackend(container=FakeContainer(), family_id="fam1", clock=lambda: later))
    await backend.upsert_fact(_mem("ended", valid_until=later - timedelta(days=1)))
    await backend.upsert_fact(_mem("started", valid_from=later - timedelta(days=1)))
    assert [r.fact_id for r in await backend.situational("email", [], [])] == ["started"]


@pytest.mark.parametrize("args, kwargs", [
    (("", [], []), {}),                                                   # no channel
    (("email", [], []), {"limit": 0}),
    (("email", [], []), {"limit": 101}),
    (("email", [f"t{i}" for i in range(201)], []), {}),
    (("email", [], [f"e{i}" for i in range(201)]), {}),
], ids=["no-channel", "limit-0", "limit-101", "types-201", "entities-201"])
async def test_situational_bounds_are_checked_before_any_read(bucket, args, kwargs):   # AC-S1-5
    if bucket.container is not None:
        bucket.container.query_items = _no_read
    with pytest.raises(ValueError):
        await bucket.backend.situational(*args, **kwargs)


async def test_bounds_count_distinct_values_and_include_their_limits(bucket):   # AC-S1-5 thresholds
    await _seed(bucket, _mem("s1"))
    types = [f"t{i}" for i in range(200)] + ["t0"] * 50
    assert await _ids(bucket, "email", types, [f"e{i}" for i in range(200)], limit=100) == ["s1"]
    assert await _ids(bucket, "email", [], [], limit=1) == ["s1"]


def _no_read(*args, **kwargs):
    raise AssertionError("a refused call must not read")


def test_no_events_does_nothing():
    assert NO_EVENTS.count("situational.limit_reached", query="situated", limit=5) is None


class _Spy(FakeContainer):
    def __init__(self) -> None:
        super().__init__()
        self.queries: list[tuple[str, dict, list]] = []

    def query_items(self, query, parameters=None, partition_key=None):
        self.queries.append((query, {p["name"]: p["value"] for p in parameters or []}, partition_key))
        return super().query_items(query, parameters=parameters, partition_key=partition_key)


async def test_cosmos_situational_reads_named_fields_in_the_indexed_order():
    container = _Spy()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    await backend.situational("email", ["GENERAL"], ["ent-1"], limit=7)
    [(query, params, pk)] = container.queries
    select = query.split(" FROM c ", 1)[0]
    assert select.startswith("SELECT TOP @n c.id, c.fact, ") and "vector_embeddings" not in select
    assert query.endswith(" ORDER BY c.constraint DESC, c.learned_at ASC, c.id ASC")
    assert (params["@n"], params["@channel"], params["@types"], params["@entity_ids"]) == (8, "email", ["GENERAL"],
                                                                                          ["ent-1"])
    assert pk == ["fam1", "shared", "FACTS"]
    await backend.situational("email", [], [], unsituated_kinds=("instruction", "instruction"))
    unsituated_query, unsituated_params, _ = container.queries[-1]
    assert "(NOT IS_DEFINED(c.situation) OR IS_NULL(c.situation))" in unsituated_query
    assert unsituated_params["@kinds"] == ["instruction"] and len(container.queries) == 3


async def test_cosmos_situational_sends_types_and_entity_ids_distinct_and_sorted():
    container = _Spy()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    await backend.situational("email", ["ORDERS", "GENERAL", "ORDERS"], ["ent-2", "ent-1", "ent-2"])
    [(_, params, _)] = container.queries
    assert (params["@types"], params["@entity_ids"]) == (["GENERAL", "ORDERS"], ["ent-1", "ent-2"])


async def test_cosmos_a_null_situation_reads_as_unsituated():   # §2.1: no situation is missing or null
    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    await backend.upsert_fact(_mem("u-null", situation=None))
    container.docs[("fam1", "shared", "FACTS", "u-null")]["situation"] = None
    assert [r.fact_id for r in await backend.situational("email", [], [], unsituated_kinds=("instruction",))] == [
        "u-null"]


async def test_cosmos_situational_sends_values_as_parameters_never_in_the_query():   # trust boundary row 1
    container = _Spy()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    await backend.situational("em'ail", ["GEN\"ERAL"], ["ent-'1"], unsituated_kinds=("instr'uction",))
    for query, params, _ in container.queries:
        assert "em'ail" not in query and 'GEN"ERAL' not in query and "ent-'1" not in query
        assert "instr'uction" not in query
