"""The recorded-memory contract (claude-assistant-xz59.9 §2–§4), run against both adapters."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from azure.cosmos.exceptions import CosmosBatchOperationError

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.entities import EntityRecord, Identifier, MissingHandleError
from graph_rag.facts import FactRecord, FactSearchRequest, fact_state
from graph_rag.record import AlreadyAppliedError, ChangeSource, FactChange, StaleWriteError, WriteCheckError
from cosmos_fake import FakeContainer  # tests/ is on sys.path (no __init__.py; pytest prepend import mode)

NOW = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def _rec(fact_id: str = "m1", body: str = "Alice swims on Tuesdays", **fields) -> FactRecord:
    return FactRecord(fact_id=fact_id, body=body, created_at=NOW, updated_at=NOW, **fields)


@pytest.fixture(params=["memory", "cosmos"])
def make_backend(request):
    """A factory for backends sharing one store: for Cosmos one fake container, for memory one bucket per instance."""
    container = FakeContainer()

    def make(family_id: str = "fam1", scope: str = "shared"):
        if request.param == "memory":
            return InMemoryFactBackend(clock=lambda: NOW)
        return CosmosFactBackend(container=container, family_id=family_id, scope=scope, clock=lambda: NOW)

    make.kind = request.param
    make.container = container
    return make


async def test_two_families_never_see_each_others_memories(make_backend):   # AC-M6
    fam1, fam2 = make_backend("fam1"), make_backend("fam2")
    await fam1.upsert_fact(_rec(vector_embeddings=(1.0, 0.0)))
    assert await fam2.read_fact("m1") is None
    assert await fam2.list_facts() == []
    assert await fam2.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=5)) == []
    assert [r.fact_id for r in await fam1.list_facts()] == ["m1"]


async def test_cosmos_scope_names_the_bucket():
    container = FakeContainer()
    personal = CosmosFactBackend(container=container, family_id="fam1", scope="parent1", clock=lambda: NOW)
    await personal.upsert_fact(_rec())
    assert list(container.docs) == [("fam1", "parent1", "FACTS", "m1")]
    assert container.docs[("fam1", "parent1", "FACTS", "m1")]["is_shared"] is False
    shared = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    assert await shared.read_fact("m1") is None


def test_cosmos_scope_must_not_be_empty():
    with pytest.raises(ValueError, match="scope"):
        CosmosFactBackend(container=FakeContainer(), family_id="fam1", scope="")


async def _ids(backend, **flags) -> tuple[list[str], list[str]]:
    listed = sorted(r.fact_id for r in await backend.list_facts(**flags))
    request = FactSearchRequest(query_vector=(1.0, 0.0), limit=10, **flags)
    searched = sorted(h.fact_id for h in await backend.search_facts(request))
    return listed, searched


async def test_forgotten_and_purged_are_never_listed_or_searched(make_backend):   # AC-M5
    backend = make_backend()
    vec = (1.0, 0.0)
    await backend.upsert_fact(_rec("live", vector_embeddings=vec))
    await backend.upsert_fact(_rec("forgot", forgotten_at=NOW, vector_embeddings=vec))
    await backend.upsert_fact(_rec("purged", purged_at=NOW, vector_embeddings=vec))
    assert await _ids(backend) == (["live"], ["live"])
    assert await _ids(backend, include_expired=True, include_future=True) == (["live"], ["live"])


async def test_not_yet_valid_only_with_include_future(make_backend):   # AC-M5, AC-14
    backend = make_backend()
    vec = (1.0, 0.0)
    await backend.upsert_fact(_rec("now", valid_from=NOW - timedelta(days=1), validity_basis="stated",
                                   vector_embeddings=vec))
    await backend.upsert_fact(_rec("later", valid_from=NOW + timedelta(days=16), validity_basis="stated",
                                   vector_embeddings=vec))
    assert await _ids(backend) == (["now"], ["now"])
    assert await _ids(backend, include_future=True) == (["later", "now"], ["later", "now"])


async def test_past_valid_until_only_with_include_expired(make_backend):   # AC-M5
    backend = make_backend()
    vec = (1.0, 0.0)
    await backend.upsert_fact(_rec("ended", valid_until=NOW - timedelta(days=1), validity_basis="ended",
                                   vector_embeddings=vec))
    assert await _ids(backend) == ([], [])
    assert await _ids(backend, include_expired=True) == (["ended"], ["ended"])


async def test_cosmos_documents_with_a_doc_type_are_never_memories():   # AC-M5 (read_fact on a non-memory id)
    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    container.upsert_item({"id": "x:1", "doc_type": "fact_change", "fact": "copy", "family_id": "fam1",
                           "user_id": "shared", "category": "FACTS", "vector_embeddings": [1.0, 0.0]})
    assert await backend.read_fact("x:1") is None
    assert await _ids(backend, include_expired=True, include_future=True) == ([], [])


SRC = ChangeSource("parent_chat", actor="parent1", ref="chat:parent1:2026-10-18T09:00:00+00:00")


def _change(seq: int, op: str, *, kind: str = "changed", fact_id: str = "m1", after=None) -> FactChange:
    return FactChange(fact_id=fact_id, seq=seq, kind=kind, at=NOW, source=SRC, reason="changed in chat", op_id=op,
                      after=after)


async def _write(backend, seq: int, op: str, body: str = "Alice swims on Tuesdays", **fields):
    record = _rec(body=body, lifespan="fading", seq=seq, **fields)
    await backend.write_change(record, [_change(seq, op, kind="created" if seq == 1 else "changed")])
    return record


async def test_a_new_memory_has_seq_1_and_one_record(make_backend):   # AC-M1
    backend = make_backend()
    record = await _write(backend, 1, "op-a")
    assert (await backend.read_fact("m1")).seq == 1
    [change] = await backend.history("m1")
    assert (change.seq, change.kind, change.source, change.after) == (1, "created", SRC, fact_state(record))


async def test_two_writers_at_one_seq_one_wins(make_backend):   # AC-M2
    backend = make_backend()
    for seq in (1, 2, 3):
        await _write(backend, seq, f"op-{seq}")
    await _write(backend, 4, "writer-a", body="Alice swims on Wednesdays")
    with pytest.raises(StaleWriteError):
        await _write(backend, 4, "writer-b", body="Alice swims on Thursdays")
    history = await backend.history("m1")
    assert [(c.seq, c.op_id) for c in history if c.seq == 4] == [(4, "writer-a")]
    assert (await backend.read_fact("m1")).body == "Alice swims on Wednesdays"
    await _write(backend, 5, "writer-b", body="Alice swims on Thursdays")   # the refused request applies once more
    assert [(c.seq, c.op_id) for c in await backend.history("m1")][-1] == (5, "writer-b")


async def test_the_same_request_again_is_already_applied(make_backend):   # AC-M3
    backend = make_backend()
    await _write(backend, 1, "tool-1")
    with pytest.raises(AlreadyAppliedError):
        await _write(backend, 2, "tool-1", body="Alice swims on Fridays")
    assert (await backend.read_fact("m1")).body == "Alice swims on Tuesdays"
    assert [c.seq for c in await backend.history("m1")] == [1]


@pytest.mark.parametrize("fields, code", [
    (dict(lifespan="dated", seq=1), "W1"),
    (dict(lifespan="lasting", valid_until=NOW + timedelta(days=1), validity_basis="stated", seq=1), "W2"),
    (dict(lifespan="dated", valid_from=NOW, valid_until=NOW + timedelta(days=1), seq=1), "W3"),
    (dict(lifespan="lasting", seq=2), "W4"),
])
async def test_a_refused_write_writes_nothing(make_backend, fields, code):   # AC-M4, every rule on both adapters
    backend = make_backend()
    with pytest.raises(WriteCheckError) as exc:
        await backend.write_change(_rec(**fields), [_change(1, "op-w", kind="created")])
    assert exc.value.code == code
    assert await backend.read_fact("m1") is None
    assert await backend.history("m1") == []


async def test_old_document_first_change_writes_legacy_then_change(make_backend):   # spec §5
    backend = make_backend()
    old = _rec(body="Alice swims on Tuesdays at 17:30")
    await backend.upsert_fact(old)
    legacy = FactChange(fact_id="m1", seq=0, kind="legacy", at=NOW, source=ChangeSource("legacy", label="chat"),
                        reason="known before the record", op_id="tool-9", after=fact_state(old))
    new = replace(old, body="Alice swims on Tuesdays at 17:00", lifespan="fading", seq=1)
    await backend.write_change(new, [legacy, _change(1, "tool-9")])
    assert [(c.seq, c.kind) for c in await backend.history("m1")] == [(0, "legacy"), (1, "changed")]
    assert (await backend.history("m1"))[0].after == fact_state(old)
    assert (await backend.read_fact("m1")).seq == 1


async def test_change_records_are_never_memories(make_backend):   # AC-M5
    backend = make_backend()
    await _write(backend, 1, "op-a", vector_embeddings=(1.0, 0.0))
    assert await backend.read_fact("m1:1") is None
    assert [r.fact_id for r in await backend.list_facts(include_expired=True, include_future=True)] == ["m1"]


async def test_history_is_per_family(make_backend):   # AC-M6
    await _write(make_backend("fam1"), 1, "op-a")
    assert await make_backend("fam2").history("m1") == []


async def test_history_pages_by_seq(make_backend):   # AC-M18
    backend = make_backend()
    for seq in range(1, 151):
        await _write(backend, seq, f"op-{seq}")
    first = await backend.history("m1", limit=100)
    second = await backend.history("m1", after_seq=first[-1].seq, limit=100)
    assert [c.seq for c in first + second] == list(range(1, 151))
    assert (len(first), len(second)) == (100, 50)


@pytest.mark.parametrize("limit", [0, 501])
async def test_history_limit_is_bounded(make_backend, limit):
    with pytest.raises(ValueError, match="limit"):
        await make_backend().history("m1", limit=limit)


async def test_vector_write_back_only_onto_the_seq_it_read(make_backend):   # AC-M3c
    backend = make_backend()
    await _write(backend, 1, "op-1")
    await _write(backend, 2, "op-2", body="Alice swims on Wednesdays")
    assert await backend.set_vectors("m1", (0.5, 0.5), expected_seq=1) is False
    record = await backend.read_fact("m1")
    assert (record.seq, record.vector_embeddings) == (2, None)
    assert await backend.set_vectors("m1", (0.5, 0.5), expected_seq=2) is True
    assert (await backend.read_fact("m1")).vector_embeddings == (0.5, 0.5)
    assert [c.seq for c in await backend.history("m1")] == [1, 2]
    assert await backend.set_vectors("missing", (0.5, 0.5), expected_seq=0) is False


async def test_cosmos_vector_write_back_loses_an_etag_race_silently():   # AC-M3c, Cosmos etag path
    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    await _write(backend, 1, "op-1")
    container.before_replace = lambda: container._put(dict(container.docs[("fam1", "shared", "FACTS", "m1")]))
    assert await backend.set_vectors("m1", (0.5, 0.5), expected_seq=1) is False
    assert (await backend.read_fact("m1")).vector_embeddings is None


async def test_cosmos_marker_conflict_inside_the_batch_is_already_applied():   # AC-M3, two copies racing
    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    container.before_batch = lambda: container._put(
        {"id": "op-tool-1", "doc_type": "fact_op", "family_id": "fam1", "user_id": "shared", "category": "FACTS"})
    with pytest.raises(AlreadyAppliedError):
        await _write(backend, 1, "tool-1")
    assert await backend.read_fact("m1") is None


async def test_cosmos_other_batch_failures_are_raised_unchanged():
    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)

    def fail():
        raise CosmosBatchOperationError(error_index=1, headers={}, status_code=413, message="too large",
                                        operation_responses=[])

    container.execute_item_batch = lambda **kwargs: fail()
    with pytest.raises(CosmosBatchOperationError) as exc:
        await _write(backend, 1, "op-1")
    assert exc.value.status_code == 413


BROWNIES = EntityRecord(entity_id="activity-0f3a", type="activity", name="Brownies", handle="brownies", aliases=("1st Oakfield Brownies",))


async def test_entities_are_stored_found_by_alias_and_never_memories(make_backend):   # AC-M17, AC-M5
    backend = make_backend()
    await backend.upsert_entity(BROWNIES)
    assert await backend.read_entity("activity-0f3a") == BROWNIES
    assert await backend.find_by_alias("activity", "brownies") == [BROWNIES]
    assert await backend.find_by_alias("activity", "1ST OAKFIELD BROWNIES") == [BROWNIES]
    assert await backend.find_by_alias("organisation", "brownies") == []
    assert await backend.list_entities(type="activity") == [BROWNIES]
    assert await backend.read_fact("activity-0f3a") is None
    assert await backend.list_facts(include_expired=True, include_future=True) == []


async def test_entities_found_by_identifier(make_backend):
    backend = make_backend()
    club = EntityRecord(entity_id="organisation-77", type="organisation", name="Swim Club",
                        handle="swim_club", identifiers=(Identifier("domain", "SwimClub.example"),))
    await backend.upsert_entity(club)
    assert await backend.find_by_identifier("domain", "swimclub.example") == [club]
    assert await backend.find_by_identifier("address", "swimclub.example") == []


async def test_a_new_entity_is_written_with_its_memory(make_backend):   # spec §4 one batch
    backend = make_backend()
    record = _rec(lifespan="lasting", seq=1, about=("activity-0f3a",))
    await backend.write_change(record, [_change(1, "op-a", kind="created")], entities=[BROWNIES])
    assert await backend.read_entity("activity-0f3a") == BROWNIES
    assert (await backend.read_fact("m1")).about == ("activity-0f3a",)


async def test_two_writes_creating_one_entity_one_wins(make_backend):   # AC-M16b
    backend = make_backend()
    first = _rec("m1", lifespan="lasting", seq=1, about=("activity-0f3a",))
    await backend.write_change(first, [_change(1, "op-a", kind="created")], entities=[BROWNIES])
    second = _rec("m2", body="Bob goes to Brownies", lifespan="lasting", seq=1, about=("activity-0f3a",))
    with pytest.raises(StaleWriteError):
        await backend.write_change(second, [_change(1, "op-b", kind="created", fact_id="m2")], entities=[BROWNIES])
    assert await backend.read_fact("m2") is None
    assert await backend.history("m2") == []


@pytest.mark.parametrize("entities, message", [
    ([EntityRecord(entity_id=f"place-{i}", type="place", name=f"Place {i}", handle=f"place_{i}") for i in range(98)], "100"),
    ([BROWNIES, BROWNIES], "distinct"),
])
async def test_a_write_cosmos_would_refuse_is_refused_on_both(make_backend, entities, message):
    backend = make_backend()
    with pytest.raises(ValueError, match=message):
        await backend.write_change(_rec(lifespan="lasting", seq=1), [_change(1, "op-big", kind="created")],
                                   entities=entities)
    assert await backend.read_fact("m1") is None


async def test_about_and_mentions_round_trip_through_write_read_search_and_history(make_backend):   # AC-R19
    backend = make_backend()
    record = _rec(lifespan="lasting", seq=1, about=("ent-a",), mentions=("ent-b", "ent-c"), vector_embeddings=(1.0, 0.0))
    await backend.write_change(record, [_change(1, "op-a", kind="created")])
    read = await backend.read_fact("m1")
    assert (read.about, read.mentions) == (("ent-a",), ("ent-b", "ent-c"))
    [hit] = await backend.search_facts(FactSearchRequest(query_vector=(1.0, 0.0), limit=5))
    assert (hit.about, hit.mentions) == (("ent-a",), ("ent-b", "ent-c"))
    [change] = await backend.history("m1")
    assert (change.after["about"], change.after["mentions"]) == (["ent-a"], ["ent-b", "ent-c"])


async def test_entity_reads_carry_the_handle_and_a_version_that_moves_on_every_write(make_backend):   # AC-R19
    backend = make_backend()
    await backend.upsert_entity(BROWNIES)
    first = await backend.read_entity("activity-0f3a")
    assert (first.handle, first.version is not None) == ("brownies", True)
    [listed] = await backend.list_entities(type="activity")
    [found] = await backend.find_by_alias("activity", "Brownies")
    assert listed.version == found.version == first.version
    assert await backend.find_by_alias("activity", "brownies_pack") == []
    await backend.upsert_entity(replace(first, aliases=("Brownies pack",)))
    second = await backend.read_entity("activity-0f3a")
    assert second.version != first.version and second.aliases == ("Brownies pack",)


async def test_an_entity_created_with_a_memory_is_read_with_a_version(make_backend):
    backend = make_backend()
    await backend.write_change(_rec(lifespan="lasting", seq=1), [_change(1, "op-a", kind="created")],
                               entities=[BROWNIES])
    assert (await backend.read_entity("activity-0f3a")).version is not None


async def test_cosmos_entity_version_is_the_documents_etag():
    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    await backend.upsert_entity(BROWNIES)
    stored = container.docs[("fam1", "shared", "FACTS", "activity-0f3a")]
    assert (await backend.read_entity("activity-0f3a")).version == stored["_etag"]
    assert stored["handle"] == "brownies" and "version" not in stored


async def test_cosmos_every_entity_read_of_a_document_without_a_handle_raises_the_typed_error():
    backend = _backend_holding_an_entity_without_a_handle()
    for read in (lambda: backend.read_entity("activity-old"), lambda: backend.list_entities(),
                 lambda: backend.find_by_alias("activity", "brownies")):
        with pytest.raises(MissingHandleError):
            await read()


def _backend_holding_an_entity_without_a_handle() -> CosmosFactBackend:
    container = FakeContainer()
    container.upsert_item({"id": "activity-old", "doc_type": "entity", "type": "activity", "name": "Brownies",
                           "aliases": [], "alias_keys": ["brownies"], "identifiers": [], "identifier_keys": [],
                           "family_id": "fam1", "user_id": "shared", "category": "FACTS"})
    return CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)


SWIM_CLUB = EntityRecord(entity_id="organisation-77", type="organisation", name="Swim Club", handle="swim_club")


async def test_read_entities_returns_those_found_sorted_by_id(make_backend):   # AC-R19
    backend = make_backend()
    await backend.upsert_entity(SWIM_CLUB)
    await backend.upsert_entity(BROWNIES)
    got = await backend.read_entities(["organisation-77", "missing", "activity-0f3a", "organisation-77"])
    assert got == [BROWNIES, SWIM_CLUB]
    assert all(entity.version is not None for entity in got)
    assert await backend.read_entities([]) == []


async def test_read_entities_sees_only_this_familys_entities(make_backend):   # BR-40
    fam1, fam2 = make_backend("fam1"), make_backend("fam2")
    await fam1.upsert_entity(BROWNIES)
    await fam1.upsert_fact(_rec("m1"))
    assert await fam2.read_entities(["activity-0f3a"]) == []
    assert await fam1.read_entities(["m1"]) == []          # a memory is never an entity


async def test_read_entities_takes_at_most_200_distinct_ids(make_backend):
    backend = make_backend()
    assert await backend.read_entities([f"place-{i}" for i in range(200)]) == []
    assert await backend.read_entities(["place-1"] * 300) == []
    with pytest.raises(ValueError, match="200"):
        await backend.read_entities([f"place-{i}" for i in range(201)])


async def test_cosmos_read_entities_is_one_query_over_the_ids():
    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    await backend.upsert_entity(BROWNIES)
    queries = []
    real = container.query_items
    container.query_items = lambda **kwargs: queries.append(kwargs) or real(**kwargs)
    await backend.read_entities(["activity-0f3a", "x", "activity-0f3a"])
    [query] = queries
    assert "ARRAY_CONTAINS(@ids, c.id)" in query["query"] and "c.doc_type = 'entity'" in query["query"]
    assert query["parameters"] == [{"name": "@ids", "value": ["activity-0f3a", "x"]}]
    await backend.read_entities([])
    assert len(queries) == 1


async def test_cosmos_read_entities_of_a_document_without_a_handle_raises_the_typed_error():
    with pytest.raises(MissingHandleError):
        await _backend_holding_an_entity_without_a_handle().read_entities(["activity-old"])
