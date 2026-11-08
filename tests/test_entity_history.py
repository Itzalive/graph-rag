"""Entity history (claude-assistant-xz59.13.9 S2 spec §6): every entity write records its seq, kind, time, source and
state after, in the same batch."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.entities import EntityRecord, Identifier
from graph_rag.facts import FactRecord
from graph_rag.record import (AlreadyAppliedError, ChangeSource, EntityChange, FactChange, StaleWriteError, entity_after,
                              entity_change_from_state, entity_change_id, entity_change_to_state, entity_writes)
from cosmos_fake import FakeContainer

NOW = datetime(2026, 11, 8, 9, 0, tzinfo=UTC)
SOURCE = ChangeSource("parent_chat", actor="parent1", ref="chat:parent1:s1")
CLUB = EntityRecord(entity_id="ent-club", type="organisation", name="Riverside Swim Club", handle="riverside",
                    aliases=("RSC",), identifiers=(Identifier("address", "bookings@riverside.example"),))


def test_entity_writes_stamp_seq_and_build_one_record_each():
    created, updated, records = entity_writes([replace(CLUB, seq=7)], [replace(CLUB, entity_id="ent-two", seq=4)],
                                              source=SOURCE, at=NOW)
    assert [(e.entity_id, e.seq) for e in (*created, *updated)] == [("ent-club", 1), ("ent-two", 5)]
    assert [(r.entity_id, r.seq, r.kind, r.at, r.source) for r in records] == [
        ("ent-club", 1, "created", NOW, SOURCE), ("ent-two", 5, "changed", NOW, SOURCE)]
    assert records[0].after == {"type": "organisation", "name": "Riverside Swim Club", "handle": "riverside",
                                "aliases": ["RSC"],
                                "identifiers": [{"kind": "address", "value": "bookings@riverside.example"}],
                                "seq": 1}


def test_a_record_round_trips_through_its_document():
    merged = replace(CLUB, seq=3, merged_into="ent-kept")
    record = EntityChange(entity_id="ent-club", seq=3, kind="changed", at=NOW,
                          source=ChangeSource("email", actor="parent1", ref="email:m1"), after=entity_after(merged))
    doc = entity_change_to_state(record)
    assert doc == {"id": "entity_change:ent-club:3", "doc_type": "entity_change", "entity_id": "ent-club", "seq": 3,
                   "kind": "changed", "at": "2026-11-08T09:00:00+00:00",
                   "source": {"kind": "email", "actor": "parent1", "ref": "email:m1"},
                   "after": {**entity_after(CLUB), "seq": 3, "merged_into": "ent-kept"}}
    assert entity_change_from_state(doc) == record
    assert "merged_into" not in entity_after(CLUB)
    assert entity_change_id("ent-club", 3) == "entity_change:ent-club:3"


@pytest.fixture(params=["memory", "cosmos"])
def make(request):
    container = FakeContainer()

    def build():
        if request.param == "memory":
            return InMemoryFactBackend(clock=lambda: NOW)
        return CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)

    build.container = container if request.param == "cosmos" else None
    return build


async def _stored(backend, entity=CLUB):
    await backend.upsert_entity(entity)
    return await backend.read_entity(entity.entity_id)


async def test_a_replace_records_its_change(make):   # AC-S2-20
    backend = make()
    read = await _stored(backend)
    await backend.replace_entities([replace(read, aliases=("RSC", "Riverside"))], source=SOURCE, at=NOW)
    first = await backend.read_entity("ent-club")
    assert first.seq == 1
    await backend.replace_entities([replace(first, aliases=())], source=ChangeSource("email", ref="email:m2"),
                                   at=NOW)
    history = await backend.entity_history("ent-club")
    assert [(c.seq, c.kind, c.source.kind, c.after["aliases"]) for c in history] == [
        (1, "changed", "parent_chat", ["RSC", "Riverside"]), (2, "changed", "email", [])]
    assert (await backend.read_entity("ent-club")).seq == 2


async def test_a_stale_replace_writes_neither_entity_nor_record(make):   # AC-S2-20
    backend = make()
    read = await _stored(backend)
    await backend.replace_entities([replace(read, aliases=("One",))], source=SOURCE, at=NOW)
    with pytest.raises(StaleWriteError):
        await backend.replace_entities([replace(read, aliases=("Two",))], source=SOURCE, at=NOW)
    assert (await backend.read_entity("ent-club")).aliases == ("One",)
    assert [c.after["aliases"] for c in await backend.entity_history("ent-club")] == [["One"]]


async def test_a_taken_record_id_is_stale_and_writes_neither(make):   # AC-S2-21
    backend = make()
    read = await _stored(backend)
    # an upsert puts the entity back at seq 0 with a new version, so the next replace takes a seq already recorded
    await backend.replace_entities([replace(read, aliases=("One",))], source=SOURCE, at=NOW)
    await backend.upsert_entity(replace(read, aliases=("Reset",)))
    reset = await backend.read_entity("ent-club")
    with pytest.raises(StaleWriteError) as exc:
        await backend.replace_entities([replace(reset, aliases=("Two",))], source=SOURCE, at=NOW)
    assert exc.value.args[0] == "ent-club"
    assert (await backend.read_entity("ent-club")).aliases == ("Reset",)
    assert [c.after["aliases"] for c in await backend.entity_history("ent-club")] == [["One"]]


async def test_history_reads_in_seq_order_after_a_seq(make):
    backend = make()
    current = await _stored(backend)
    for n in range(3):
        await backend.replace_entities([replace(current, aliases=(f"A{n}",))], source=SOURCE, at=NOW)
        current = await backend.read_entity("ent-club")
    assert [c.seq for c in await backend.entity_history("ent-club")] == [1, 2, 3]
    assert [c.seq for c in await backend.entity_history("ent-club", after_seq=1)] == [2, 3]
    assert [c.seq for c in await backend.entity_history("ent-club", limit=2)] == [1, 2]
    assert await backend.entity_history("ent-other") == []
    with pytest.raises(ValueError):
        await backend.entity_history("ent-club", limit=0)


async def test_records_are_never_read_as_memories_entities_or_changes(make):   # §6 Other readers
    backend = make()
    read = await _stored(backend)
    await backend.replace_entities([replace(read, aliases=("RSC", "Riverside"))], source=SOURCE, at=NOW)
    assert await backend.list_facts() == []
    assert [e.entity_id for e in await backend.list_entities()] == ["ent-club"]
    assert await backend.history("ent-club") == []
    assert [e.entity_id for e in await backend.find_by_alias("organisation", "RSC")] == ["ent-club"]
    assert [e.entity_id for e in await backend.find_by_aliases(["Riverside"])] == ["ent-club"]
    assert [e.entity_id for e in await backend.find_by_identifier("address", "bookings@riverside.example")] == [
        "ent-club"]


def _memory(seq=1):
    return FactRecord(fact_id="m1", body="Bob swims at Riverside", created_at=NOW, updated_at=NOW, lifespan="lasting",
                      seq=seq)


def _change(seq, kind, op_id, source=SOURCE, after=None):
    return FactChange(fact_id="m1", seq=seq, kind=kind, at=NOW, source=source, reason="told in chat", op_id=op_id,
                      after=after)


async def test_a_memory_write_records_the_entities_it_creates_and_updates(make):   # AC-S2-20
    backend = make()
    other = await _stored(backend, replace(CLUB, entity_id="ent-two", handle="two", name="Two"))
    legacy = ChangeSource("legacy", label="chat")
    await backend.write_change(_memory(), [_change(0, "legacy", "op-0", source=legacy, after={"fact": "x"}),
                                           _change(1, "created", "op-1")],
                               entities=[replace(CLUB, seq=9)], entity_updates=[replace(other, aliases=("2",))])
    assert [(e.entity_id, e.seq) for e in [await backend.read_entity("ent-club"), await backend.read_entity("ent-two")]
            ] == [("ent-club", 1), ("ent-two", 1)]
    [made] = await backend.entity_history("ent-club")
    [changed] = await backend.entity_history("ent-two")
    assert (made.kind, made.seq, made.source, made.at) == ("created", 1, SOURCE, NOW)
    assert (changed.kind, changed.seq, changed.after["aliases"]) == ("changed", 1, ["2"])


async def test_a_memory_write_whose_entity_record_is_taken_writes_nothing(make):   # AC-S2-21
    backend = make()
    read = await _stored(backend)
    await backend.replace_entities([replace(read, aliases=("One",))], source=SOURCE, at=NOW)
    await backend.upsert_entity(replace(read, aliases=("Reset",)))   # back at seq 0: the next write takes seq 1 again
    reset = await backend.read_entity("ent-club")
    with pytest.raises(StaleWriteError):
        await backend.write_change(_memory(), [_change(1, "created", "op-1")],
                                   entity_updates=[replace(reset, aliases=("Two",))])
    assert await backend.read_fact("m1") is None
    assert (await backend.read_entity("ent-club")).aliases == ("Reset",)


async def test_an_applied_write_records_nothing_again(make):
    backend = make()
    await backend.write_change(_memory(), [_change(1, "created", "op-1")], entities=[CLUB])
    with pytest.raises(AlreadyAppliedError):   # the same op with another new entity: nothing of it is written
        await backend.write_change(_memory(), [_change(1, "created", "op-1")],
                                   entities=[replace(CLUB, entity_id="ent-new", handle="new")])
    assert await backend.entity_history("ent-new") == []
    assert await backend.read_entity("ent-new") is None
    assert [c.seq for c in await backend.entity_history("ent-club")] == [1]


async def test_cosmos_memory_write_batch_layout():
    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    other = await _stored(backend, replace(CLUB, entity_id="ent-two", handle="two", name="Two"))
    await backend.write_change(_memory(), [_change(1, "created", "op-1")], entities=[CLUB],
                               entity_updates=[replace(other, aliases=("2",))])
    batch = container.batches[-1]
    ids = [op[1][0] if op[0] == "replace" else op[1][0]["id"] for op in batch]
    assert [op[0] for op in batch] == ["create", "create", "create", "replace", "create", "create", "upsert"]
    assert ids[2:6] == ["ent-club", "ent-two", "entity_change:ent-club:1", "entity_change:ent-two:1"]


async def test_an_upsert_writes_no_record(make):   # §6: test and eval fixtures
    backend = make()
    await _stored(backend, replace(CLUB, seq=4))
    assert (await backend.read_entity("ent-club")).seq == 4
    assert await backend.entity_history("ent-club") == []
