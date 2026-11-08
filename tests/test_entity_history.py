"""Entity history (claude-assistant-xz59.13.9 S2 spec §6): every entity write records its seq, kind, time, source and
state after, in the same batch."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.entities import EntityRecord, Identifier
from graph_rag.record import (ChangeSource, EntityChange, StaleWriteError, entity_after, entity_change_from_state,
                              entity_change_id, entity_change_to_state, entity_writes)
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


async def test_an_upsert_writes_no_record(make):   # §6: test and eval fixtures
    backend = make()
    await _stored(backend, replace(CLUB, seq=4))
    assert (await backend.read_entity("ent-club")).seq == 4
    assert await backend.entity_history("ent-club") == []
