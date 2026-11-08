"""Entity history (claude-assistant-xz59.13.9 S2 spec §6): every entity write records its seq, kind, time, source and
state after, in the same batch."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from graph_rag.entities import EntityRecord, Identifier
from graph_rag.record import (ChangeSource, EntityChange, entity_after, entity_change_from_state, entity_change_id,
                              entity_change_to_state, entity_writes)

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
