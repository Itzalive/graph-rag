"""Entity reads and replaces stay inside one family's partition: a backend built for one family never finds, nor
changes, another family's entity, even one with the same id, name, alias and identifier."""
from __future__ import annotations

from datetime import UTC, datetime

import pytest
from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.entities import EntityRecord, Identifier
from graph_rag.record import ChangeSource, StaleWriteError
from cosmos_fake import FakeContainer

NOW = datetime(2026, 11, 8, 9, 0, tzinfo=UTC)
SOURCE = ChangeSource("parent_chat", actor="parent1", ref="chat:parent1:s1")

_CLUB = EntityRecord(entity_id="ent-club", type="organisation", name="Riverside Swim Club", handle="riverside",
                     aliases=("RSC",), identifiers=(Identifier("address", "bookings@riverside.example"),
                                                    Identifier("domain", "riverside.example")))


def _two_families() -> tuple[CosmosFactBackend, CosmosFactBackend]:
    container = FakeContainer()
    return (CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW),
            CosmosFactBackend(container=container, family_id="fam2", clock=lambda: NOW))


async def test_another_familys_entity_is_never_found():
    fam1, fam2 = _two_families()
    await fam2.upsert_entity(_CLUB)
    assert await fam1.find_by_identifier("address", "bookings@riverside.example") == []
    assert await fam1.find_by_identifier("domain", "riverside.example") == []
    assert await fam1.find_by_aliases(["Riverside Swim Club", "RSC"]) == []
    assert await fam1.find_by_alias("organisation", "RSC") == []
    assert await fam1.read_entities(["ent-club"]) == []
    assert [e.entity_id for e in await fam2.find_by_identifier("address", "bookings@riverside.example")] == ["ent-club"]


async def test_another_familys_entity_is_never_changed():
    fam1, fam2 = _two_families()
    await fam2.upsert_entity(_CLUB)
    [theirs] = await fam2.read_entities(["ent-club"])
    with pytest.raises(StaleWriteError):
        await fam1.replace_entities([EntityRecord(**{**vars(theirs), "aliases": ("Changed",)})], source=SOURCE, at=NOW)
    [after] = await fam2.read_entities(["ent-club"])
    assert after.aliases == ("RSC",)
