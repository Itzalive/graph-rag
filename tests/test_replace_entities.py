"""replace_entities (claude-assistant-xz59.13.8 §2.3), run against both recorded backends."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.entities import EntityRecord
from graph_rag.record import StaleWriteError
from cosmos_fake import FakeContainer

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)


@pytest.fixture(params=["memory"])   # Task 9 adds "cosmos"
def make(request):
    container = FakeContainer()

    def build():
        if request.param == "memory":
            return InMemoryFactBackend(clock=lambda: NOW)
        return CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)

    build.container = container if request.param == "cosmos" else None
    return build


def _entity(entity_id: str, name: str) -> EntityRecord:
    return EntityRecord(entity_id=entity_id, type="person", name=name, handle=entity_id.replace("-", "_"))


async def _stored(backend, *entities):
    for entity in entities:
        await backend.upsert_entity(entity)
    return [await backend.read_entity(e.entity_id) for e in entities]


async def test_two_writers_at_one_version_one_applies(make):   # AC-R17b
    backend = make()
    [ann] = await _stored(backend, _entity("person-1", "Ann"))
    await backend.replace_entities([replace(ann, aliases=("Annie",))])
    with pytest.raises(StaleWriteError) as exc:
        await backend.replace_entities([replace(ann, aliases=("Nan",))])
    assert str(exc.value) == "person-1"
    assert (await backend.read_entity("person-1")).aliases == ("Annie",)


async def test_a_resend_after_it_applied_changes_nothing(make):   # SPEC_DESIGN item 1
    backend = make()
    [ann] = await _stored(backend, _entity("person-1", "Ann"))
    update = [replace(ann, aliases=("Annie",))]
    await backend.replace_entities(update)
    after_first = await backend.read_entity("person-1")
    with pytest.raises(StaleWriteError):
        await backend.replace_entities(update)
    assert (await backend.read_entity("person-1")).version == after_first.version


async def test_a_batch_whose_second_update_is_stale_leaves_the_first_unchanged(make):   # AC-R17b
    backend = make()
    ann, bob = await _stored(backend, _entity("person-1", "Ann"), _entity("person-2", "Bob"))
    await backend.upsert_entity(replace(bob, aliases=("Robert",)))     # bob moves on after it was read
    with pytest.raises(StaleWriteError) as exc:
        await backend.replace_entities([replace(ann, aliases=("Annie",)), replace(bob, aliases=("Bobby",))])
    assert str(exc.value) == "person-2"
    assert await backend.read_entity("person-1") == ann
    assert (await backend.read_entity("person-1")).version == ann.version
    assert (await backend.read_entity("person-2")).aliases == ("Robert",)


async def test_an_update_of_a_missing_entity_is_stale(make):
    backend = make()
    with pytest.raises(StaleWriteError):
        await backend.replace_entities([replace(_entity("person-9", "Gone"), version="1")])
    assert await backend.read_entity("person-9") is None


async def test_no_updates_send_nothing(make):
    backend = make()
    if make.container is not None:
        make.container.execute_item_batch = _no_write
    await backend.replace_entities([])


async def test_100_updates_are_accepted(make):
    backend = make()
    stored = await _stored(backend, *[_entity(f"person-{i:03}", f"P{i}") for i in range(100)])
    await backend.replace_entities([replace(e, aliases=("x",)) for e in stored])
    assert (await backend.read_entity("person-099")).aliases == ("x",)


@pytest.mark.parametrize("updates", [
    [replace(_entity(f"person-{i:03}", "P"), version="1") for i in range(101)],
    [replace(_entity("person-1", "Ann"), version="1"), replace(_entity("person-1", "Ann"), version="1")],
    [_entity("person-1", "Ann")],                                            # no version
], ids=["101", "repeated-id", "no-version"])
async def test_malformed_updates_are_refused_before_anything_is_written(make, updates):   # AC-S1-5
    backend = make()
    if make.container is not None:
        make.container.execute_item_batch = _no_write
    with pytest.raises(ValueError):
        await backend.replace_entities(updates)


def _no_write(*args, **kwargs):
    raise AssertionError("nothing may be sent")
