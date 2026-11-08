"""replace_entities (claude-assistant-xz59.13.8 §2.3), run against both recorded backends."""
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from graph_rag.adapters.cosmos import CosmosFactBackend
from graph_rag.adapters.in_memory_facts import InMemoryFactBackend
from graph_rag.entities import EntityRecord
from graph_rag.record import ChangeSource, StaleWriteError
from cosmos_fake import FakeContainer

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=UTC)
SOURCE = ChangeSource("parent_chat", actor="parent1", ref="chat:parent1:s1")


@pytest.fixture(params=["memory", "cosmos"])
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
    await backend.replace_entities([replace(ann, aliases=("Annie",))], source=SOURCE, at=NOW)
    with pytest.raises(StaleWriteError) as exc:
        await backend.replace_entities([replace(ann, aliases=("Nan",))], source=SOURCE, at=NOW)
    assert str(exc.value) == "person-1"
    assert (await backend.read_entity("person-1")).aliases == ("Annie",)


async def test_a_resend_after_it_applied_changes_nothing(make):   # SPEC_DESIGN item 1
    backend = make()
    [ann] = await _stored(backend, _entity("person-1", "Ann"))
    update = [replace(ann, aliases=("Annie",))]
    await backend.replace_entities(update, source=SOURCE, at=NOW)
    after_first = await backend.read_entity("person-1")
    with pytest.raises(StaleWriteError):
        await backend.replace_entities(update, source=SOURCE, at=NOW)
    assert (await backend.read_entity("person-1")).version == after_first.version


async def test_a_batch_whose_second_update_is_stale_leaves_the_first_unchanged(make):   # AC-R17b
    backend = make()
    ann, bob = await _stored(backend, _entity("person-1", "Ann"), _entity("person-2", "Bob"))
    await backend.upsert_entity(replace(bob, aliases=("Robert",)))     # bob moves on after it was read
    with pytest.raises(StaleWriteError) as exc:
        await backend.replace_entities([replace(ann, aliases=("Annie",)), replace(bob, aliases=("Bobby",))],
                                       source=SOURCE, at=NOW)
    assert str(exc.value) == "person-2"
    assert await backend.read_entity("person-1") == ann
    assert (await backend.read_entity("person-1")).version == ann.version
    assert (await backend.read_entity("person-2")).aliases == ("Robert",)


async def test_a_batch_whose_first_update_is_stale_names_it_and_writes_nothing(make):   # AC-R17b
    backend = make()
    ann, bob = await _stored(backend, _entity("person-1", "Ann"), _entity("person-2", "Bob"))
    await backend.upsert_entity(replace(ann, aliases=("Annie",)))     # ann moves on after it was read
    with pytest.raises(StaleWriteError) as exc:
        await backend.replace_entities([replace(ann, aliases=("Nan",)), replace(bob, aliases=("Bobby",))],
                                       source=SOURCE, at=NOW)
    assert str(exc.value) == "person-1"
    assert await backend.read_entity("person-2") == bob


async def test_an_update_at_a_version_it_never_read_is_stale(make):
    backend = make()
    [ann] = await _stored(backend, _entity("person-1", "Ann"))
    with pytest.raises(StaleWriteError):
        await backend.replace_entities([replace(ann, aliases=("Annie",), version=ann.version + "0")],
                                       source=SOURCE, at=NOW)
    assert await backend.read_entity("person-1") == ann


async def test_an_update_of_a_missing_entity_is_stale(make):
    backend = make()
    with pytest.raises(StaleWriteError):
        await backend.replace_entities([replace(_entity("person-9", "Gone"), version="1")], source=SOURCE, at=NOW)
    assert await backend.read_entity("person-9") is None


async def test_no_updates_send_nothing(make):
    backend = make()
    if make.container is not None:
        make.container.execute_item_batch = _no_write
    await backend.replace_entities([], source=SOURCE, at=NOW)


async def test_50_updates_are_accepted(make):
    backend = make()
    stored = await _stored(backend, *[_entity(f"person-{i:03}", f"P{i}") for i in range(50)])
    await backend.replace_entities([replace(e, aliases=("x",)) for e in stored], source=SOURCE, at=NOW)
    assert (await backend.read_entity("person-049")).aliases == ("x",)


@pytest.mark.parametrize("updates", [
    [replace(_entity(f"person-{i:03}", "P"), version="1") for i in range(51)],
    [replace(_entity("person-1", "Ann"), version="1"), replace(_entity("person-1", "Ann"), version="1")],
    [_entity("person-1", "Ann")],                                            # no version
], ids=["51", "repeated-id", "no-version"])
async def test_malformed_updates_are_refused_before_anything_is_written(make, updates):   # AC-S1-5
    backend = make()
    if make.container is not None:
        make.container.execute_item_batch = _no_write
    with pytest.raises(ValueError):
        await backend.replace_entities(updates, source=SOURCE, at=NOW)


def _no_write(*args, **kwargs):
    raise AssertionError("nothing may be sent")


async def test_cosmos_replace_entities_is_one_conditional_batch():
    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    ann, bob = await _stored(backend, _entity("person-1", "Ann"), _entity("person-2", "Bob"))
    await backend.replace_entities([replace(ann, aliases=("Annie",)), bob], source=SOURCE, at=NOW)
    [batch] = container.batches
    assert [(op[0], op[1][0] if op[0] == "replace" else op[1][0]["id"], op[2] if len(op) > 2 else {})
            for op in batch] == [("replace", "person-1", {"if_match_etag": ann.version}),
                                 ("replace", "person-2", {"if_match_etag": bob.version}),
                                 ("create", "entity_change:person-1:1", {}),
                                 ("create", "entity_change:person-2:1", {})]
    assert batch[0][1][1]["family_id"] == "fam1" and batch[0][1][1]["alias_keys"] == ["ann", "annie"]
    assert batch[2][1][0]["family_id"] == "fam1" and batch[2][1][0]["source"] == {"kind": "parent_chat",
                                                                                     "actor": "parent1",
                                                                                     "ref": "chat:parent1:s1"}


@pytest.mark.parametrize("status", [400, 409, 429])
async def test_cosmos_other_batch_failures_are_raised_unchanged(status):
    from azure.cosmos.exceptions import CosmosBatchOperationError  # noqa: PLC0415

    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    [ann] = await _stored(backend, _entity("person-1", "Ann"))

    def fail(**kwargs):
        raise CosmosBatchOperationError(error_index=0, headers={}, status_code=status, message="failed",
                                        operation_responses=[])

    container.execute_item_batch = fail
    with pytest.raises(CosmosBatchOperationError):
        await backend.replace_entities([ann], source=SOURCE, at=NOW)


async def test_cosmos_a_conflict_on_a_record_names_its_entity():
    from azure.cosmos.exceptions import CosmosBatchOperationError  # noqa: PLC0415

    container = FakeContainer()
    backend = CosmosFactBackend(container=container, family_id="fam1", clock=lambda: NOW)
    ann, bob = await _stored(backend, _entity("person-1", "Ann"), _entity("person-2", "Bob"))

    def fail(**kwargs):
        raise CosmosBatchOperationError(error_index=3, headers={}, status_code=409, message="conflict",
                                        operation_responses=[])

    container.execute_item_batch = fail
    with pytest.raises(StaleWriteError) as exc:
        await backend.replace_entities([ann, bob], source=SOURCE, at=NOW)
    assert exc.value.args[0] == "person-2"
