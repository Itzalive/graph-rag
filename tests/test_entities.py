"""Entity documents (spec xz59.11 §7): a required handle, and a version that is read, never stored."""
from __future__ import annotations

from dataclasses import replace

import pytest

from graph_rag.entities import EntityRecord, MissingHandleError, alias_keys, entity_from_state, entity_to_state

JO = EntityRecord(entity_id="person-1", type="person", name="Jo Jones", handle="jo_jones", aliases=("Grandma",))


def test_an_entity_needs_a_handle():
    with pytest.raises(TypeError):
        EntityRecord(entity_id="person-1", type="person", name="Jo Jones")


@pytest.mark.parametrize("handle", [None, ""])
def test_a_document_without_a_handle_raises_the_typed_error(handle):   # no compatibility layer (spec §7)
    state = {k: v for k, v in entity_to_state(JO).items() if k != "handle"}
    if handle is not None:
        state["handle"] = handle
    with pytest.raises(MissingHandleError) as exc:
        entity_from_state(state)
    assert isinstance(exc.value, ValueError) and exc.value.entity_id == "person-1"
    assert "person-1" in str(exc.value) and "Jo" not in str(exc.value)   # the id only: names never in messages


def test_the_handle_is_stored_and_read_back():
    state = entity_to_state(JO)
    assert state["handle"] == "jo_jones"
    assert entity_from_state(state) == JO


def test_a_handle_is_not_an_alias_key():
    assert alias_keys(JO) == ["grandma", "jo jones"]


def test_the_version_is_read_from_the_etag_and_never_stored():
    state = entity_to_state(replace(JO, version="e7"))
    assert "version" not in state and "_etag" not in state
    assert entity_from_state({**state, "_etag": "e9"}).version == "e9"
    assert entity_from_state(state).version is None


def test_the_version_is_not_part_of_equality():
    assert replace(JO, version="e1") == replace(JO, version="e2") == JO
