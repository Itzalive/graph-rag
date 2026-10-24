"""Entities in a memory bucket (claude-assistant-xz59.9 §4): stored beside the memories so a memory and the entity it
needs are written in one batch. Types are plain strings chosen by the caller."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Identifier:
    kind: str     # for example "address" or "domain"
    value: str


@dataclass(frozen=True)
class EntityRecord:
    """An entity in a bucket. `handle` is the caller's readable id for it (its rule is the caller's); it is not an alias
    key: find_by_alias matches the name and aliases only. `version` is set by every read (Cosmos: the document's _etag;
    in memory: a counter) and moves on every write; it is never stored and not part of equality. An update carries the
    version it read."""
    entity_id: str
    type: str
    name: str
    handle: str
    aliases: tuple[str, ...] = ()
    identifiers: tuple[Identifier, ...] = ()
    seq: int = 0
    version: str | None = field(default=None, compare=False)


def alias_key(alias: str) -> str:
    return alias.casefold()


def identifier_key(kind: str, value: str) -> str:
    return f"{kind}:{value.casefold()}"


def alias_keys(entity: EntityRecord) -> list[str]:
    """The keys an exact, case-insensitive alias match uses: the name and every alias."""
    return sorted({alias_key(entity.name), *(alias_key(alias) for alias in entity.aliases)})


def identifier_keys(entity: EntityRecord) -> list[str]:
    return sorted({identifier_key(i.kind, i.value) for i in entity.identifiers})


def entity_to_state(entity: EntityRecord) -> dict[str, Any]:
    return {
        "id": entity.entity_id,
        "doc_type": "entity",
        "type": entity.type,
        "name": entity.name,
        "handle": entity.handle,
        "aliases": list(entity.aliases),
        "identifiers": [{"kind": i.kind, "value": i.value} for i in entity.identifiers],
        "seq": entity.seq,
        "alias_keys": alias_keys(entity),
        "identifier_keys": identifier_keys(entity),
    }


class MissingHandleError(ValueError):
    """An entity document with no handle: it was written before entities had handles. There is no compatibility layer;
    the store holding it must be reset. The message holds the entity id only (ids never hold names)."""

    def __init__(self, entity_id: str) -> None:
        super().__init__(f"entity {entity_id} has no handle: it was written before entities had handles; "
                         "reset the store that holds it")
        self.entity_id = entity_id


def entity_from_state(doc: dict[str, Any]) -> EntityRecord:
    if not doc.get("handle"):
        raise MissingHandleError(doc["id"])
    return EntityRecord(
        entity_id=doc["id"],
        type=doc["type"],
        name=doc["name"],
        handle=doc["handle"],
        aliases=tuple(doc.get("aliases", ())),
        identifiers=tuple(Identifier(i["kind"], i["value"]) for i in doc.get("identifiers", ())),
        seq=doc.get("seq", 0),
        version=doc.get("_etag"),
    )
