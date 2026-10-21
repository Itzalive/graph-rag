"""Entities in a memory bucket (claude-assistant-xz59.9 §4): stored beside the memories so a memory and the entity it
needs are written in one batch. Types are plain strings chosen by the caller."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Identifier:
    kind: str     # for example "address" or "domain"
    value: str


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    type: str
    name: str
    aliases: tuple[str, ...] = ()
    identifiers: tuple[Identifier, ...] = ()
    seq: int = 0


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
        "aliases": list(entity.aliases),
        "identifiers": [{"kind": i.kind, "value": i.value} for i in entity.identifiers],
        "seq": entity.seq,
        "alias_keys": alias_keys(entity),
        "identifier_keys": identifier_keys(entity),
    }


def entity_from_state(doc: dict[str, Any]) -> EntityRecord:
    return EntityRecord(
        entity_id=doc["id"],
        type=doc["type"],
        name=doc["name"],
        aliases=tuple(doc.get("aliases", ())),
        identifiers=tuple(Identifier(i["kind"], i["value"]) for i in doc.get("identifiers", ())),
        seq=doc.get("seq", 0),
    )
