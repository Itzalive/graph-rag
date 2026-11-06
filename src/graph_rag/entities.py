"""Entities in a memory bucket (claude-assistant-xz59.9 §4): stored beside the memories so a memory and the entity it
needs are written in one batch. Types are plain strings chosen by the caller."""
from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
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
    version it read. `merged_into` is stored only when set."""
    entity_id: str
    type: str
    name: str
    handle: str
    aliases: tuple[str, ...] = ()
    identifiers: tuple[Identifier, ...] = ()
    seq: int = 0
    version: str | None = field(default=None, compare=False)
    merged_into: str | None = None   # the entity this one was merged into; what it means to readers is the caller's


def alias_key(alias: str) -> str:
    return alias.casefold()


def identifier_key(kind: str, value: str) -> str:
    return f"{kind}:{value.casefold()}"


def alias_keys(entity: EntityRecord) -> list[str]:
    """The keys an exact, case-insensitive alias match uses: the name and every alias."""
    return sorted({alias_key(entity.name), *(alias_key(alias) for alias in entity.aliases)})


def identifier_keys(entity: EntityRecord) -> list[str]:
    return sorted({identifier_key(i.kind, i.value) for i in entity.identifiers})


ALIASES_MAX_LIMIT = 50
ALIASES_MAX_KEYS = 300
ALIASES_MAX_READ = 300


def check_alias_limit(limit: int) -> None:
    if not 1 <= limit <= ALIASES_MAX_LIMIT:
        raise ValueError(f"find_by_aliases limit must be 1-{ALIASES_MAX_LIMIT}")


def alias_lookup_keys(keys: Iterable[str]) -> tuple[list[str], int]:
    """The keys a lookup uses, folded with alias_key, empty ones dropped, deduplicated in the caller's order, the first
    ALIASES_MAX_KEYS; and how many distinct keys were given."""
    folded = list(dict.fromkeys(key for key in (alias_key(k) for k in keys) if key))
    return folded[:ALIASES_MAX_KEYS], len(folded)


def rank_alias_matches(entities: Iterable[EntityRecord], keys: Collection[str]) -> list[EntityRecord]:
    """Longest matching alias key (characters) first, then entity id."""
    wanted = set(keys)
    return sorted(entities, key=lambda e: (-max((len(k) for k in alias_keys(e) if k in wanted), default=0),
                                           e.entity_id))


def entity_to_state(entity: EntityRecord) -> dict[str, Any]:
    state: dict[str, Any] = {
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
    if entity.merged_into is not None:
        state["merged_into"] = entity.merged_into
    return state


class MissingHandleError(ValueError):
    """An entity document with no handle: it cannot be read, and the store holding it must be reset. The message holds
    the entity id only (ids never hold names)."""

    def __init__(self, entity_id: str) -> None:
        super().__init__(f"entity {entity_id} has no handle; reset the store that holds it")
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
        merged_into=doc.get("merged_into"),
    )


# read_entities answers one query; a caller with more ids reads them in several calls.
READ_ENTITIES_MAX_IDS = 200


def check_entity_ids(ids: Sequence[str]) -> list[str]:
    """The distinct ids of one read_entities call, sorted; more than READ_ENTITIES_MAX_IDS is refused (ValueError)."""
    distinct = sorted(set(ids))
    if len(distinct) > READ_ENTITIES_MAX_IDS:
        raise ValueError(f"read_entities takes at most {READ_ENTITIES_MAX_IDS} ids")
    return distinct
