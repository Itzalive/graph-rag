"""A Cosmos container fake for CosmosFactBackend's contract tests.

Documents are kept per hierarchical partition key. Transactional batches check every operation first and apply all or
none; a create on an existing id fails with 409 at that operation's index, as the SDK reports it
(azure/cosmos/exceptions.py CosmosBatchOperationError). A replace, alone or in a batch, honours an etag condition with
412 and fails with 404 on a missing id. Queries are answered only for the clauses the adapter defines: an unknown WHERE
clause raises AssertionError, so a filter the adapter drops or misspells fails a test instead of passing silently. The
situational order is applied before TOP; a select of named fields returns only those fields."""
from __future__ import annotations

import copy
import itertools
import math
import re
from collections.abc import Callable

from azure.cosmos.exceptions import (
    CosmosAccessConditionFailedError,
    CosmosBatchOperationError,
    CosmosResourceNotFoundError,
)

from graph_rag.adapters import cosmos as adapter


def _pk_of(doc: dict) -> tuple:
    return (doc["family_id"], doc["user_id"], doc["category"])


def _similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 0.0 if mag == 0.0 else dot / mag


def _situation(d: dict) -> dict:
    return d.get("situation") or {}


def _shares(values, wanted) -> bool:
    return not values or bool(set(values) & set(wanted))


_CLAUSES: dict[str, Callable[[dict, dict], bool]] = {
    adapter.CLAUSE_MEMORY: lambda d, p: "doc_type" not in d,
    adapter.CLAUSE_NOT_DELETED: lambda d, p: d.get("deleted") is not True,
    adapter.CLAUSE_NOT_FORGOTTEN: lambda d, p: d.get("forgotten_at") is None,
    adapter.CLAUSE_NOT_PURGED: lambda d, p: d.get("purged_at") is None,
    adapter.CLAUSE_NOT_EXPIRED: lambda d, p: d.get("valid_until") is None or d["valid_until"] > p["@now"],
    adapter.CLAUSE_STARTED: lambda d, p: d.get("valid_from") is None or d["valid_from"] <= p["@now"],
    adapter.CLAUSE_NOT_EXCLUDED: lambda d, p: d["id"] not in p["@exclude_ids"],
    adapter.CLAUSE_NOT_EXCLUDED_VERSION: lambda d, p: f'{d["id"]}:{d.get("seq", 0)}' not in p["@exclude_keys"],
    adapter.CLAUSE_CHANGES: lambda d, p: d.get("doc_type") == "fact_change",
    adapter.CLAUSE_CHANGE_OF_FACT: lambda d, p: d.get("fact_id") == p["@fact_id"],
    adapter.CLAUSE_AFTER_SEQ: lambda d, p: d.get("seq", 0) > p["@after"],
    adapter.CLAUSE_ENTITIES: lambda d, p: d.get("doc_type") == "entity",
    adapter.CLAUSE_ENTITY_TYPE: lambda d, p: d.get("type") == p["@type"],
    adapter.CLAUSE_ALIAS: lambda d, p: p["@alias"] in d.get("alias_keys", []),
    adapter.CLAUSE_ALIAS_ANY: lambda d, p: bool(set(d.get("alias_keys", [])) & set(p["@keys"])),
    adapter.CLAUSE_IDENTIFIER:lambda d, p: p["@identifier"] in d.get("identifier_keys", []),
    adapter.CLAUSE_ENTITY_IDS: lambda d, p: d["id"] in p["@ids"],
    adapter.CLAUSE_SITUATED_CHANNEL: lambda d, p: p["@channel"] in (_situation(d).get("channels") or []),
    adapter.CLAUSE_SITUATED_TYPES: lambda d, p: _shares(_situation(d).get("types"), p["@types"]),
    adapter.CLAUSE_SITUATED_ENTITIES: lambda d, p: _shares(_situation(d).get("entities"), p["@entity_ids"]),
    adapter.CLAUSE_SITUATED_TOPICS: lambda d, p: not _situation(d).get("topics"),
    adapter.CLAUSE_UNSITUATED: lambda d, p: d.get("situation") is None,
    adapter.CLAUSE_KIND_IN: lambda d, p: d.get("kind") in p["@kinds"],
}


class FakeContainer:
    def __init__(self) -> None:
        self.docs: dict[tuple, dict] = {}
        self._etags = itertools.count(1)
        self.before_batch: Callable[[], None] | None = None     # run once, just before the next batch is checked
        self.before_replace: Callable[[], None] | None = None   # run once, just before the next replace is checked
        self.batches: list[list] = []

    def _put(self, doc: dict) -> None:
        stored = copy.deepcopy(doc)
        stored["_etag"] = f"e{next(self._etags)}"
        self.docs[(*_pk_of(doc), doc["id"])] = stored

    def read_item(self, item: str, partition_key: list) -> dict:
        doc = self.docs.get((*partition_key, item))
        if doc is None:
            raise CosmosResourceNotFoundError(status_code=404, message="not found")
        return copy.deepcopy(doc)

    def upsert_item(self, body: dict) -> dict:
        self._put(body)
        return copy.deepcopy(body)

    def replace_item(self, item: str, body: dict, etag: str | None = None, match_condition=None) -> dict:
        if self.before_replace is not None:
            hook, self.before_replace = self.before_replace, None
            hook()
        current = self.docs.get((*_pk_of(body), item))
        if current is None:
            raise CosmosResourceNotFoundError(status_code=404, message="not found")
        if etag is not None and current["_etag"] != etag:
            raise CosmosAccessConditionFailedError(status_code=412, message="precondition failed")
        self._put(body)
        return copy.deepcopy(body)

    def execute_item_batch(self, batch_operations: list, partition_key: list) -> list:
        if self.before_batch is not None:
            hook, self.before_batch = self.before_batch, None
            hook()
        self.batches.append(batch_operations)
        for index, operation in enumerate(batch_operations):
            kind, args = operation[0], operation[1]
            options = operation[2] if len(operation) > 2 else {}
            if kind not in ("create", "upsert", "replace"):
                raise AssertionError(f"unsupported batch operation: {kind}")
            doc = args[-1]
            if list(_pk_of(doc)) != list(partition_key):
                raise _batch_error(index, 400)
            key = (*partition_key, doc["id"])
            if kind == "create" and key in self.docs:
                raise _batch_error(index, 409)
            if kind == "replace":
                if args[0] != doc["id"]:
                    raise AssertionError("a batch replace names one id and carries another")
                if key not in self.docs:
                    raise _batch_error(index, 404)
                if "if_match_etag" in options and self.docs[key]["_etag"] != options["if_match_etag"]:
                    raise _batch_error(index, 412)
        for operation in batch_operations:
            self._put(operation[1][-1])
        return [{"statusCode": 201 if op[0] == "create" else 200} for op in batch_operations]

    def query_items(self, query: str, parameters=None, partition_key=None) -> list[dict]:
        params = {p["name"]: p["value"] for p in (parameters or [])}
        rows = [copy.deepcopy(d) for key, d in self.docs.items() if list(key[:3]) == list(partition_key)]
        select, rest = query.split(" FROM c", 1)
        where = rest.split(" WHERE ", 1)[1] if " WHERE " in rest else ""
        where = where.split(" ORDER BY ", 1)[0]
        for clause in (where.split(" AND ") if where else []):
            if clause not in _CLAUSES:
                raise AssertionError(f"FakeContainer does not know the clause {clause!r}")
            rows = [d for d in rows if _CLAUSES[clause](d, params)]
        if "ORDER BY c.seq ASC" in query:
            rows.sort(key=lambda d: d["seq"])
        if adapter.ORDER_SITUATIONAL in query:
            rows.sort(key=lambda d: (d.get("constraint") is not True,
                                     "learned_at" in d, d.get("learned_at") or "", d["id"]))
        if "VectorDistance" in select:
            rows = [d for d in rows if d.get("vector_embeddings")]
            for d in rows:
                d["score"] = _similarity(params["@qv"], d["vector_embeddings"])
            rows.sort(key=lambda d: -d["score"])
        fields_part = re.sub(r"^SELECT (TOP (@\w+|\d+) )?", "", select)
        if fields_part.strip() != "*":
            fields = [part[2:] for part in fields_part.split(", ") if part.startswith("c.")]
            rows = [{**{f: d[f] for f in fields if f in d}, **({"score": d["score"]} if "score" in d else {})}
                    for d in rows]
        top = re.match(r"SELECT TOP (@\w+|\d+) ", query)
        if top:
            rows = rows[: params[top[1]] if top[1].startswith("@") else int(top[1])]
        return rows


def _batch_error(index: int, status: int) -> CosmosBatchOperationError:
    return CosmosBatchOperationError(error_index=index, headers={}, status_code=status,
                                     message=f"batch operation {index} failed", operation_responses=[])
