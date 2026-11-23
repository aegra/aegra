"""Compile `@auth.on` handler filter dicts into SQLAlchemy predicates.

A handler may return a filter dict (`Auth.types.FilterType`) to scope a query.
Keys match against the resource's metadata JSONB, except the keys listed in
``_FILTER_COLUMNS``, which name a real column and match that column.

Supported shapes:
    {"owner": "u1"}                       -> metadata @> {"owner": "u1"}        (eq)
    {"owner": {"$eq": "u1"}}              -> same, explicit
    {"tags": {"$contains": "x"}}          -> metadata->'tags' @> ["x"]          (array contains)
    {"tags": {"$contains": ["x", "y"]}}   -> metadata->'tags' @> ["x", "y"]
    {"graph_id": "agent"}                 -> assistant.graph_id = 'agent'       (column, eq only)
    {"metadata": {"graph_id": "agent"}}   -> metadata @> {"graph_id": "agent"}  (envelope: always metadata)
    {"$or": [f1, f2, ...]}                -> OR of branches (>= 2, depth <= 2)
    {"$and": [f1, f2, ...]}              -> AND of branches (>= 2)
    multiple top-level keys               -> AND (logical conjunction)

Invalid operators or over-nested filters raise HTTPException(500): a malformed
handler filter is a programmer error, surfaced loudly rather than silently
matching the wrong rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fastapi import HTTPException
from sqlalchemy import and_, or_
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql.elements import ColumnElement

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Thread as ThreadORM

_MAX_FILTER_DEPTH = 2

_Columns = Mapping[str, InstrumentedAttribute[Any]]

# Filter keys that match a real column, by mapped class. Without this a handler
# returning {"graph_id": ...} matched the metadata JSONB and found nothing (#360).
_FILTER_COLUMNS: dict[object, _Columns] = {
    AssistantORM: {"graph_id": AssistantORM.graph_id, "name": AssistantORM.name},
    ThreadORM: {"status": ThreadORM.status},
}
_NO_COLUMNS: _Columns = {}


def build_metadata_filter(
    column: InstrumentedAttribute[Any],
    filters: dict[str, Any] | None,
) -> ColumnElement[bool] | None:
    """Compile a handler filter dict into a predicate for the resource that owns `column`.

    Returns None when there is nothing to filter (so callers can skip adding a
    WHERE clause). `column` is the JSONB metadata column (e.g.
    `AssistantORM.metadata_dict`); its model selects the column keys.

    Aegra compatibility: handlers historically return a nested
    ``{"metadata": {...}}`` envelope to mean "filter on these metadata fields"
    (see examples/jwt_mock_auth_example.py). Its fields are compiled together
    with any sibling flat constraints, so ``{"metadata": {"team_id": "t"}}`` and
    ``{"team_id": "t"}`` filter the same way. Inside the envelope every key is
    a metadata key, including one that shares its name with a column.
    """
    if not filters:
        return None
    columns = _FILTER_COLUMNS.get(column.class_, _NO_COLUMNS)
    envelope, top_level = _split_metadata_envelope(filters, columns)
    clauses = [
        *_compile(column, envelope, depth=0, columns=_NO_COLUMNS),
        *_compile(column, top_level, depth=0, columns=columns),
    ]
    if not clauses:
        return None
    return and_(*clauses) if len(clauses) > 1 else clauses[0]


def _split_metadata_envelope(filters: dict[str, Any], columns: _Columns) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(envelope fields, other keys)`` for a top-level ``metadata`` envelope.

    A sibling key replaces the same key in the envelope, unless the sibling
    filters a column: then the two constrain different things and both apply.
    """
    envelope = filters.get("metadata")
    if not isinstance(envelope, dict):
        return {}, filters
    top_level = {k: v for k, v in filters.items() if k != "metadata"}
    metadata_fields = {k: v for k, v in envelope.items() if k not in top_level or k in columns}
    return metadata_fields, top_level


def _compile(
    column: InstrumentedAttribute[Any],
    filters: dict[str, Any],
    *,
    depth: int,
    columns: _Columns,
) -> list[ColumnElement[bool]]:
    clauses: list[ColumnElement[bool]] = []
    for key, value in filters.items():
        if key in ("$or", "$and"):
            clauses.append(_compile_bool_op(column, key, value, depth=depth, columns=columns))
        elif key in columns:
            clauses.append(_compile_column_field(columns[key], key, value))
        else:
            clauses.append(_compile_field(column, key, value))
    return clauses


def _compile_bool_op(
    column: InstrumentedAttribute[Any],
    op: str,
    branches: Any,
    *,
    depth: int,
    columns: _Columns,
) -> ColumnElement[bool]:
    if depth >= _MAX_FILTER_DEPTH:
        raise HTTPException(
            500,
            f"Auth handler filter nests deeper than {_MAX_FILTER_DEPTH} levels. "
            "Check the filter returned by your auth handler.",
        )
    if not isinstance(branches, list) or len(branches) < 2:
        raise HTTPException(
            500,
            f"Auth handler filter '{op}' must be a list of at least 2 filter objects. "
            "Check the filter returned by your auth handler.",
        )
    # An empty branch would compile to and_() == TRUE, making OR collapse to TRUE
    # and silently bypass the constraint. Reject it.
    for branch in branches:
        if not isinstance(branch, dict) or not branch:
            raise HTTPException(
                500,
                f"Auth handler filter '{op}' contains an empty or non-dict branch. "
                "Check the filter returned by your auth handler.",
            )
    compiled = [and_(*_compile(column, b, depth=depth + 1, columns=columns)) for b in branches]
    return or_(*compiled) if op == "$or" else and_(*compiled)


def _compile_column_field(target: InstrumentedAttribute[Any], key: str, value: Any) -> ColumnElement[bool]:
    """Equality on a real column. ``$contains`` has no meaning for a scalar column."""
    operand = value
    if isinstance(value, dict):
        if set(value) != {"$eq"}:
            raise HTTPException(
                500,
                f"Auth handler filter on the '{key}' column takes a plain value or $eq only. "
                "Check the filter returned by your auth handler.",
            )
        operand = value["$eq"]
    # Every routed column is text; any other operand would fail in the driver as an opaque 500.
    if not isinstance(operand, str):
        raise HTTPException(
            500,
            f"Auth handler filter on the '{key}' column must compare against a string. "
            "Check the filter returned by your auth handler.",
        )
    return target == operand


def _compile_field(
    column: InstrumentedAttribute[Any],
    key: str,
    value: Any,
) -> ColumnElement[bool]:
    if not isinstance(value, dict):
        # Bare value -> equality via JSONB containment: metadata @> {key: value}.
        return column.op("@>")({key: value})

    if len(value) != 1:
        raise HTTPException(
            500,
            "Auth handler filter value must be a dict with exactly one operator key "
            "($eq or $contains). Check the filter returned by your auth handler.",
        )
    operator, operand = next(iter(value.items()))
    if operator == "$eq":
        return column.op("@>")({key: operand})
    if operator == "$contains":
        items = operand if isinstance(operand, list) else [operand]
        # metadata->'key' @> '[items...]' : the array at `key` contains every item.
        return column[key].op("@>")(items)
    raise HTTPException(
        500,
        f"Auth handler filter operator '{operator}' is not supported. "
        "Use $eq or $contains. Check the filter returned by your auth handler.",
    )
