"""SQL helpers for merging into JSONB columns."""

from typing import Any

from sqlalchemy import BindParameter, ColumnElement, bindparam, case, func, literal_column
from sqlalchemy.orm import InstrumentedAttribute

from aegra_api.core.orm import JsonbSafe


def jsonb_patch(patch: dict[str, Any], name: str) -> BindParameter[Any]:
    """Bind *patch* as a JSONB parameter.

    ``JsonbSafe`` strips NUL bytes, which Postgres rejects in ``jsonb``; binding
    a patch any other way would fail writes that the columns themselves accept.

    *name* must be unique within the statement the parameter is used in.
    """
    return bindparam(name, value=patch, type_=JsonbSafe)


def jsonb_shallow_merge(column: InstrumentedAttribute[Any], *patches: ColumnElement[Any]) -> ColumnElement[Any]:
    """Merge *patches* into a JSONB *column*, top level only, inside the database.

    Renders ``CASE WHEN jsonb_typeof(col) = 'object' THEN col ELSE '{}'::jsonb
    END || :patch``. Postgres reads and writes the column in one statement, so
    the row lock the ``UPDATE`` takes serializes concurrent writers instead of
    letting them overwrite each other's keys. Later *patches* win over earlier
    ones, as successive ``dict.update`` calls do.

    The type guard matters because ``'[1,2]'::jsonb || '{"a":1}'::jsonb`` appends
    to an array instead of merging: a column holding a non-object (or SQL NULL)
    starts from ``{}``, which is what ``dict(value or {})`` meant in Python.
    """
    merged: ColumnElement[Any] = case(
        (func.jsonb_typeof(column) == "object", column),
        else_=literal_column("'{}'::jsonb"),
    )
    for patch in patches:
        merged = merged.op("||", return_type=JsonbSafe)(patch)
    return merged
