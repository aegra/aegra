"""Thread queries must compile handler filters, not just their ``metadata`` key.

Thread search accepted only ``{"metadata": {...}}`` and silently discarded the
flat shape and every operator, while ``list_threads`` computed the filter and
threw it away (``if filters: pass``). Assistants already used
``build_metadata_filter``; these tests hold threads to the same contract.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from aegra_api.core.auth_filters import build_metadata_filter
from aegra_api.core.orm import Thread as ThreadORM


def _compiled_where(filters: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Return the rendered SQL and its bound parameters.

    JSONB operands have no literal renderer, so the values live in the bind
    params rather than the SQL text.
    """
    clause = build_metadata_filter(ThreadORM.metadata_json, filters)
    stmt = select(ThreadORM).where(ThreadORM.user_id == "u1")
    if clause is not None:
        stmt = stmt.where(clause)
    compiled = stmt.compile(dialect=postgresql.dialect())
    return str(compiled), dict(compiled.params)


def test_flat_filter_shape_is_compiled() -> None:
    """`{"team_id": "t1"}` must reach SQL, not be dropped for lacking a metadata key."""
    sql, params = _compiled_where({"team_id": "t1"})

    assert "metadata" in sql
    assert {"team_id": "t1"} in params.values()


def test_metadata_envelope_shape_is_compiled() -> None:
    """The historical nested envelope keeps working."""
    _, params = _compiled_where({"metadata": {"team_id": "t1"}})

    assert {"team_id": "t1"} in params.values()


@pytest.mark.parametrize(
    "filters",
    [
        {"team_id": {"$eq": "t1"}},
        {"tags": {"$contains": "x"}},
        {"$or": [{"team_id": "a"}, {"team_id": "b"}]},
        {"$and": [{"team_id": "a"}, {"tier": "pro"}]},
    ],
)
def test_operators_are_not_silently_dropped(filters: dict[str, Any]) -> None:
    """Every documented operator must produce a predicate."""
    clause = build_metadata_filter(ThreadORM.metadata_json, filters)

    assert clause is not None, f"{filters} compiled to no predicate"


def test_no_filter_adds_no_predicate() -> None:
    """An empty or absent filter must not narrow the query."""
    assert build_metadata_filter(ThreadORM.metadata_json, None) is None
    assert build_metadata_filter(ThreadORM.metadata_json, {}) is None


def test_user_scope_survives_alongside_handler_filter() -> None:
    """The handler filter narrows further; it never replaces the ownership check."""
    sql, params = _compiled_where({"team_id": "t1"})

    assert "user_id" in sql
    assert "u1" in params.values()
    assert {"team_id": "t1"} in params.values()
