"""Unit tests for the auth filter builder (core/auth_filters.py)."""

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from aegra_api.core.auth_filters import build_metadata_filter
from aegra_api.core.orm import Assistant as AssistantORM

COL = AssistantORM.metadata_dict


def _sql(filters: dict | None) -> tuple[str, dict]:
    clause = build_metadata_filter(COL, filters)
    if clause is None:
        return "", {}
    compiled = clause.compile(dialect=postgresql.dialect())
    return str(compiled), dict(compiled.params)


class TestBuildMetadataFilter:
    def test_none_and_empty_return_no_clause(self) -> None:
        assert build_metadata_filter(COL, None) is None
        assert build_metadata_filter(COL, {}) is None

    def test_flat_eq_uses_containment(self) -> None:
        sql, params = _sql({"owner": "u1"})
        assert "@>" in sql
        assert {"owner": "u1"} in params.values()

    def test_explicit_eq_operator(self) -> None:
        sql, params = _sql({"owner": {"$eq": "u1"}})
        assert "@>" in sql
        assert {"owner": "u1"} in params.values()

    def test_contains_scalar_wraps_in_list(self) -> None:
        sql, params = _sql({"tags": {"$contains": "x"}})
        assert "->" in sql and "@>" in sql
        assert ["x"] in params.values()

    def test_contains_list(self) -> None:
        _, params = _sql({"tags": {"$contains": ["x", "y"]}})
        assert ["x", "y"] in params.values()

    def test_multiple_keys_are_anded(self) -> None:
        sql, _ = _sql({"owner": "u1", "team": "t1"})
        assert " AND " in sql

    def test_or_operator(self) -> None:
        sql, _ = _sql({"$or": [{"owner": "u1"}, {"owner": "u2"}]})
        assert " OR " in sql

    def test_and_operator(self) -> None:
        sql, _ = _sql({"$and": [{"owner": "u1"}, {"team": "t1"}]})
        assert " AND " in sql

    def test_metadata_envelope_unwrapped(self) -> None:
        """Aegra-compat: {"metadata": {...}} filters the same as flat keys."""
        nested, _ = _sql({"metadata": {"team_id": "t1"}})
        flat, _ = _sql({"team_id": "t1"})
        assert nested == flat

    def test_metadata_envelope_merges_with_siblings(self) -> None:
        sql, params = _sql({"metadata": {"team_id": "t1"}, "owner": "u1"})
        assert " AND " in sql
        assert {"team_id": "t1"} in params.values()
        assert {"owner": "u1"} in params.values()

    @pytest.mark.parametrize(
        "bad",
        [
            {"$or": [{"owner": "u1"}]},  # fewer than 2 branches
            {"$or": "notalist"},
            {"$and": [{"owner": "u1"}]},
            {"k": {"$bad": 1}},  # unknown operator
            {"k": {"$eq": 1, "$contains": 2}},  # more than one operator key
            {"$or": [{}, {"owner": "u1"}]},  # empty branch -> would collapse to TRUE
            {"$and": [{"owner": "u1"}, {}]},
            {"$or": [{"owner": "u1"}, "notadict"]},  # non-dict branch
        ],
    )
    def test_invalid_filters_raise_500(self, bad: dict) -> None:
        with pytest.raises(HTTPException) as exc:
            build_metadata_filter(COL, bad)
        assert exc.value.status_code == 500

    def test_empty_branch_does_not_collapse_to_true(self) -> None:
        """An empty $or branch must not bypass the constraint via and_() == TRUE."""
        with pytest.raises(HTTPException) as exc:
            build_metadata_filter(COL, {"$or": [{}, {"owner": "u1"}]})
        assert exc.value.status_code == 500
        assert "empty" in exc.value.detail.lower()

    def test_nesting_depth_cap(self) -> None:
        """$or inside $or inside $or exceeds the depth-2 cap."""
        deep = {"$or": [{"$or": [{"$or": [{"a": 1}, {"b": 2}]}, {"c": 3}]}, {"d": 4}]}
        with pytest.raises(HTTPException) as exc:
            build_metadata_filter(COL, deep)
        assert exc.value.status_code == 500
