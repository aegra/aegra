"""Unit tests for the auth filter builder (core/auth_filters.py)."""

import pytest
from fastapi import HTTPException
from sqlalchemy.dialects import postgresql

from aegra_api.core.auth_filters import build_metadata_filter
from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Thread as ThreadORM

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


def _thread_sql(filters: dict) -> tuple[str, dict]:
    clause = build_metadata_filter(ThreadORM.metadata_json, filters)
    assert clause is not None
    compiled = clause.compile(dialect=postgresql.dialect())
    return str(compiled), dict(compiled.params)


class TestColumnRouting:
    """A top-level key that names a real column filters that column, not the metadata JSONB (#360)."""

    @pytest.mark.parametrize("key", ["graph_id", "name"])
    def test_assistant_column_key_filters_the_column(self, key: str) -> None:
        sql, params = _sql({key: "agent-v2"})

        assert f"assistant.{key} = " in sql
        assert "@>" not in sql
        assert "agent-v2" in params.values()

    def test_explicit_eq_filters_the_column(self) -> None:
        sql, params = _sql({"graph_id": {"$eq": "agent-v2"}})

        assert "assistant.graph_id = " in sql
        assert "@>" not in sql
        assert "agent-v2" in params.values()

    def test_thread_status_filters_the_column(self) -> None:
        sql, params = _thread_sql({"status": "idle"})

        assert "thread.status = " in sql
        assert "@>" not in sql
        assert "idle" in params.values()

    def test_column_and_metadata_keys_are_anded(self) -> None:
        sql, params = _sql({"graph_id": "agent-v2", "owner": "u1"})

        assert "assistant.graph_id = " in sql
        assert " AND " in sql
        assert {"owner": "u1"} in params.values()

    def test_column_key_inside_an_or_branch(self) -> None:
        sql, params = _sql({"$or": [{"graph_id": "agent-v2"}, {"owner": "u1"}]})

        assert "assistant.graph_id = " in sql
        assert " OR " in sql
        assert {"owner": "u1"} in params.values()

    def test_key_that_is_a_column_on_another_resource_stays_in_metadata(self) -> None:
        """Threads carry graph_id in their metadata; assistants have no status column."""
        thread_sql, thread_params = _thread_sql({"graph_id": "agent-v2"})
        assistant_sql, assistant_params = _sql({"status": "idle"})

        assert "thread.graph_id" not in thread_sql
        assert {"graph_id": "agent-v2"} in thread_params.values()
        assert "assistant.status" not in assistant_sql
        assert {"status": "idle"} in assistant_params.values()

    def test_metadata_envelope_keeps_a_column_name_in_metadata(self) -> None:
        """The envelope is how a handler says 'the metadata key', even when a column has that name."""
        sql, params = _sql({"metadata": {"graph_id": "agent-v2"}})

        assert "assistant.graph_id = " not in sql
        assert {"graph_id": "agent-v2"} in params.values()

    def test_envelope_and_top_level_key_filter_metadata_and_column(self) -> None:
        sql, params = _sql({"metadata": {"graph_id": "in-metadata"}, "graph_id": "in-column"})

        assert "assistant.graph_id = " in sql
        assert "in-column" in params.values()
        assert {"graph_id": "in-metadata"} in params.values()

    def test_sibling_still_wins_over_the_envelope_for_a_metadata_key(self) -> None:
        _, params = _sql({"metadata": {"team": "from-envelope"}, "team": "from-sibling"})

        assert {"team": "from-sibling"} in params.values()
        assert {"team": "from-envelope"} not in params.values()

    @pytest.mark.parametrize(
        "bad",
        [
            {"graph_id": {"$contains": "agent"}},  # a text column has nothing to contain
            {"graph_id": {"$bad": "agent"}},
            {"graph_id": {"$eq": "a", "$contains": "b"}},
            {"graph_id": 7},  # text column, non-string operand
            {"graph_id": {"$eq": None}},
            {"name": ["a", "b"]},
        ],
    )
    def test_invalid_column_filters_raise_500(self, bad: dict) -> None:
        with pytest.raises(HTTPException) as exc:
            build_metadata_filter(COL, bad)

        assert exc.value.status_code == 500
        assert next(iter(bad)) in exc.value.detail
