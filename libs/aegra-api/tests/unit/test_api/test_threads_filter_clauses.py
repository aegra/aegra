"""Unit tests for thread query filter clauses and count endpoint in isolation."""

from typing import Any, Self
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from sqlalchemy.sql.elements import BinaryExpression

from aegra_api.api.threads import _build_thread_filter_clauses, count_threads
from aegra_api.core.auth_deps import User
from aegra_api.models import ThreadCountRequest


def _clause_left_name(clause: Any) -> str:
    """Return the column name from the left side of a binary expression."""
    left = getattr(clause, "left", None)
    return getattr(left, "key", None) or getattr(left, "name", "") or str(left)


def _clause_right_value(clause: Any) -> Any:
    """Return the value from the right side of a binary expression."""
    right = getattr(clause, "right", None)
    return getattr(right, "value", None)


class TestBuildThreadFilterClauses:
    """Unit tests for _build_thread_filter_clauses."""

    def test_user_id_clause_is_always_present(self: Self) -> None:
        """Every filter set must scope queries to the authenticated caller."""
        clauses = _build_thread_filter_clauses("usr-123", {})
        assert len(clauses) == 1
        assert isinstance(clauses[0], BinaryExpression)
        assert _clause_left_name(clauses[0]) == "user_id"
        assert _clause_right_value(clauses[0]) == "usr-123"

    def test_status_clause_added_when_provided(self: Self) -> None:
        """Status filter produces an equality condition on thread.status."""
        clauses = _build_thread_filter_clauses("usr-123", {}, status="idle")
        status_clauses = [c for c in clauses if _clause_left_name(c) == "status"]
        assert len(status_clauses) == 1
        assert _clause_right_value(status_clauses[0]) == "idle"

    def test_metadata_clause_added_when_provided(self: Self) -> None:
        """Metadata filter produces a JSONB containment expression."""
        meta = {"env": "production", "team": "platform"}
        clauses = _build_thread_filter_clauses("usr-123", {}, metadata=meta)
        meta_clauses = [c for c in clauses if "metadata_json" in _clause_left_name(c)]
        assert len(meta_clauses) == 1
        assert _clause_right_value(meta_clauses[0]) == meta

    def test_auth_filter_clause_added_when_present(self: Self) -> None:
        """Auth handler filter dictionary produces an auth predicate."""
        filters = {"metadata": {"department": "engineering"}}
        clauses = _build_thread_filter_clauses("usr-123", filters)
        assert len(clauses) >= 2

    def test_unsupported_values_filter_raises_400(self: Self) -> None:
        """Non-empty state values filter is explicitly rejected with HTTP 400."""
        with pytest.raises(HTTPException) as exc_info:
            _build_thread_filter_clauses("usr-123", {}, values={"state_key": "val"})
        assert exc_info.value.status_code == 400
        assert "not currently supported" in exc_info.value.detail

    def test_empty_or_none_values_filter_does_not_raise(self: Self) -> None:
        """None and empty dict values filters are accepted as no-op."""
        clauses_none = _build_thread_filter_clauses("usr-123", {}, values=None)
        assert len(clauses_none) == 1

        clauses_empty = _build_thread_filter_clauses("usr-123", {}, values={})
        assert len(clauses_empty) == 1


class TestCountThreadsUnit:
    """Isolated unit test for count_threads endpoint handler."""

    @pytest.mark.asyncio
    async def test_count_threads_executes_scalar_query(self: Self) -> None:
        """count_threads calls session.scalar with aggregate COUNT statement."""
        mock_session = AsyncMock()
        mock_session.scalar.return_value = 42

        user = User(identity="usr-abc", is_authenticated=True)
        request = ThreadCountRequest(status="idle", metadata={"env": "prod"})

        result = await count_threads(request=request, user=user, session=mock_session)
        assert result == 42
        mock_session.scalar.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_count_threads_returns_zero_when_scalar_is_none(self: Self) -> None:
        """count_threads returns 0 when the scalar query returns None."""
        mock_session = AsyncMock()
        mock_session.scalar.return_value = None

        user = User(identity="usr-abc", is_authenticated=True)
        request = ThreadCountRequest()

        result = await count_threads(request=request, user=user, session=mock_session)
        assert result == 0
