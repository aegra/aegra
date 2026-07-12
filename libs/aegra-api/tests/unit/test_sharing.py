"""Tests for assistant sharing visibility (core/sharing.py)."""

from aegra_api.core.sharing import grantees, visible_assistant_filter
from aegra_api.models.auth import User


class TestGrantees:
    """grantees() lists the grant targets matching a user."""

    def test_without_tenant(self) -> None:
        assert grantees(User(user_id="u1")) == ["public", "user:u1"]

    def test_with_tenant(self) -> None:
        assert grantees(User(user_id="u1", tenant_id="t1")) == ["public", "user:u1", "tenant:t1"]


class TestVisibleAssistantFilter:
    """visible_assistant_filter() = owner composite isolation OR system OR shared (subquery)."""

    def test_combines_owner_and_shared_subquery(self) -> None:
        sql = str(visible_assistant_filter(User(user_id="u1", tenant_id="t1")))
        assert " OR " in sql
        assert "assistant_share" in sql
        assert "grantee" in sql

    def test_includes_system_branch(self) -> None:
        sql = str(visible_assistant_filter(User(user_id="u1")))
        assert sql.count("user_id") >= 2
