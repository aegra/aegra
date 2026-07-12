"""Tests for assistant sharing visibility (core/sharing.py)。"""

from aegra_api.core.sharing import shared_to_user, visible_assistant_filter
from aegra_api.models.auth import User


class TestSharedToUser:
    """shared_to_user() branches on whether the user has a tenant."""

    def test_includes_public_and_user_target(self) -> None:
        sql = str(shared_to_user(User(user_id="u1")))
        assert "share_type" in sql
        assert "target_user_id" in sql

    def test_no_tenant_excludes_tenant_branch(self) -> None:
        sql = str(shared_to_user(User(user_id="u1")))
        assert "target_tenant_id" not in sql

    def test_with_tenant_includes_tenant_branch(self) -> None:
        sql = str(shared_to_user(User(user_id="u1", tenant_id="t1")))
        assert "target_tenant_id" in sql


class TestVisibleAssistantFilter:
    """visible_assistant_filter() = owner composite isolation OR system OR shared (subquery)."""

    def test_combines_owner_and_shared_subquery(self) -> None:
        sql = str(visible_assistant_filter(User(user_id="u1", tenant_id="t1")))
        assert " OR " in sql
        assert "assistant_share" in sql

    def test_includes_system_branch(self) -> None:
        sql = str(visible_assistant_filter(User(user_id="u1")))
        # allow_system brings in the user_id == 'system' branch via owner_filter
        assert sql.count("user_id") >= 2
