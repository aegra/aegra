"""Tests for tenant/user authorization filters (core/authz.py)."""

from types import SimpleNamespace

import pytest

from aegra_api.core.authz import owner_filter, owns, query_scope, scope
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models.auth import User


def _user(identity: str, tenant_id: str | None = None) -> User:
    return User(identity=identity, tenant_id=tenant_id)


def _row(user_id: str, tenant_id: str | None) -> SimpleNamespace:
    return SimpleNamespace(user_id=user_id, tenant_id=tenant_id)


class TestOwns:
    """owns() composite isolation: equal user_id + NULL-safe equal tenant_id."""

    def test_same_user_no_tenant(self) -> None:
        assert owns(_row("u1", None), _user("u1")) is True

    def test_same_user_same_tenant(self) -> None:
        assert owns(_row("u1", "t1"), _user("u1", "t1")) is True

    def test_same_user_different_tenant_denied(self) -> None:
        assert owns(_row("u1", "t1"), _user("u1", "t2")) is False

    def test_user_with_tenant_cannot_see_null_tenant_resource(self) -> None:
        assert owns(_row("u1", None), _user("u1", "t1")) is False

    def test_null_tenant_user_cannot_see_tenant_resource(self) -> None:
        assert owns(_row("u1", "t1"), _user("u1")) is False

    def test_different_user_denied(self) -> None:
        assert owns(_row("u2", None), _user("u1")) is False

    def test_system_allowed_when_flagged(self) -> None:
        assert owns(_row("system", None), _user("u1", "t1"), allow_system=True) is True

    def test_system_denied_by_default(self) -> None:
        assert owns(_row("system", None), _user("u1")) is False


class TestScopeSql:
    """scope()/owner_filter() WHERE must constrain both user_id and tenant_id."""

    def test_scope_constrains_user_and_tenant(self) -> None:
        sql = str(scope(ThreadORM, "u1", "t1"))
        assert "user_id" in sql
        assert "tenant_id" in sql

    def test_scope_is_null_safe(self) -> None:
        sql = str(scope(ThreadORM, "u1", None))
        assert "DISTINCT" in sql.upper()

    def test_owner_filter_matches_scope_from_user(self) -> None:
        sql = str(owner_filter(ThreadORM, _user("u1", "t1")))
        assert "user_id" in sql
        assert "tenant_id" in sql

    def test_allow_system_adds_system_branch(self) -> None:
        # allow_system adds a user_id == "system" branch via OR (a bound param, not a literal)
        sql = str(scope(ThreadORM, "u1", None, allow_system=True))
        assert " OR " in sql
        assert sql.count("user_id") >= 2


class TestUserIdAlias:
    """user_id is the canonical primary field (requirement A); identity is an SDK-compat alias property."""

    def test_identity_input_maps_to_user_id(self) -> None:
        assert User(identity="alice").user_id == "alice"

    def test_accepts_user_id_as_input(self) -> None:
        u = User(user_id="bob")
        assert u.identity == "bob"
        assert u.user_id == "bob"

    def test_user_id_takes_precedence_when_both(self) -> None:
        assert User(identity="alice", user_id="bob").user_id == "bob"


class TestQueryScope:
    """query_scope() three modes: user_id only / tenant_id only / composite (requirement B)."""

    def test_user_only_filters_user_not_tenant(self) -> None:
        sql = str(query_scope(ThreadORM, user_id="u1"))
        assert "user_id" in sql
        assert "tenant_id" not in sql

    def test_tenant_only_filters_tenant_not_user(self) -> None:
        sql = str(query_scope(ThreadORM, tenant_id="t1"))
        assert "tenant_id" in sql
        assert "user_id" not in sql

    def test_composite_filters_both(self) -> None:
        sql = str(query_scope(ThreadORM, user_id="u1", tenant_id="t1"))
        assert "user_id" in sql
        assert "tenant_id" in sql

    def test_raises_when_no_dimension_given(self) -> None:
        with pytest.raises(ValueError, match="at least one filter dimension"):
            query_scope(ThreadORM)

    def test_allow_system_adds_system_branch(self) -> None:
        sql = str(query_scope(ThreadORM, user_id="u1", allow_system=True))
        assert " OR " in sql
        assert sql.count("user_id") >= 2
