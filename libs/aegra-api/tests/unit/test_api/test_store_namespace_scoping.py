"""Tests for store namespace scoping."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import HTTPException

from aegra_api.api.store import _scope_attr_map, apply_namespace_scoping
from aegra_api.models import User


def _user(identity: str, *, org_id: str | None = None, **extra: Any | None) -> User:
    """Build a User with arbitrary scope attributes for scoping tests."""
    return User(identity=identity, org_id=org_id, **extra)


class TestUserNamespaceScoping:
    """Verify that apply_namespace_scoping enforces per-user isolation."""

    def test_empty_namespace_defaults_to_user_prefix(self) -> None:
        """An empty namespace resolves to the caller's user scope."""
        result = apply_namespace_scoping([], _user("user-123"))
        assert result == ["users", "user-123"]

    def test_own_namespace_passes_through(self) -> None:
        """A namespace already under the caller's user scope is left unchanged."""
        ns = ["users", "user-123", "documents"]
        result = apply_namespace_scoping(ns, _user("user-123"))
        assert result == ["users", "user-123", "documents"]

    def test_own_namespace_exact_passes_through(self) -> None:
        """The exact user-scope namespace passes through without duplication."""
        ns = ["users", "user-123"]
        result = apply_namespace_scoping(ns, _user("user-123"))
        assert result == ["users", "user-123"]

    def test_other_user_namespace_is_scoped(self) -> None:
        """A user cannot access another user's namespace — it gets remapped."""
        ns = ["users", "victim-456", "secrets"]
        result = apply_namespace_scoping(ns, _user("user-123"))
        assert result == ["users", "user-123", "users", "victim-456", "secrets"]
        assert result != ns

    def test_other_user_namespace_no_passthrough(self) -> None:
        """Ensure attacker-supplied namespace for another user is never returned as-is."""
        ns = ["users", "victim-456"]
        result = apply_namespace_scoping(ns, _user("user-123"))
        assert result[1] == "user-123"

    def test_arbitrary_namespace_is_scoped_under_user(self) -> None:
        """Non-user namespaces get prefixed with the caller's user scope."""
        ns = ["global", "shared-data"]
        result = apply_namespace_scoping(ns, _user("user-123"))
        assert result == ["users", "user-123", "global", "shared-data"]

    def test_single_element_namespace_is_scoped(self) -> None:
        """A single-element namespace is buried under the caller's user scope."""
        ns = ["configs"]
        result = apply_namespace_scoping(ns, _user("user-123"))
        assert result == ["users", "user-123", "configs"]

    def test_users_prefix_without_id_is_scoped(self) -> None:
        """["users"] alone (no user_id) should be scoped, not passed through."""
        ns = ["users"]
        result = apply_namespace_scoping(ns, _user("user-123"))
        assert result == ["users", "user-123", "users"]

    def test_users_prefix_with_wrong_id_is_scoped(self) -> None:
        """A "users" prefix bearing another id is re-scoped to the caller."""
        ns = ["users", "other-user"]
        result = apply_namespace_scoping(ns, _user("user-123"))
        assert result[1] == "user-123"

    def test_deeply_nested_own_namespace(self) -> None:
        """A deeply nested namespace under the caller's scope passes through intact."""
        ns = ["users", "user-123", "a", "b", "c"]
        result = apply_namespace_scoping(ns, _user("user-123"))
        assert result == ["users", "user-123", "a", "b", "c"]


class TestConfigurableScopes:
    """Verify scopes can map any namespace prefix to any User attribute (e.g. org_id)."""

    def test_fully_qualified_org_namespace_passes_through(self) -> None:
        """A namespace already under the caller's own org value passes through."""
        scopes = {"orgs": "org_id"}
        user = _user("user-123", org_id="org-1")
        result = apply_namespace_scoping(["orgs", "org-1", "shared-prompts"], user, scopes=scopes)
        assert result == ["orgs", "org-1", "shared-prompts"]

    def test_other_org_namespace_is_buried(self) -> None:
        """A foreign org id never passes through — it is buried under the caller's org."""
        scopes = {"orgs": "org_id"}
        user = _user("user-123", org_id="org-1")
        result = apply_namespace_scoping(["orgs", "victim-org", "secrets"], user, scopes=scopes)
        assert result == ["orgs", "org-1", "orgs", "victim-org", "secrets"]
        assert result[1] == "org-1"

    def test_user_scope_takes_precedence_for_empty_namespace(self) -> None:
        """Empty namespace defaults to user scope even when a scope is configured."""
        scopes = {"orgs": "org_id"}
        user = _user("user-123", org_id="org-1")
        result = apply_namespace_scoping([], user, scopes=scopes)
        assert result == ["users", "user-123"]

    def test_custom_scope_buries_foreign_value(self) -> None:
        """A foreign value under a custom scope is buried under the caller's value."""
        scopes = {"teams": "team_id"}
        user = _user("user-123", team_id="team-42")
        result = apply_namespace_scoping(["teams", "other-team"], user, scopes=scopes)
        assert result == ["teams", "team-42", "teams", "other-team"]

    def test_custom_scope_missing_attribute_raises_403(self) -> None:
        """Using a scope the user lacks the backing attribute for is forbidden."""
        scopes = {"teams": "team_id"}
        user = _user("user-123")
        with pytest.raises(HTTPException) as exc_info:
            apply_namespace_scoping(["teams", "x"], user, scopes=scopes)
        assert exc_info.value.status_code == 403

    def test_custom_scope_empty_string_attribute_raises_403(self) -> None:
        """An empty-string scope value is falsy and must be rejected, like a missing attribute."""
        scopes = {"orgs": "org_id"}
        user = _user("user-123", org_id="")
        with pytest.raises(HTTPException) as exc_info:
            apply_namespace_scoping(["orgs", "x"], user, scopes=scopes)
        assert exc_info.value.status_code == 403

    def test_custom_scope_zero_attribute_is_not_rejected(self) -> None:
        """Integer 0 is a valid primary-key value; only None/empty means 'attribute missing'."""
        scopes = {"tenants": "tenant_id"}
        user = _user("user-123", tenant_id=0)
        result = apply_namespace_scoping(["tenants", "ignored"], user, scopes=scopes)
        assert result == ["tenants", "0", "tenants", "ignored"]

    def test_scope_prefix_alone_is_buried_not_rejected(self) -> None:
        """[prefix] with no id is buried under the caller's scope, not treated as an error."""
        scopes = {"orgs": "org_id"}
        user = _user("user-123", org_id="org-1")
        result = apply_namespace_scoping(["orgs"], user, scopes=scopes)
        assert result == ["orgs", "org-1", "orgs"]

    def test_non_string_attribute_value_is_coerced(self) -> None:
        """A non-string scope value is coerced to its string form in the namespace."""
        scopes = {"tenants": "tenant_id"}
        user = _user("user-123", tenant_id=7)
        result = apply_namespace_scoping(["tenants", "ignored"], user, scopes=scopes)
        assert result == ["tenants", "7", "tenants", "ignored"]

    def test_unconfigured_prefix_falls_back_to_user_scope(self) -> None:
        """A prefix not in the scopes map is treated as ordinary user-scoped data."""
        scopes = {"teams": "team_id"}
        user = _user("user-123")
        result = apply_namespace_scoping(["orgs", "org-1"], user, scopes=scopes)
        assert result == ["users", "user-123", "orgs", "org-1"]

    def test_multiple_scopes(self) -> None:
        """Each configured scope resolves against its own User attribute."""
        scopes = {"orgs": "org_id", "teams": "team_id"}
        user = _user("user-123", org_id="org-1", team_id="team-42")
        assert apply_namespace_scoping(["orgs", "org-1"], user, scopes=scopes) == ["orgs", "org-1"]
        assert apply_namespace_scoping(["teams", "team-42"], user, scopes=scopes) == ["teams", "team-42"]


class TestScopeAttrMap:
    """Verify the aegra.json store.scopes loader, including the reserved-prefix guard."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> Iterator[None]:
        """Reset the @cache on _scope_attr_map around each test."""
        _scope_attr_map.cache_clear()
        yield
        _scope_attr_map.cache_clear()

    def test_no_config_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No store config yields no extra scopes."""
        monkeypatch.setattr("aegra_api.api.store.load_store_config", lambda: None)
        assert _scope_attr_map() == {}

    def test_config_without_scopes_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A store config without a scopes key yields no extra scopes."""
        monkeypatch.setattr("aegra_api.api.store.load_store_config", lambda: {})
        assert _scope_attr_map() == {}

    def test_configured_scopes_are_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Configured scopes are surfaced verbatim from the store config."""
        monkeypatch.setattr("aegra_api.api.store.load_store_config", lambda: {"scopes": {"orgs": "org_id"}})
        assert _scope_attr_map() == {"orgs": "org_id"}

    def test_reserved_users_prefix_is_dropped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The "users" prefix is reserved for per-user isolation and can never be remapped."""
        monkeypatch.setattr(
            "aegra_api.api.store.load_store_config",
            lambda: {"scopes": {"users": "org_id", "orgs": "org_id"}},
        )
        assert _scope_attr_map() == {"orgs": "org_id"}
