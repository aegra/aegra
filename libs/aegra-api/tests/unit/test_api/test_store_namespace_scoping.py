"""Tests for store namespace scoping / user isolation."""

from aegra_api.api.store import apply_user_namespace_scoping


class TestApplyUserNamespaceScoping:
    """Verify that apply_user_namespace_scoping enforces per-user isolation."""

    def test_empty_namespace_defaults_to_user_prefix(self) -> None:
        result = apply_user_namespace_scoping("user-123", None,[])
        assert result == ["users", "user-123"]

    def test_own_namespace_passes_through(self) -> None:
        ns = ["users", "user-123", "documents"]
        result = apply_user_namespace_scoping("user-123", None,ns)
        assert result == ["users", "user-123", "documents"]

    def test_own_namespace_exact_passes_through(self) -> None:
        ns = ["users", "user-123"]
        result = apply_user_namespace_scoping("user-123", None,ns)
        assert result == ["users", "user-123"]

    def test_other_user_namespace_is_scoped(self) -> None:
        """A user cannot access another user's namespace — it gets remapped."""
        ns = ["users", "victim-456", "secrets"]
        result = apply_user_namespace_scoping("user-123", None,ns)
        assert result == ["users", "user-123", "users", "victim-456", "secrets"]
        assert result != ns

    def test_other_user_namespace_no_passthrough(self) -> None:
        """Ensure attacker-supplied namespace for another user is never returned as-is."""
        ns = ["users", "victim-456"]
        result = apply_user_namespace_scoping("user-123", None,ns)
        assert result[1] == "user-123"

    def test_arbitrary_namespace_is_scoped_under_user(self) -> None:
        """Non-user namespaces get prefixed with the caller's user scope."""
        ns = ["global", "shared-data"]
        result = apply_user_namespace_scoping("user-123", None,ns)
        assert result == ["users", "user-123", "global", "shared-data"]

    def test_single_element_namespace_is_scoped(self) -> None:
        ns = ["configs"]
        result = apply_user_namespace_scoping("user-123", None,ns)
        assert result == ["users", "user-123", "configs"]

    def test_users_prefix_without_id_is_scoped(self) -> None:
        """["users"] alone (no user_id) should be scoped, not passed through."""
        ns = ["users"]
        result = apply_user_namespace_scoping("user-123", None,ns)
        assert result == ["users", "user-123", "users"]

    def test_users_prefix_with_wrong_id_is_scoped(self) -> None:
        ns = ["users", "other-user"]
        result = apply_user_namespace_scoping("user-123", None,ns)
        assert result[1] == "user-123"

    def test_deeply_nested_own_namespace(self) -> None:
        ns = ["users", "user-123", "a", "b", "c"]
        result = apply_user_namespace_scoping("user-123", None,ns)
        assert result == ["users", "user-123", "a", "b", "c"]

    # --- tenant 维度 ---

    def test_empty_namespace_with_tenant(self) -> None:
        result = apply_user_namespace_scoping("user-123", "tenant-a", [])
        assert result == ["tenants", "tenant-a", "users", "user-123"]

    def test_namespace_scoped_under_tenant_prefix(self) -> None:
        result = apply_user_namespace_scoping("user-123", "tenant-a", ["docs"])
        assert result == ["tenants", "tenant-a", "users", "user-123", "docs"]

    def test_own_tenant_namespace_passes_through(self) -> None:
        ns = ["tenants", "tenant-a", "users", "user-123", "docs"]
        result = apply_user_namespace_scoping("user-123", "tenant-a", ns)
        assert result == ns

    def test_cross_tenant_namespace_is_rescoped(self) -> None:
        """带他人 tenant 前缀的 namespace 被强制包在自己的 tenant 下,无法越权。"""
        ns = ["tenants", "victim-tenant", "users", "user-123", "secrets"]
        result = apply_user_namespace_scoping("user-123", "tenant-a", ns)
        assert result[:2] == ["tenants", "tenant-a"]
        assert result != ns
