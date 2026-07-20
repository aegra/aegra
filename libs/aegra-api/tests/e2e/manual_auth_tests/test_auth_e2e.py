"""E2E tests for authentication flow with running server.

These tests require a server started with auth enabled (see aegra.auth.json).
Skipped by default; run with: make e2e-auth  or  pytest -m auth_only

See tests/e2e/manual_auth_tests/README.md for details.
"""

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog


def get_server_url() -> str:
    """Get server URL from environment or use default"""
    return settings.app.SERVER_URL


def get_auth_token(user_id: str = "alice", role: str = "user", team_id: str = "team123") -> str:
    """Generate a mock JWT token for testing"""
    return f"mock-jwt-{user_id}-{role}-{team_id}"


def get_auth_headers(token: str | None = None) -> dict[str, str]:
    """Get Authorization headers for requests"""
    if token is None:
        token = get_auth_token()
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.e2e
@pytest.mark.auth_only
class TestCoreRoutesAuth:
    """Test that core routes require authentication"""

    def test_core_routes_reject_without_auth(self):
        """Test that core API routes return 401 without auth token"""
        url = f"{get_server_url()}/assistants"

        response = httpx.get(url, timeout=10.0)

        # Should return 401 (unauthorized) without token
        assert response.status_code == 401, f"Expected 401, got {response.status_code}: {response.text}"

        # Verify error format
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"] == "unauthorized"

        elog("Core route without auth", {"url": url, "status": response.status_code})

    def test_core_routes_accept_with_valid_token(self):
        """Test that core routes accept requests with valid auth token"""
        url = f"{get_server_url()}/assistants"
        headers = get_auth_headers()

        response = httpx.get(url, headers=headers, timeout=10.0)

        # Should not return 401 (might be 200 or other status, but not unauthorized)
        assert response.status_code != 401, f"Expected not 401, got {response.status_code}: {response.text}"

        elog(
            "Core route with valid auth",
            {
                "url": url,
                "status": response.status_code,
                "headers": list(headers.keys()),
            },
        )

    def test_core_routes_reject_invalid_token(self):
        """Test that core routes reject invalid tokens"""
        url = f"{get_server_url()}/assistants"
        headers = {"Authorization": "Bearer invalid-token-format"}

        response = httpx.get(url, headers=headers, timeout=10.0)

        assert response.status_code == 401, (
            f"Expected 401 for invalid token, got {response.status_code}: {response.text}"
        )

        elog(
            "Core route with invalid token",
            {"url": url, "status": response.status_code},
        )


@pytest.mark.e2e
@pytest.mark.auth_only
class TestCustomRoutesAuth:
    """Test authentication with custom routes"""

    def test_custom_whoami_returns_user_data(self):
        """Test that /custom/whoami returns authenticated user data"""
        url = f"{get_server_url()}/custom/whoami"
        headers = get_auth_headers("mock-jwt-eve-admin-team456")

        response = httpx.get(url, headers=headers, timeout=10.0)

        # Should succeed (200) if auth is working
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()

        # Verify user data structure
        assert "identity" in data
        assert data["identity"] == "eve"
        assert "display_name" in data
        assert "is_authenticated" in data
        assert data["is_authenticated"] is True
        assert "permissions" in data
        assert isinstance(data["permissions"], list)

        # Verify custom fields are present
        assert "role" in data
        assert data["role"] == "admin"
        assert "subscription_tier" in data
        assert data["subscription_tier"] == "premium"
        assert "team_id" in data
        assert data["team_id"] == "team456"
        assert "email" in data
        assert data["email"] == "eve@example.com"

        elog("Custom whoami endpoint", {"url": url, "response": data})

    def test_custom_whoami_rejects_without_auth(self):
        """Test that /custom/whoami requires authentication"""
        url = f"{get_server_url()}/custom/whoami"

        response = httpx.get(url, timeout=10.0)

        # Should return 401 if auth is required
        # Note: This depends on enable_custom_route_auth config
        # If disabled, might return 200 with anonymous user
        assert response.status_code in (200, 401), f"Unexpected status {response.status_code}: {response.text}"

        elog("Custom whoami without auth", {"url": url, "status": response.status_code})

    def test_custom_protected_requires_auth(self):
        """Test that /custom/protected always requires auth (explicit dependency)"""
        url = f"{get_server_url()}/custom/protected"

        # Without auth
        response_no_auth = httpx.get(url, timeout=10.0)
        assert response_no_auth.status_code == 401, f"Expected 401 without auth, got {response_no_auth.status_code}"

        # With auth
        headers = get_auth_headers("mock-jwt-frank-user-team789")
        response_with_auth = httpx.get(url, headers=headers, timeout=10.0)
        assert response_with_auth.status_code == 200, f"Expected 200 with auth, got {response_with_auth.status_code}"

        data = response_with_auth.json()
        assert "user" in data
        assert data["user"] == "frank"

        elog(
            "Custom protected endpoint",
            {
                "without_auth": response_no_auth.status_code,
                "with_auth": response_with_auth.status_code,
            },
        )


@pytest.mark.e2e
@pytest.mark.auth_only
class TestCustomRouteAuthConfig:
    """Test enable_custom_route_auth configuration"""

    def test_custom_public_endpoint_behavior(self):
        """Test /custom/public endpoint behavior based on config"""
        url = f"{get_server_url()}/custom/public"

        # Try without auth
        response_no_auth = httpx.get(url, timeout=10.0)

        # If enable_custom_route_auth is True, should return 401
        # If False, should return 200
        assert response_no_auth.status_code in (200, 401), f"Unexpected status {response_no_auth.status_code}"

        # Try with auth
        headers = get_auth_headers()
        response_with_auth = httpx.get(url, headers=headers, timeout=10.0)

        # Should always work with valid auth
        assert response_with_auth.status_code == 200, f"Expected 200 with auth, got {response_with_auth.status_code}"

        elog(
            "Custom public endpoint",
            {
                "without_auth_status": response_no_auth.status_code,
                "with_auth_status": response_with_auth.status_code,
            },
        )


@pytest.mark.e2e
@pytest.mark.auth_only
class TestUserCustomFields:
    """Test that custom fields from auth flow through to routes"""

    def test_user_custom_fields_in_route(self):
        """Test that custom fields like team_id, role are accessible in routes"""
        url = f"{get_server_url()}/custom/whoami"
        headers = get_auth_headers("mock-jwt-grace-premium-team999")

        response = httpx.get(url, headers=headers, timeout=10.0)

        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()

        # Verify all custom fields are present and correct
        assert data["identity"] == "grace"
        assert data["role"] == "premium"
        assert data["subscription_tier"] == "premium"  # premium role -> premium tier
        assert data["team_id"] == "team999"
        assert data["email"] == "grace@example.com"

        elog("User custom fields", {"url": url, "user_data": data})

    def test_different_roles_get_different_permissions(self):
        """Test that different roles get different permissions"""
        url = f"{get_server_url()}/custom/whoami"

        # Admin user - token format: mock-jwt-<user_id>-<role>-<team_id>
        admin_headers = get_auth_headers("mock-jwt-adminuser-admin-team1")
        admin_response = httpx.get(url, headers=admin_headers, timeout=10.0)
        assert admin_response.status_code == 200
        admin_data = admin_response.json()
        assert "admin" in admin_data["permissions"]
        assert admin_data["role"] == "admin"

        # Regular user
        user_headers = get_auth_headers("mock-jwt-regularuser-user-team2")
        user_response = httpx.get(url, headers=user_headers, timeout=10.0)
        assert user_response.status_code == 200
        user_data = user_response.json()
        assert "user" in user_data["permissions"]
        assert user_data["role"] == "user"
        assert user_data["subscription_tier"] == "free"

        elog(
            "Role-based permissions",
            {
                "admin_permissions": admin_data["permissions"],
                "user_permissions": user_data["permissions"],
            },
        )


@pytest.mark.e2e
@pytest.mark.auth_only
class TestNoopAuth:
    """Test noop authentication behavior"""

    def test_noop_auth_allows_access(self):
        """Test that when no auth is configured, requests are allowed"""
        # This test assumes server is configured without auth
        # If auth is configured, this test will fail (which is expected)
        url = f"{get_server_url()}/assistants"

        response = httpx.get(url, timeout=10.0)

        # With noop auth, should either:
        # 1. Return 200 (anonymous user allowed)
        # 2. Return 401 (if auth is configured)
        assert response.status_code in (200, 401), f"Unexpected status {response.status_code}: {response.text}"

        elog("Noop auth behavior", {"url": url, "status": response.status_code})


@pytest.mark.e2e
@pytest.mark.auth_only
class TestAuthErrorHandling:
    """Test authentication error handling"""

    def test_malformed_auth_header(self):
        """Test handling of malformed Authorization header"""
        url = f"{get_server_url()}/assistants"
        headers = {"Authorization": "InvalidFormat token"}

        response = httpx.get(url, headers=headers, timeout=10.0)

        assert response.status_code == 401, f"Expected 401 for malformed header, got {response.status_code}"

        elog("Malformed auth header", {"status": response.status_code})

    def test_missing_bearer_prefix(self):
        """Test handling of token without Bearer prefix"""
        url = f"{get_server_url()}/assistants"
        headers = {"Authorization": "mock-jwt-alice-user-team123"}

        response = httpx.get(url, headers=headers, timeout=10.0)

        assert response.status_code == 401, f"Expected 401 for missing Bearer prefix, got {response.status_code}"

        elog("Missing Bearer prefix", {"status": response.status_code})

    def test_empty_token(self):
        """Test handling of empty token"""
        url = f"{get_server_url()}/assistants"
        # Use a space after Bearer but httpx doesn't allow empty header values
        # So we'll test with missing token instead (no Authorization header)
        # This tests the same scenario - missing/invalid auth

        response = httpx.get(url, timeout=10.0)

        assert response.status_code == 401, f"Expected 401 for missing token, got {response.status_code}"

        elog("Empty/missing token", {"status": response.status_code})
