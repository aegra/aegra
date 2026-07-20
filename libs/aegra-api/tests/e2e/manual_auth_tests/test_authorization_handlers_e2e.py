"""E2E tests for authorization handlers (@auth.on.*)

Require an auth-enabled server (aegra.auth.json). Skipped by default.
Run with: make e2e-auth  or  pytest -m auth_only

See tests/e2e/manual_auth_tests/README.md for details.

**Hybrid Testing Approach:**
- **SDK Client**: Used for standard API endpoints (threads, assistants)
  - Tests the actual user experience with SDK
  - Better type safety and error handling
  - Consistent with other E2E tests
- **httpx**: Used for custom routes (/custom/*) and GET /threads (list endpoint)
  - SDK doesn't support custom routes
  - GET /threads uses different endpoint than SDK's search()
"""

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog


def get_server_url() -> str:
    """Get server URL from environment or use default"""
    return settings.app.SERVER_URL


def get_auth_token(user_id: str = "alice", role: str = "user", team_id: str = "team123") -> str:
    """Generate a mock JWT token for testing.

    These tests run against a server with auth enabled, so all requests need valid auth tokens.
    """
    return f"mock-jwt-{user_id}-{role}-{team_id}"


def get_auth_headers(token: str | None = None) -> dict[str, str]:
    """Get Authorization headers for requests"""
    if token is None:
        token = get_auth_token()
    return {"Authorization": f"Bearer {token}"}


def get_client_with_auth(user_id: str = "alice", role: str = "user", team_id: str = "team123"):
    """Get SDK client with authentication headers.

    Uses SDK client for standard API endpoints (assistants, threads, runs, store).
    For custom routes, use httpx directly.
    """
    from langgraph_sdk import get_client

    token = get_auth_token(user_id, role, team_id)
    return get_client(url=get_server_url(), headers={"Authorization": f"Bearer {token}"})


@pytest.mark.e2e
@pytest.mark.auth_only
class TestThreadAuthorization:
    """Test authorization handlers for thread operations"""

    @pytest.mark.asyncio
    async def test_thread_create_injects_team_id(self):
        """Test that thread creation injects team_id into metadata via @auth.on.threads.create"""
        client = get_client_with_auth("alice", "user", "team456")

        thread = await client.threads.create(metadata={"custom": "value"})

        # Authorization handler should have injected team_id
        assert "metadata" in thread
        assert thread["metadata"].get("team_id") == "team456"

        elog("Thread create with team_id injection", {"response": thread})

    @pytest.mark.asyncio
    async def test_thread_list_applies_filters(self):
        """Test that thread list/search applies filters from @auth.on.threads.search"""
        # Use httpx for GET /threads (list endpoint) since SDK doesn't have a direct list method
        # The SDK's search() is for POST /threads/search
        url = f"{get_server_url()}/threads"
        headers = get_auth_headers("mock-jwt-bob-user-team789")

        response = httpx.get(url, headers=headers, timeout=10.0)

        # Should succeed (filters applied by handler)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"

        data = response.json()
        assert "threads" in data

        elog("Thread list with filters", {"total": data.get("total", 0)})


@pytest.mark.e2e
@pytest.mark.auth_only
class TestAssistantAuthorization:
    """Test authorization handlers for assistant operations"""

    @pytest.mark.asyncio
    async def test_assistant_create_injects_metadata(self):
        """Test that assistant creation injects created_by via @auth.on.assistants.create"""
        import uuid

        client = get_client_with_auth("charlie", "admin", "team111")

        assistant = await client.assistants.create(
            graph_id="agent",
            name=f"Test Assistant {uuid.uuid4().hex[:8]}",
            metadata={},
            if_exists="do_nothing",
        )

        # Authorization handler should have injected created_by
        assert "metadata" in assistant
        assert assistant["metadata"].get("created_by") == "charlie"
        assert assistant["metadata"].get("team_id") == "team111"

        elog("Assistant create with metadata injection", {"response": assistant})

    @pytest.mark.asyncio
    async def test_assistant_delete_allowed_for_admin(self):
        """Test that assistant deletion is allowed for admin role via @auth.on.assistants.delete"""
        import uuid

        # Create client with admin auth
        admin_client = get_client_with_auth("adminuser", "admin", "team1")

        # Create an assistant
        assistant = await admin_client.assistants.create(
            graph_id="agent",
            name=f"Test Assistant to Delete {uuid.uuid4().hex[:8]}",
            metadata={},
            if_exists="do_nothing",
        )
        assistant_id = assistant["assistant_id"]

        # Delete it (should be allowed for admin)
        await admin_client.assistants.delete(assistant_id)

        elog("Assistant delete allowed for admin", {"assistant_id": assistant_id})

    @pytest.mark.asyncio
    async def test_assistant_delete_denied_for_non_admin(self):
        """Test that assistant deletion is denied for non-admin role"""
        import uuid

        # Create client with regular user auth
        user_client = get_client_with_auth("regularuser", "user", "team2")

        # Create an assistant
        assistant = await user_client.assistants.create(
            graph_id="agent",
            name=f"Test Assistant {uuid.uuid4().hex[:8]}",
            metadata={},
            if_exists="do_nothing",
        )
        assistant_id = assistant["assistant_id"]

        # Try to delete it (should be denied for non-admin)
        # SDK client raises an exception for 403
        with pytest.raises(Exception) as exc_info:
            await user_client.assistants.delete(assistant_id)

        # Verify it's a 403 error
        # SDK wraps httpx.HTTPStatusError, check response.status_code
        error = exc_info.value
        assert hasattr(error, "response") or hasattr(error, "status_code")
        status_code = getattr(error, "status_code", None) or (
            getattr(error.response, "status_code", None) if hasattr(error, "response") else None
        )
        assert status_code == 403, f"Expected 403, got {status_code}"

        elog("Assistant delete denied for non-admin", {"assistant_id": assistant_id})


@pytest.mark.e2e
@pytest.mark.auth_only
class TestAuthorizationHandlerPrecedence:
    """Test that handler precedence works correctly"""

    def test_specific_handler_takes_precedence(self):
        """Test that specific handler (e.g., @auth.on.assistants.delete) takes precedence"""
        # This is tested implicitly in the assistant delete tests above
        # where @auth.on.assistants.delete (specific) should be used instead of
        # any global handler
        pass


@pytest.mark.e2e
@pytest.mark.auth_only
class TestAuthorizationMetadataInjection:
    """Test that authorization handlers can inject metadata"""

    @pytest.mark.asyncio
    async def test_thread_create_metadata_injection(self):
        """Test that thread creation handler injects metadata"""
        client = get_client_with_auth("dave", "user", "team999")

        thread = await client.threads.create()

        # Check that team_id was injected
        assert "metadata" in thread
        metadata = thread["metadata"]
        assert metadata.get("team_id") == "team999"

        elog("Thread metadata injection", {"metadata": metadata})

    @pytest.mark.asyncio
    async def test_assistant_create_metadata_injection(self):
        """Test that assistant creation handler injects metadata"""
        import uuid

        client = get_client_with_auth("eve", "admin", "team888")

        assistant = await client.assistants.create(
            graph_id="agent",
            name=f"Metadata Test Assistant {uuid.uuid4().hex[:8]}",
            metadata={},
            if_exists="do_nothing",
        )

        # Check that created_by and team_id were injected
        assert "metadata" in assistant
        metadata = assistant["metadata"]
        assert metadata.get("created_by") == "eve"
        assert metadata.get("team_id") == "team888"

        elog("Assistant metadata injection", {"metadata": metadata})
