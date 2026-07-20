"""E2E tests verifying thread user isolation when authentication is enabled.

Claim under test:
  When authentication is enabled, threads are automatically scoped to the
  authenticated user. Users can only see and interact with their own threads.

Require an auth-enabled server. See README.md for setup.
Run with: make e2e-auth  or  pytest -m auth_only
"""

import uuid

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog


def get_server_url() -> str:
    return settings.app.SERVER_URL


def get_auth_headers(user_id: str, role: str = "user", team_id: str = "team1") -> dict[str, str]:
    token = f"mock-jwt-{user_id}-{role}-{team_id}"
    return {"Authorization": f"Bearer {token}"}


def get_client_with_auth(user_id: str, role: str = "user", team_id: str = "team1"):
    from langgraph_sdk import get_client

    token = f"mock-jwt-{user_id}-{role}-{team_id}"
    return get_client(url=get_server_url(), headers={"Authorization": f"Bearer {token}"})


@pytest.mark.e2e
@pytest.mark.auth_only
class TestThreadOwnership:
    """Threads are created and stored under the authenticated user's identity."""

    @pytest.mark.asyncio
    async def test_created_thread_is_owned_by_creator(self) -> None:
        """GET /threads/<id> returns the thread to its creator."""
        client = get_client_with_auth("alice")
        thread = await client.threads.create(metadata={"isolation_test": "ownership"})
        thread_id = thread["thread_id"]
        elog("Created thread", thread)

        fetched = await client.threads.get(thread_id)
        assert fetched["thread_id"] == thread_id, "Creator should be able to fetch their own thread"

    @pytest.mark.asyncio
    async def test_other_user_cannot_get_thread(self) -> None:
        """GET /threads/<id> returns 404 when a different user requests it."""
        alice_client = get_client_with_auth("alice")
        thread = await alice_client.threads.create(metadata={"isolation_test": "cross-user-get"})
        thread_id = thread["thread_id"]
        elog("Alice created thread", {"thread_id": thread_id})

        bob_headers = get_auth_headers("bob")
        async with httpx.AsyncClient(base_url=get_server_url(), headers=bob_headers, timeout=30.0) as http:
            resp = await http.get(f"/threads/{thread_id}")

        elog("Bob GET Alice's thread", {"status": resp.status_code})
        assert resp.status_code == 404, (
            f"Expected 404 when bob requests alice's thread, got {resp.status_code}: {resp.text}"
        )


@pytest.mark.e2e
@pytest.mark.auth_only
class TestThreadSearch:
    """Thread search/list only returns threads belonging to the requesting user."""

    @pytest.mark.asyncio
    async def test_search_returns_only_own_threads(self) -> None:
        """POST /threads/search returns threads for the requesting user only."""
        tag = f"isolation-search-{uuid.uuid4().hex[:8]}"

        alice_client = get_client_with_auth("alice")
        bob_client = get_client_with_auth("bob")

        alice_thread = await alice_client.threads.create(metadata={"isolation_tag": tag})
        bob_thread = await bob_client.threads.create(metadata={"isolation_tag": tag})
        elog("Seeded threads", {"alice": alice_thread["thread_id"], "bob": bob_thread["thread_id"]})

        alice_headers = get_auth_headers("alice")
        async with httpx.AsyncClient(base_url=get_server_url(), headers=alice_headers, timeout=30.0) as http:
            resp = await http.post(
                "/threads/search",
                json={"metadata": {"isolation_tag": tag}, "limit": 100},
            )
        assert resp.status_code == 200, resp.text
        thread_ids = {t["thread_id"] for t in resp.json()}
        elog("Alice search results", sorted(thread_ids))

        assert alice_thread["thread_id"] in thread_ids, "Alice should see her own thread"
        assert bob_thread["thread_id"] not in thread_ids, "Alice must not see Bob's thread"

    @pytest.mark.asyncio
    async def test_list_endpoint_returns_only_own_threads(self) -> None:
        """GET /threads returns only threads owned by the requesting user."""
        tag = f"isolation-list-{uuid.uuid4().hex[:8]}"

        alice_client = get_client_with_auth("alice")
        bob_client = get_client_with_auth("bob")

        alice_thread = await alice_client.threads.create(metadata={"isolation_tag": tag})
        bob_thread = await bob_client.threads.create(metadata={"isolation_tag": tag})

        alice_headers = get_auth_headers("alice")
        async with httpx.AsyncClient(base_url=get_server_url(), headers=alice_headers, timeout=30.0) as http:
            resp = await http.get("/threads", params={"limit": 1000})
        assert resp.status_code == 200, resp.text

        data = resp.json()
        thread_ids = {t["thread_id"] for t in (data if isinstance(data, list) else data.get("threads", []))}
        elog("Alice list results", {"count": len(thread_ids)})

        assert alice_thread["thread_id"] in thread_ids, "Alice should see her own thread"
        assert bob_thread["thread_id"] not in thread_ids, "Alice must not see Bob's thread"


@pytest.mark.e2e
@pytest.mark.auth_only
class TestThreadMutationIsolation:
    """Users cannot mutate threads that belong to another user."""

    @pytest.mark.asyncio
    async def test_other_user_cannot_update_thread(self) -> None:
        """PATCH /threads/<id> returns 404 when a different user attempts to update."""
        alice_client = get_client_with_auth("alice")
        thread = await alice_client.threads.create(metadata={"isolation_test": "update"})
        thread_id = thread["thread_id"]

        bob_headers = get_auth_headers("bob")
        async with httpx.AsyncClient(base_url=get_server_url(), headers=bob_headers, timeout=30.0) as http:
            resp = await http.patch(
                f"/threads/{thread_id}",
                json={"metadata": {"hijacked": True}},
            )
        elog("Bob PATCH Alice's thread", {"status": resp.status_code})
        assert resp.status_code == 404, (
            f"Expected 404 when bob patches alice's thread, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_other_user_cannot_delete_thread(self) -> None:
        """DELETE /threads/<id> returns 404 when a different user attempts to delete."""
        alice_client = get_client_with_auth("alice")
        thread = await alice_client.threads.create(metadata={"isolation_test": "delete"})
        thread_id = thread["thread_id"]

        bob_headers = get_auth_headers("bob")
        async with httpx.AsyncClient(base_url=get_server_url(), headers=bob_headers, timeout=30.0) as http:
            resp = await http.delete(f"/threads/{thread_id}")
        elog("Bob DELETE Alice's thread", {"status": resp.status_code})
        assert resp.status_code == 404, (
            f"Expected 404 when bob deletes alice's thread, got {resp.status_code}: {resp.text}"
        )

        # Thread still accessible by Alice after Bob's failed delete attempt
        fetched = await alice_client.threads.get(thread_id)
        assert fetched["thread_id"] == thread_id, "Thread must still exist after unauthorized delete attempt"

    @pytest.mark.asyncio
    async def test_other_user_cannot_add_run_to_thread(self) -> None:
        """POST /threads/<id>/runs returns 404 when a different user attempts to create a run."""
        alice_client = get_client_with_auth("alice")
        thread = await alice_client.threads.create(metadata={"isolation_test": "run"})
        thread_id = thread["thread_id"]

        bob_headers = get_auth_headers("bob")
        async with httpx.AsyncClient(base_url=get_server_url(), headers=bob_headers, timeout=30.0) as http:
            resp = await http.post(
                f"/threads/{thread_id}/runs",
                json={"assistant_id": "agent", "input": {"messages": [{"role": "human", "content": "hi"}]}},
            )
        elog("Bob POST run to Alice's thread", {"status": resp.status_code})
        assert resp.status_code == 404, (
            f"Expected 404 when bob tries to run against alice's thread, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_other_user_cannot_stream_run_on_thread(self) -> None:
        """POST /threads/<id>/runs/stream returns 404 when a different user attempts to stream a run."""
        alice_client = get_client_with_auth("alice")
        thread = await alice_client.threads.create(metadata={"isolation_test": "stream-run"})
        thread_id = thread["thread_id"]

        bob_headers = get_auth_headers("bob")
        async with httpx.AsyncClient(base_url=get_server_url(), headers=bob_headers, timeout=30.0) as http:
            resp = await http.post(
                f"/threads/{thread_id}/runs/stream",
                json={"assistant_id": "agent", "input": {"messages": [{"role": "human", "content": "hi"}]}},
            )
        elog("Bob POST stream run to Alice's thread", {"status": resp.status_code})
        assert resp.status_code == 404, (
            f"Expected 404 when bob streams against alice's thread, got {resp.status_code}: {resp.text}"
        )

    @pytest.mark.asyncio
    async def test_other_user_cannot_wait_run_on_thread(self) -> None:
        """POST /threads/<id>/runs/wait returns 404 when a different user attempts to create a run."""
        alice_client = get_client_with_auth("alice")
        thread = await alice_client.threads.create(metadata={"isolation_test": "wait-run"})
        thread_id = thread["thread_id"]

        bob_headers = get_auth_headers("bob")
        async with httpx.AsyncClient(base_url=get_server_url(), headers=bob_headers, timeout=30.0) as http:
            resp = await http.post(
                f"/threads/{thread_id}/runs/wait",
                json={"assistant_id": "agent", "input": {"messages": [{"role": "human", "content": "hi"}]}},
            )
        elog("Bob POST wait run to Alice's thread", {"status": resp.status_code})
        assert resp.status_code == 404, (
            f"Expected 404 when bob waits against alice's thread, got {resp.status_code}: {resp.text}"
        )


@pytest.mark.e2e
@pytest.mark.auth_only
class TestUnauthenticatedAccess:
    """Requests without a valid token are rejected entirely."""

    def test_get_thread_without_auth_returns_401(self) -> None:
        """GET /threads/<id> without Authorization header returns 401."""
        fake_thread_id = str(uuid.uuid4())
        resp = httpx.get(f"{get_server_url()}/threads/{fake_thread_id}", timeout=10.0)
        elog("Unauthenticated GET thread", {"status": resp.status_code})
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"

    def test_search_without_auth_returns_401(self) -> None:
        """POST /threads/search without Authorization header returns 401."""
        resp = httpx.post(
            f"{get_server_url()}/threads/search",
            json={"limit": 10},
            timeout=10.0,
        )
        elog("Unauthenticated search", {"status": resp.status_code})
        assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
