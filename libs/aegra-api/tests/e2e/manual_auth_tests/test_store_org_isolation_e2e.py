"""E2E tests verifying store org scoping when authentication is enabled.

Claim under test:
  A leading "orgs" element in a store namespace scopes items to the caller's
  organization. Members of the same org share those items; members of a
  different org (or no org) never reach them.

⚠️ MANUAL TESTS - These are skipped by default. Run with: pytest -m manual_auth

Requires a running Aegra server with auth enabled (the jwt_mock_auth_example
config) and store.scopes {"orgs": "org_id"} configured (see aegra.auth.json).
See README.md for setup. The mock auth maps a token's team segment to org_id:
mock-jwt-<user>-<role>-<org>. A token without a team segment has no org.

Run:
    pytest tests/e2e/manual_auth_tests/test_store_org_isolation_e2e.py -v -m manual_auth
"""

import uuid

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog


def get_server_url() -> str:
    """Return the base URL of the running server under test."""
    return settings.app.SERVER_URL


def auth_headers(user_id: str, *, role: str = "user", org: str | None = "org1") -> dict[str, str]:
    """Build mock-JWT auth headers. org=None omits the team segment, so the user has no org."""
    token = f"mock-jwt-{user_id}-{role}" if org is None else f"mock-jwt-{user_id}-{role}-{org}"
    return {"Authorization": f"Bearer {token}"}


@pytest.mark.e2e
@pytest.mark.manual_auth
class TestOrgStoreSharing:
    """Items written under the "orgs" prefix are shared across members of that org."""

    @pytest.mark.asyncio
    async def test_org_member_reads_item_written_by_another_member(self) -> None:
        """Bob reads a shared item that Alice wrote, because both are in org1."""
        key = f"greeting-{uuid.uuid4().hex[:8]}"
        namespace = ["orgs", "org1", "shared-prompts"]
        value = {"text": "hello org"}

        async with httpx.AsyncClient(base_url=get_server_url(), timeout=30.0) as http:
            put = await http.put(
                "/store/items",
                headers=auth_headers("alice", org="org1"),
                json={"namespace": namespace, "key": key, "value": value},
            )
            assert put.status_code == 204, f"Alice put failed: {put.status_code} {put.text}"

            got = await http.get(
                "/store/items",
                headers=auth_headers("bob", org="org1"),
                params=[("namespace", "orgs"), ("namespace", "org1"), ("namespace", "shared-prompts"), ("key", key)],
            )
        elog("Bob GET org1 shared item", {"status": got.status_code, "body": got.text})
        assert got.status_code == 200, f"Bob (same org) should read the item: {got.status_code} {got.text}"
        assert got.json()["value"] == value

    @pytest.mark.asyncio
    async def test_org_member_search_sees_shared_item(self) -> None:
        """Search under the org prefix returns items written by any member of the org."""
        key = f"doc-{uuid.uuid4().hex[:8]}"
        namespace = ["orgs", "org1", "docs"]

        async with httpx.AsyncClient(base_url=get_server_url(), timeout=30.0) as http:
            put = await http.put(
                "/store/items",
                headers=auth_headers("alice", org="org1"),
                json={"namespace": namespace, "key": key, "value": {"body": "shared"}},
            )
            assert put.status_code == 204, put.text

            search = await http.post(
                "/store/items/search",
                headers=auth_headers("bob", org="org1"),
                json={"namespace_prefix": ["orgs", "org1", "docs"], "limit": 100},
            )
        assert search.status_code == 200, search.text
        keys = {item["key"] for item in search.json()["items"]}
        elog("Bob search org1 docs", sorted(keys))
        assert key in keys, "Bob (same org) should see the item Alice wrote"


@pytest.mark.e2e
@pytest.mark.manual_auth
class TestOrgStoreIsolation:
    """An org's items are unreachable from a different org or from no org at all."""

    @pytest.mark.asyncio
    async def test_other_org_cannot_read_item(self) -> None:
        """Carol in org2 requesting org1's namespace is buried under org2, so she gets 404."""
        key = f"secret-{uuid.uuid4().hex[:8]}"
        namespace = ["orgs", "org1", "shared-prompts"]

        async with httpx.AsyncClient(base_url=get_server_url(), timeout=30.0) as http:
            put = await http.put(
                "/store/items",
                headers=auth_headers("alice", org="org1"),
                json={"namespace": namespace, "key": key, "value": {"text": "org1 only"}},
            )
            assert put.status_code == 204, put.text

            got = await http.get(
                "/store/items",
                headers=auth_headers("carol", org="org2"),
                params=[("namespace", "orgs"), ("namespace", "org1"), ("namespace", "shared-prompts"), ("key", key)],
            )
        elog("Carol (org2) GET org1 item", {"status": got.status_code})
        assert got.status_code == 404, f"Expected 404 cross-org, got {got.status_code}: {got.text}"

    @pytest.mark.asyncio
    async def test_other_org_search_does_not_see_item(self) -> None:
        """Search from org2 under the org1 prefix is scoped to org2 and finds nothing."""
        key = f"secret-{uuid.uuid4().hex[:8]}"

        async with httpx.AsyncClient(base_url=get_server_url(), timeout=30.0) as http:
            put = await http.put(
                "/store/items",
                headers=auth_headers("alice", org="org1"),
                json={"namespace": ["orgs", "org1", "docs"], "key": key, "value": {"body": "org1 only"}},
            )
            assert put.status_code == 204, put.text

            search = await http.post(
                "/store/items/search",
                headers=auth_headers("carol", org="org2"),
                json={"namespace_prefix": ["orgs", "org1", "docs"], "limit": 100},
            )
        assert search.status_code == 200, search.text
        keys = {item["key"] for item in search.json()["items"]}
        elog("Carol (org2) search org1 docs", sorted(keys))
        assert key not in keys, "Carol (org2) must not see org1's item"

    @pytest.mark.asyncio
    async def test_user_without_org_is_forbidden(self) -> None:
        """A user with no org using the "orgs" prefix is rejected with 403."""
        async with httpx.AsyncClient(base_url=get_server_url(), timeout=30.0) as http:
            resp = await http.put(
                "/store/items",
                headers=auth_headers("dave", org=None),
                json={"namespace": ["orgs", "shared"], "key": "x", "value": {"text": "nope"}},
            )
        elog("No-org user PUT orgs namespace", {"status": resp.status_code, "body": resp.text})
        assert resp.status_code == 403, f"Expected 403 for no-org user, got {resp.status_code}: {resp.text}"
