"""E2E tests verifying store search and list_namespaces never cross user scopes.

Store items live under ("users", <identity>, ...). Postgres stores scoped reads with a
LIKE on the dot-joined namespace, so langgraph-checkpoint-postgres < 3.1.1 matched
"users.alice" against "users.alice2..." and treated % and _ in an identity as
wildcards (GHSA-47pj-3jcm-6whg).

Requires a running Aegra server with auth enabled (see README.md). Run:
    pytest tests/e2e/manual_auth_tests/test_store_user_isolation_e2e.py -v -m auth_only
"""

import uuid

import httpx
import pytest

from aegra_api.settings import settings
from tests.e2e._utils import elog


def get_server_url() -> str:
    url = settings.app.SERVER_URL
    assert url is not None, "SERVER_URL is derived from HOST/PORT and is never None"
    return url


def auth_headers(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer mock-jwt-{user_id}-user-team1"}


async def put_item(http: httpx.AsyncClient, user_id: str, namespace: list[str], key: str) -> None:
    resp = await http.put(
        "/store/items",
        headers=auth_headers(user_id),
        json={"namespace": namespace, "key": key, "value": {"owner": user_id}},
    )
    assert resp.status_code == 204, f"{user_id} put failed: {resp.status_code} {resp.text}"


@pytest.mark.e2e
@pytest.mark.auth_only
class TestStorePrefixSiblingIsolation:
    """A user never reads items of another user whose identity starts with theirs."""

    @pytest.mark.asyncio
    async def test_search_does_not_return_sibling_users_items(self) -> None:
        user = f"u{uuid.uuid4().hex[:8]}"
        sibling = f"{user}2"
        key = f"note-{uuid.uuid4().hex[:8]}"

        async with httpx.AsyncClient(base_url=get_server_url(), timeout=30.0) as http:
            await put_item(http, sibling, ["notes"], key)
            resp = await http.post(
                "/store/items/search",
                headers=auth_headers(user),
                json={"namespace_prefix": [], "limit": 100},
            )

        assert resp.status_code == 200, f"search failed: {resp.status_code} {resp.text}"
        owners = {item["value"].get("owner") for item in resp.json()["items"]}
        elog("Search as user with sibling data present", {"user": user, "owners": sorted(o or "" for o in owners)})
        assert sibling not in owners, f"{user} must not see items written by {sibling}"

    @pytest.mark.asyncio
    async def test_list_namespaces_does_not_return_sibling_users_namespaces(self) -> None:
        user = f"u{uuid.uuid4().hex[:8]}"
        sibling = f"{user}2"

        async with httpx.AsyncClient(base_url=get_server_url(), timeout=30.0) as http:
            await put_item(http, sibling, ["private"], "k")
            resp = await http.post(
                "/store/namespaces",
                headers=auth_headers(user),
                json={"prefix": [], "limit": 100},
            )

        assert resp.status_code == 200, f"list_namespaces failed: {resp.status_code} {resp.text}"
        namespaces = resp.json()["namespaces"]
        elog("List namespaces as user with sibling data present", {"user": user, "namespaces": namespaces})
        assert all(sibling not in ns for ns in namespaces), f"{user} must not see {sibling}'s namespaces"


async def search_owners(http: httpx.AsyncClient, user_id: str) -> set[str]:
    resp = await http.post(
        "/store/items/search",
        headers=auth_headers(user_id),
        json={"namespace_prefix": [], "limit": 1000},
    )
    assert resp.status_code == 200, f"search failed: {resp.status_code} {resp.text}"
    return {item["value"].get("owner") for item in resp.json()["items"]}


@pytest.mark.e2e
@pytest.mark.auth_only
class TestStoreWildcardIdentityIsolation:
    """LIKE metacharacters in an identity match literally, not as wildcards."""

    @pytest.mark.asyncio
    async def test_underscore_identity_does_not_read_matching_user(self) -> None:
        victim = f"v{uuid.uuid4().hex[:8]}"
        attacker = f"{victim[:3]}_{victim[4:]}"

        async with httpx.AsyncClient(base_url=get_server_url(), timeout=30.0) as http:
            await put_item(http, victim, ["secrets"], "k")
            owners = await search_owners(http, attacker)

        elog("Search as underscore identity", {"attacker": attacker, "victim": victim, "owners": sorted(owners)})
        assert victim not in owners, f"{attacker} must not see items written by {victim}"

    @pytest.mark.asyncio
    async def test_percent_identity_reads_only_its_own_items(self) -> None:
        victim = f"v{uuid.uuid4().hex[:8]}"

        async with httpx.AsyncClient(base_url=get_server_url(), timeout=30.0) as http:
            await put_item(http, victim, ["secrets"], "k")
            owners = await search_owners(http, "%")

        elog("Search as percent identity", {"victim": victim, "owners": sorted(owners)})
        assert owners <= {"%"}, f"identity '%' must only see its own items, got owners {sorted(owners)}"
