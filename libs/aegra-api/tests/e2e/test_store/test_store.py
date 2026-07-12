import pytest

from tests.e2e._utils import elog, get_e2e_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_store_endpoints_via_sdk():
    client = get_e2e_client()

    # Use a user-private namespace implicitly; server will scope to ["users", <identity>]
    # Insert item
    ns = ["notes"]
    key = "e2e-item-1"
    value = {"title": "Hello", "tags": ["e2e", "store"], "score": 42}

    await client.store.put_item(ns, key=key, value=value)
    elog("store.put_item", {"namespace": ns, "key": key, "value": value})

    # Get item (SDK sends dotted namespace on GET)
    got = await client.store.get_item(ns, key=key)
    elog("store.get_item", got)
    assert got["key"] == key
    assert got["value"] == value
    assert got.get("namespace") in (ns, ["users"]) or isinstance(got.get("namespace"), list)

    # Search by namespace prefix
    search = await client.store.search_items(["notes"], limit=10)
    elog("store.search_items", search)
    assert isinstance(search, dict)
    assert "items" in search
    assert any(item.get("key") == key for item in search["items"])

    # Delete item (SDK sends JSON body)
    await client.store.delete_item(ns, key=key)
    elog("store.delete_item", {"namespace": ns, "key": key})

    # Ensure deleted
    with pytest.raises(Exception):  # noqa: B017 - SDK doesn't expose specific exception type
        await client.store.get_item(ns, key=key)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_org_prefix_without_org_membership_is_forbidden():
    """The anonymous user has no org_id, so the "orgs" prefix is rejected with 403.

    Relies on aegra.json configuring store.scopes {"orgs": ["org_id"]}. The org-scoped
    happy path needs auth that sets org_id; it lives in the auth-enabled suite
    (manual_auth_tests/test_store_org_isolation_e2e.py).
    """
    client = get_e2e_client()

    with pytest.raises(Exception) as exc_info:  # noqa: B017 - SDK doesn't expose specific exception type
        await client.store.put_item(["orgs", "shared-prompts"], key="greeting", value={"text": "hi"})
    elog("store.put_item orgs prefix rejected", str(exc_info.value))
    assert "403" in str(exc_info.value) or "organization" in str(exc_info.value).lower()


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_store_rejects_non_dict_values():
    """Test that store API rejects non-dictionary values"""
    client = get_e2e_client()

    ns = ["test"]
    key = "invalid-value"

    # Test array value (should be rejected)
    with pytest.raises(Exception) as exc_info:  # noqa: B017
        await client.store.put_item(ns, key=f"{key}-array", value=[1, 2, 3])
    error_msg = str(exc_info.value).lower()
    assert "dictionary" in error_msg or "object" in error_msg or "422" in str(exc_info.value)

    # Test scalar values (should be rejected)
    for scalar_value in [42, "string", True, None]:
        with pytest.raises(Exception) as exc_info:  # noqa: B017
            await client.store.put_item(ns, key=f"{key}-{type(scalar_value).__name__}", value=scalar_value)
        error_msg = str(exc_info.value).lower()
        assert "dictionary" in error_msg or "object" in error_msg or "422" in str(exc_info.value)
