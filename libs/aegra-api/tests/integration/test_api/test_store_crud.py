"""Integration tests for store CRUD operations"""

import pytest

from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.test_helpers import DummyStoreItem


@pytest.fixture
def client(mock_store):
    """Create test client with mocked store"""
    app = create_test_app(include_runs=False, include_threads=False)

    # Import and mount store router
    from aegra_api.api import store as store_module

    app.include_router(store_module.router)

    # Mock db_manager.get_store()
    def mock_get_store():
        return mock_store

    # Patch at module level
    import aegra_api.core.database as db_module

    db_module.db_manager.get_store = mock_get_store

    return make_client(app)


class TestPutStoreItem:
    """Test PUT /store/items"""

    def test_put_item_success(self, client, mock_store):
        """Test storing an item"""
        resp = client.put(
            "/store/items",
            json={
                "namespace": ["test", "namespace"],
                "key": "test-key",
                "value": {"data": "test-value"},
            },
        )

        assert resp.status_code == 204
        assert resp.content == b""
        mock_store.aput.assert_called_once()

    def test_put_item_with_empty_namespace(self, client, mock_store):
        """Test storing item with empty namespace uses default"""
        resp = client.put(
            "/store/items",
            json={
                "namespace": [],
                "key": "test-key",
                "value": {"data": "test"},
            },
        )

        assert resp.status_code == 204
        # Should use default user namespace
        mock_store.aput.assert_called_once()
        call_args = mock_store.aput.call_args
        namespace = call_args.kwargs["namespace"]
        # Default namespace should be ["users", "test-user"]
        assert "users" in namespace
        assert "test-user" in namespace

    def test_put_item_complex_value(self, client, mock_store):
        """Test storing item with complex nested value"""
        resp = client.put(
            "/store/items",
            json={
                "namespace": ["data"],
                "key": "complex-key",
                "value": {
                    "nested": {"deep": {"value": 123}},
                    "array": [1, 2, 3],
                    "string": "test",
                },
            },
        )

        assert resp.status_code == 204

    def test_put_item_rejects_array_value(self, client, mock_store):
        """Test that array values are rejected"""
        resp = client.put(
            "/store/items",
            json={
                "namespace": ["test"],
                "key": "array-key",
                "value": [1, 2, 3],
            },
        )

        assert resp.status_code == 422
        error_detail = resp.json()["detail"]
        assert any(
            "dictionary" in str(err).lower() or "object" in str(err).lower()
            for err in error_detail
            if isinstance(err, dict)
        )

    def test_put_item_rejects_scalar_value(self, client, mock_store):
        """Test that scalar values (string, number, boolean) are rejected"""
        # Test string
        resp = client.put(
            "/store/items",
            json={
                "namespace": ["test"],
                "key": "string-key",
                "value": "not-a-dict",
            },
        )
        assert resp.status_code == 422

        # Test number
        resp = client.put(
            "/store/items",
            json={
                "namespace": ["test"],
                "key": "number-key",
                "value": 42,
            },
        )
        assert resp.status_code == 422

        # Test boolean
        resp = client.put(
            "/store/items",
            json={
                "namespace": ["test"],
                "key": "bool-key",
                "value": True,
            },
        )
        assert resp.status_code == 422

    def test_put_item_rejects_null_value(self, client, mock_store):
        """Test that null values are rejected"""
        resp = client.put(
            "/store/items",
            json={
                "namespace": ["test"],
                "key": "null-key",
                "value": None,
            },
        )
        assert resp.status_code == 422


class TestGetStoreItem:
    """Test GET /store/items"""

    def test_get_item_success(self, client, mock_store):
        """Test getting an existing item"""
        # Mock store returns an item
        mock_item = DummyStoreItem(
            key="test-key",
            value={"data": "test-value"},
            namespace=("test", "namespace"),
        )
        mock_store.aget.return_value = mock_item

        resp = client.get("/store/items?key=test-key&namespace=test.namespace")

        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "test-key"
        assert data["value"] == {"data": "test-value"}
        assert "namespace" in data

    def test_get_item_not_found(self, client, mock_store):
        """Test getting a non-existent item"""
        mock_store.aget.return_value = None

        resp = client.get("/store/items?key=nonexistent")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_get_item_with_list_namespace(self, client, mock_store):
        """Test getting item with namespace as list"""
        mock_item = DummyStoreItem(
            key="test-key",
            value={"data": "test"},
            namespace=("a", "b", "c"),
        )
        mock_store.aget.return_value = mock_item

        # FastAPI Query with list is tricky, use dotted notation
        resp = client.get("/store/items?key=test-key&namespace=a.b.c")

        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "test-key"

    def test_get_item_no_namespace(self, client, mock_store):
        """Test getting item without namespace"""
        mock_item = DummyStoreItem(
            key="test-key",
            value={"data": "test"},
            namespace=("users", "test-user"),
        )
        mock_store.aget.return_value = mock_item

        resp = client.get("/store/items?key=test-key")

        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "test-key"

    def test_get_item_with_empty_namespace_query_param(self, client, mock_store):
        """Test empty namespace query param is treated as no namespace"""
        mock_item = DummyStoreItem(
            key="test-key",
            value={"data": "test"},
            namespace=("users", "test-user"),
        )
        mock_store.aget.return_value = mock_item

        resp = client.get("/store/items?namespace=&key=test-key")

        assert resp.status_code == 200
        call_args = mock_store.aget.call_args
        namespace = call_args[0][0]
        assert namespace == ("users", "test-user")


class TestDeleteStoreItem:
    """Test DELETE /store/items"""

    def test_delete_item_with_body(self, client, mock_store):
        """Test deleting item via request body"""
        resp = client.request(
            "DELETE",
            "/store/items",
            json={"namespace": ["test", "ns"], "key": "test-key"},
        )

        assert resp.status_code == 204
        assert resp.content == b""
        mock_store.adelete.assert_called_once()

    def test_delete_item_with_query_params(self, client, mock_store):
        """Test deleting item via query parameters"""
        resp = client.delete("/store/items?key=test-key")

        assert resp.status_code == 204
        assert resp.content == b""
        mock_store.adelete.assert_called_once()

    def test_delete_item_missing_key(self, client, mock_store):
        """Test deleting without providing key"""
        resp = client.delete("/store/items")

        assert resp.status_code == 422
        assert "key" in resp.json()["detail"].lower()

    def test_delete_item_with_namespace(self, client, mock_store):
        """Test deleting item with specific namespace"""
        resp = client.request(
            "DELETE",
            "/store/items",
            json={"namespace": ["custom", "namespace"], "key": "test-key"},
        )

        assert resp.status_code == 204
        assert resp.content == b""


class TestSearchStoreItems:
    """Test POST /store/items/search"""

    def test_search_items_success(self, client, mock_store):
        """Test searching for items"""
        # Mock search results
        mock_results = [
            DummyStoreItem("key1", {"data": "value1"}, ("test", "ns")),
            DummyStoreItem("key2", {"data": "value2"}, ("test", "ns")),
        ]
        mock_store.asearch.return_value = mock_results
        filter_payload = {"type": "note", "status": "active"}

        resp = client.post(
            "/store/items/search",
            json={
                "namespace_prefix": ["test"],
                "query": None,
                "filter": filter_payload,
                "limit": 10,
                "offset": 0,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert len(data["items"]) == 2
        assert data["total"] == 2
        assert data["limit"] == 10
        assert data["offset"] == 0
        mock_store.asearch.assert_called_once()
        call_args = mock_store.asearch.call_args
        assert call_args.kwargs["filter"] == filter_payload

    def test_search_items_empty(self, client, mock_store):
        """Test searching with no results"""
        mock_store.asearch.return_value = []

        resp = client.post(
            "/store/items/search",
            json={"namespace_prefix": ["empty"], "query": None},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["items"] == []
        assert data["total"] == 0
        mock_store.asearch.assert_called_once()
        call_args = mock_store.asearch.call_args
        assert call_args.kwargs["filter"] is None

    def test_search_items_with_query(self, client, mock_store):
        """Test searching with a query string"""
        mock_results = [
            DummyStoreItem("matching-key", {"data": "value"}, ("test",)),
        ]
        mock_store.asearch.return_value = mock_results

        resp = client.post(
            "/store/items/search",
            json={
                "namespace_prefix": ["test"],
                "query": "matching",
                "limit": 20,
                "offset": 0,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["key"] == "matching-key"
        mock_store.asearch.assert_called_once()
        call_args = mock_store.asearch.call_args
        assert call_args.kwargs["filter"] is None

    def test_search_items_with_filter_only(self, client, mock_store):
        """Test searching with filter only"""
        mock_results = [
            DummyStoreItem("filtered-key", {"data": "value"}, ("test",)),
        ]
        mock_store.asearch.return_value = mock_results
        filter_payload = {"tag": "important"}

        resp = client.post(
            "/store/items/search",
            json={"namespace_prefix": ["test"], "filter": filter_payload},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["key"] == "filtered-key"
        mock_store.asearch.assert_called_once()
        call_args = mock_store.asearch.call_args
        assert call_args.kwargs["filter"] == filter_payload

    def test_search_items_with_pagination(self, client, mock_store):
        """Test searching with pagination"""
        mock_results = [DummyStoreItem(f"key{i}", {"data": f"val{i}"}, ("test",)) for i in range(5)]
        mock_store.asearch.return_value = mock_results

        resp = client.post(
            "/store/items/search",
            json={"namespace_prefix": [], "query": None, "limit": 5, "offset": 10},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["limit"] == 5
        assert data["offset"] == 10
        mock_store.asearch.assert_called_once()
        call_args = mock_store.asearch.call_args
        assert call_args.kwargs["filter"] is None

    def test_search_items_default_limit(self, client, mock_store):
        """Test search uses default limit when not provided"""
        mock_store.asearch.return_value = []

        resp = client.post(
            "/store/items/search",
            json={"namespace_prefix": ["test"], "query": None},
        )

        assert resp.status_code == 200
        data = resp.json()
        # Default limit should be 20
        assert data["limit"] == 20
        assert data["offset"] == 0
        mock_store.asearch.assert_called_once()
        call_args = mock_store.asearch.call_args
        assert call_args.kwargs["filter"] is None

    def test_search_items_with_auth_handler_filter_merge(self, client, mock_store):
        """Test that auth handler filters are merged with request filters"""
        from unittest.mock import AsyncMock, patch

        mock_store.asearch.return_value = []

        # Mock handle_event to return auth filters
        async def mock_handle_event(ctx, value):
            return {"auth_field": "auth_value"}

        with patch(
            "aegra_api.api.store.handle_event",
            new=AsyncMock(side_effect=mock_handle_event),
        ):
            request_filter = {"user_field": "user_value"}
            resp = client.post(
                "/store/items/search",
                json={
                    "namespace_prefix": ["test"],
                    "query": None,
                    "filter": request_filter,
                },
            )

        assert resp.status_code == 200
        mock_store.asearch.assert_called_once()
        call_args = mock_store.asearch.call_args
        # Both auth handler filter and request filter should be present
        merged_filter = call_args.kwargs["filter"]
        assert merged_filter is not None
        assert "user_field" in merged_filter
        assert merged_filter["user_field"] == "user_value"
        assert "auth_field" in merged_filter
        assert merged_filter["auth_field"] == "auth_value"


class TestListNamespaces:
    """Test POST /store/namespaces"""

    def test_list_namespaces_success(self, client, mock_store) -> None:
        """Test listing namespaces returns expected results"""
        mock_store.alist_namespaces.return_value = [
            ("users", "test-user", "notes"),
            ("users", "test-user", "settings"),
        ]

        resp = client.post(
            "/store/namespaces",
            json={"prefix": ["users", "test-user"]},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["namespaces"]) == 2
        assert ["users", "test-user", "notes"] in data["namespaces"]
        assert ["users", "test-user", "settings"] in data["namespaces"]
        mock_store.alist_namespaces.assert_called_once()

    def test_list_namespaces_empty_result(self, client, mock_store) -> None:
        """Test listing namespaces with no results"""
        mock_store.alist_namespaces.return_value = []

        resp = client.post(
            "/store/namespaces",
            json={},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["namespaces"] == []

    def test_list_namespaces_with_suffix(self, client, mock_store) -> None:
        """Test listing namespaces with suffix filter"""
        mock_store.alist_namespaces.return_value = [
            ("users", "test-user", "notes"),
        ]

        resp = client.post(
            "/store/namespaces",
            json={"prefix": ["users"], "suffix": ["notes"]},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["namespaces"]) == 1

        call_args = mock_store.alist_namespaces.call_args
        assert call_args.kwargs["suffix"] == ("notes",)

    def test_list_namespaces_with_max_depth(self, client, mock_store) -> None:
        """Test listing namespaces with max_depth"""
        mock_store.alist_namespaces.return_value = [
            ("users",),
            ("data",),
        ]

        resp = client.post(
            "/store/namespaces",
            json={"max_depth": 1},
        )

        assert resp.status_code == 200
        call_args = mock_store.alist_namespaces.call_args
        assert call_args.kwargs["max_depth"] == 1

    def test_list_namespaces_with_pagination(self, client, mock_store) -> None:
        """Test listing namespaces with limit and offset"""
        mock_store.alist_namespaces.return_value = [
            ("users", "test-user", "batch"),
        ]

        resp = client.post(
            "/store/namespaces",
            json={"limit": 5, "offset": 10},
        )

        assert resp.status_code == 200
        call_args = mock_store.alist_namespaces.call_args
        assert call_args.kwargs["limit"] == 5
        assert call_args.kwargs["offset"] == 10

    def test_list_namespaces_applies_user_scoping(self, client, mock_store) -> None:
        """Test that user namespace scoping is applied to prefix"""
        mock_store.alist_namespaces.return_value = []

        resp = client.post(
            "/store/namespaces",
            json={},
        )

        assert resp.status_code == 200
        # With no prefix, should default to user's namespace
        call_args = mock_store.alist_namespaces.call_args
        prefix = call_args.kwargs["prefix"]
        assert "users" in prefix
        assert "test-user" in prefix

    def test_list_namespaces_max_depth_validation(self, client, mock_store) -> None:
        """Test that max_depth validation rejects out-of-range values"""
        resp = client.post(
            "/store/namespaces",
            json={"max_depth": 0},
        )
        assert resp.status_code == 422

        resp = client.post(
            "/store/namespaces",
            json={"max_depth": 101},
        )
        assert resp.status_code == 422

    def test_list_namespaces_with_auth_handler_filters(self, client, mock_store) -> None:
        """Test that auth handler filters override prefix and suffix"""
        from unittest.mock import AsyncMock, patch

        mock_store.alist_namespaces.return_value = []

        async def mock_handle_event(ctx, value):
            return {"prefix": ["override-prefix"], "suffix": ["override-suffix"]}

        with patch(
            "aegra_api.api.store.handle_event",
            new=AsyncMock(side_effect=mock_handle_event),
        ):
            resp = client.post(
                "/store/namespaces",
                json={"prefix": ["original"]},
            )

        assert resp.status_code == 200
        call_args = mock_store.alist_namespaces.call_args
        assert call_args.kwargs["suffix"] == ("override-suffix",)


class TestNamespaceScoping:
    """Test user namespace scoping"""

    def test_put_item_applies_user_scoping(self, client, mock_store):
        """Test that user scoping is applied on put"""
        resp = client.put(
            "/store/items",
            json={
                "namespace": [],  # Empty namespace
                "key": "test-key",
                "value": {"data": "test"},
            },
        )

        assert resp.status_code == 204
        # Verify the namespace was scoped to the user
        call_args = mock_store.aput.call_args
        namespace = call_args.kwargs["namespace"]
        assert "users" in namespace or "test-user" in namespace

    def test_get_item_applies_user_scoping(self, client, mock_store):
        """Test that user scoping is applied on get"""
        mock_item = DummyStoreItem("key", {"data": "val"}, ("users", "test-user"))
        mock_store.aget.return_value = mock_item

        resp = client.get("/store/items?key=test-key")

        assert resp.status_code == 200
        # Verify the namespace was scoped
        call_args = mock_store.aget.call_args
        namespace = call_args[0][0]  # First positional arg
        assert "users" in namespace or "test-user" in namespace

    def test_put_cross_user_namespace_is_remapped(self, client, mock_store) -> None:
        """Test that a cross-user namespace is scoped under the caller's prefix"""
        resp = client.put(
            "/store/items",
            json={
                "namespace": ["users", "other-user", "secrets"],
                "key": "stolen",
                "value": {"data": "nope"},
            },
        )

        assert resp.status_code == 204
        call_args = mock_store.aput.call_args
        namespace = call_args.kwargs["namespace"]
        assert namespace == ("users", "test-user", "users", "other-user", "secrets")

    def test_get_cross_user_namespace_is_remapped(self, client, mock_store) -> None:
        """Test that reading from another user's namespace is remapped"""
        mock_store.aget.return_value = None

        resp = client.get("/store/items?key=secret&namespace=users&namespace=other-user&namespace=data")

        assert resp.status_code == 404
        call_args = mock_store.aget.call_args
        namespace = call_args[0][0]
        assert namespace == ("users", "test-user", "users", "other-user", "data")

    def test_delete_cross_user_namespace_is_remapped(self, client, mock_store) -> None:
        """Test that deleting from another user's namespace is remapped"""
        resp = client.request(
            "DELETE",
            "/store/items",
            json={
                "namespace": ["users", "other-user"],
                "key": "victim-key",
            },
        )

        assert resp.status_code == 204
        call_args = mock_store.adelete.call_args
        namespace = call_args[0][0]
        assert namespace == ("users", "test-user", "users", "other-user")

    def test_search_cross_user_namespace_is_remapped(self, client, mock_store) -> None:
        """Test that searching another user's namespace is remapped"""
        mock_store.asearch.return_value = []

        resp = client.post(
            "/store/items/search",
            json={
                "namespace_prefix": ["users", "other-user", "docs"],
            },
        )

        assert resp.status_code == 200
        call_args = mock_store.asearch.call_args
        namespace_prefix = call_args[0][0]
        assert namespace_prefix == ("users", "test-user", "users", "other-user", "docs")


class TestStoreIntegration:
    """Test complete store workflows"""

    def test_put_get_delete_workflow(self, client, mock_store):
        """Test complete lifecycle: put -> get -> delete"""
        # 1. Put an item
        put_resp = client.put(
            "/store/items",
            json={
                "namespace": ["workflow"],
                "key": "lifecycle-key",
                "value": {"stage": "initial"},
            },
        )
        assert put_resp.status_code == 204

        # 2. Mock getting it back
        mock_item = DummyStoreItem(
            "lifecycle-key",
            {"stage": "initial"},
            ("workflow",),
        )
        mock_store.aget.return_value = mock_item

        get_resp = client.get("/store/items?key=lifecycle-key&namespace=workflow")
        assert get_resp.status_code == 200
        assert get_resp.json()["value"]["stage"] == "initial"

        # 3. Delete it
        delete_resp = client.request(
            "DELETE",
            "/store/items",
            json={"namespace": ["workflow"], "key": "lifecycle-key"},
        )
        assert delete_resp.status_code == 204

    def test_search_after_multiple_puts(self, client, mock_store):
        """Test searching after storing multiple items"""
        # Store multiple items
        for i in range(3):
            resp = client.put(
                "/store/items",
                json={
                    "namespace": ["batch"],
                    "key": f"item-{i}",
                    "value": {"index": i},
                },
            )
            assert resp.status_code == 204

        # Mock search results
        mock_results = [DummyStoreItem(f"item-{i}", {"index": i}, ("batch",)) for i in range(3)]
        mock_store.asearch.return_value = mock_results

        # Search for them
        search_resp = client.post(
            "/store/items/search",
            json={"namespace_prefix": ["batch"], "query": None},
        )
        assert search_resp.status_code == 200
        assert len(search_resp.json()["items"]) == 3
