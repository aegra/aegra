"""Integration tests for assistants CRUD operations"""

import pytest

from aegra_api.services.assistant_service import get_assistant_service
from tests.fixtures.clients import create_test_app, make_client
from tests.fixtures.test_helpers import make_assistant


@pytest.fixture
def client(mock_assistant_service):
    """Create test client with a fully mocked service.

    The service is overridden wholesale, so auth dispatch (which lives inside
    the real service) never runs here — these tests cover the API layer only.
    """
    app = create_test_app(include_runs=False, include_threads=False)

    from aegra_api.api import assistants as assistants_module

    app.include_router(assistants_module.router)

    app.dependency_overrides[get_assistant_service] = lambda: mock_assistant_service
    yield make_client(app)


class TestCreateAssistant:
    """Test POST /assistants"""

    def test_create_assistant_basic(self, client, mock_assistant_service):
        """Test creating a basic assistant"""
        assistant = make_assistant()
        mock_assistant_service.create_assistant.return_value = assistant

        resp = client.post(
            "/assistants",
            json={
                "name": "Test Assistant",
                "graph_id": "test-graph",
                "metadata": {},
                "config": {},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_id"] == "test-assistant-123"
        assert data["name"] == "Test Assistant"
        assert data["graph_id"] == "test-graph"
        mock_assistant_service.create_assistant.assert_called_once()

    def test_create_assistant_with_metadata(self, client, mock_assistant_service):
        """Test creating assistant with rich metadata"""
        assistant = make_assistant(metadata={"description": "A helpful assistant", "tags": ["prod", "v1"]})
        mock_assistant_service.create_assistant.return_value = assistant

        resp = client.post(
            "/assistants",
            json={
                "name": "Test Assistant",
                "graph_id": "test-graph",
                "metadata": {
                    "description": "A helpful assistant",
                    "tags": ["prod", "v1"],
                },
                "config": {},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        # Metadata may be aliased as metadata_dict in the response
        metadata = data.get("metadata", data.get("metadata_dict", {}))
        assert metadata.get("description") == "A helpful assistant" or metadata.get("tags") == [
            "prod",
            "v1",
        ]

    def test_create_assistant_with_config(self, client, mock_assistant_service):
        """Test creating assistant with custom config"""
        assistant = make_assistant()
        assistant.config = {"temperature": 0.7, "max_tokens": 1000}
        mock_assistant_service.create_assistant.return_value = assistant

        resp = client.post(
            "/assistants",
            json={
                "name": "Test Assistant",
                "graph_id": "test-graph",
                "metadata": {},
                "config": {"temperature": 0.7, "max_tokens": 1000},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["temperature"] == 0.7
        assert data["config"]["max_tokens"] == 1000


class TestListAssistants:
    """Test GET /assistants"""

    def test_list_assistants_with_results(self, client, mock_assistant_service):
        """Test listing assistants when user has some"""
        assistants = [
            make_assistant("asst-1", "Assistant 1", "graph-1"),
            make_assistant("asst-2", "Assistant 2", "graph-2"),
            make_assistant("asst-3", "Assistant 3", "graph-1"),
        ]
        mock_assistant_service.list_assistants.return_value = assistants

        resp = client.get("/assistants")

        assert resp.status_code == 200
        data = resp.json()
        assert "assistants" in data
        assert "total" in data
        assert data["total"] == 3
        assert len(data["assistants"]) == 3
        assert data["assistants"][0]["assistant_id"] == "asst-1"

    def test_list_assistants_empty(self, client, mock_assistant_service):
        """Test listing assistants when user has none"""
        mock_assistant_service.list_assistants.return_value = []

        resp = client.get("/assistants")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["assistants"] == []


class TestGetAssistant:
    """Test GET /assistants/{assistant_id}"""

    def test_get_assistant_success(self, client, mock_assistant_service):
        """Test getting an existing assistant"""
        assistant = make_assistant(
            "asst-123",
            "My Assistant",
            "my-graph",
            metadata={"purpose": "testing"},
        )
        mock_assistant_service.get_assistant.return_value = assistant

        resp = client.get("/assistants/asst-123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_id"] == "asst-123"
        assert data["name"] == "My Assistant"
        assert data["graph_id"] == "my-graph"

    def test_get_assistant_not_found(self, client, mock_assistant_service):
        """Test getting a non-existent assistant"""
        from fastapi import HTTPException

        mock_assistant_service.get_assistant.side_effect = HTTPException(
            status_code=404, detail="Assistant 'nonexistent' not found"
        )

        resp = client.get("/assistants/nonexistent")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]


class TestUpdateAssistant:
    """Test PATCH /assistants/{assistant_id}"""

    def test_update_assistant_name(self, client, mock_assistant_service):
        """Test updating assistant name"""
        updated = make_assistant(name="Updated Name")
        mock_assistant_service.update_assistant.return_value = updated

        resp = client.patch(
            "/assistants/test-assistant-123",
            json={"name": "Updated Name"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Updated Name"

    def test_update_assistant_metadata(self, client, mock_assistant_service):
        """Test updating assistant metadata"""
        updated = make_assistant(metadata={"env": "production", "version": "2.0"})
        mock_assistant_service.update_assistant.return_value = updated

        resp = client.patch(
            "/assistants/test-assistant-123",
            json={"metadata": {"env": "production", "version": "2.0"}},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["assistant_id"] == "test-assistant-123"

    def test_update_assistant_config(self, client, mock_assistant_service):
        """Test updating assistant config"""
        updated = make_assistant()
        updated.config = {"model": "gpt-4", "temperature": 0.5}
        mock_assistant_service.update_assistant.return_value = updated

        resp = client.patch(
            "/assistants/test-assistant-123",
            json={"config": {"model": "gpt-4", "temperature": 0.5}},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["config"]["model"] == "gpt-4"

    def test_update_assistant_not_found(self, client, mock_assistant_service):
        """Test updating non-existent assistant"""
        from fastapi import HTTPException

        mock_assistant_service.update_assistant.side_effect = HTTPException(
            status_code=404, detail="Assistant not found"
        )

        resp = client.patch(
            "/assistants/nonexistent",
            json={"name": "New Name"},
        )

        assert resp.status_code == 404


class TestDeleteAssistant:
    """Test DELETE /assistants/{assistant_id}"""

    def test_delete_assistant_success(self, client, mock_assistant_service):
        """Test deleting an existing assistant"""
        mock_assistant_service.delete_assistant.return_value = {"status": "deleted"}

        resp = client.delete("/assistants/test-assistant-123")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        mock_assistant_service.delete_assistant.assert_called_once_with("test-assistant-123")

    def test_delete_assistant_not_found(self, client, mock_assistant_service):
        """Test deleting non-existent assistant"""
        from fastapi import HTTPException

        mock_assistant_service.delete_assistant.side_effect = HTTPException(
            status_code=404, detail="Assistant not found"
        )

        resp = client.delete("/assistants/nonexistent")

        assert resp.status_code == 404


class TestSearchAssistants:
    """Test POST /assistants/search"""

    def test_search_assistants_no_filters(self, client, mock_assistant_service):
        """Test searching without filters"""
        assistants = [
            make_assistant("asst-1", "Assistant 1"),
            make_assistant("asst-2", "Assistant 2"),
        ]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post("/assistants/search", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_search_assistants_zero_results(self, client, mock_assistant_service):
        """Test searching when no assistants match"""
        mock_assistant_service.search_assistants.return_value = []

        resp = client.post(
            "/assistants/search",
            json={"graph_id": "nonexistent-graph"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_search_assistants_single_result(self, client, mock_assistant_service):
        """Test searching with exactly one result"""
        assistant = make_assistant("asst-1", "Single Assistant", "unique-graph")
        mock_assistant_service.search_assistants.return_value = [assistant]

        resp = client.post(
            "/assistants/search",
            json={"name": "Single Assistant"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["assistant_id"] == "asst-1"
        assert data[0]["name"] == "Single Assistant"

    def test_search_assistants_multiple_results(self, client, mock_assistant_service):
        """Test searching with multiple results"""
        assistants = [
            make_assistant("asst-1", "Assistant 1", "graph-1"),
            make_assistant("asst-2", "Assistant 2", "graph-1"),
            make_assistant("asst-3", "Assistant 3", "graph-1"),
        ]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post(
            "/assistants/search",
            json={"graph_id": "graph-1"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 3
        for i, assistant in enumerate(data, 1):
            assert assistant["assistant_id"] == f"asst-{i}"
            assert assistant["graph_id"] == "graph-1"

    def test_search_assistants_by_graph_id(self, client, mock_assistant_service):
        """Test searching by graph_id"""
        assistants = [
            make_assistant("asst-1", "Assistant 1", "my-graph"),
        ]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post(
            "/assistants/search",
            json={"graph_id": "my-graph"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["graph_id"] == "my-graph"

    def test_search_assistants_by_name(self, client, mock_assistant_service):
        """Test searching by name"""
        assistants = [
            make_assistant("asst-1", "Test Assistant"),
        ]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post(
            "/assistants/search",
            json={"name": "Test Assistant"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Test Assistant"

    def test_search_assistants_by_description(self, client, mock_assistant_service):
        """Test searching by description"""
        assistants = [
            make_assistant("asst-1", description="A helpful assistant"),
        ]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post(
            "/assistants/search",
            json={"description": "helpful"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_search_assistants_by_metadata(self, client, mock_assistant_service):
        """Test searching by metadata"""
        assistants = [
            make_assistant("asst-1", metadata={"env": "prod"}),
        ]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post(
            "/assistants/search",
            json={"metadata": {"env": "prod"}},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_search_assistants_by_multiple_metadata_fields(self, client, mock_assistant_service):
        """Test searching by multiple metadata fields"""
        assistants = [
            make_assistant("asst-1", metadata={"env": "prod", "region": "us-east"}),
        ]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post(
            "/assistants/search",
            json={"metadata": {"env": "prod", "region": "us-east"}},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1

    def test_search_assistants_with_pagination(self, client, mock_assistant_service):
        """Test searching with offset and limit"""
        assistants = [make_assistant(f"asst-{i}") for i in range(2)]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post(
            "/assistants/search",
            json={"offset": 0, "limit": 10},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_search_assistants_with_offset(self, client, mock_assistant_service):
        """Test searching with offset to skip results"""
        assistants = [
            make_assistant("asst-2", "Assistant 2"),
            make_assistant("asst-3", "Assistant 3"),
        ]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post(
            "/assistants/search",
            json={"offset": 2, "limit": 10},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 2

    def test_search_assistants_with_small_limit(self, client, mock_assistant_service):
        """Test searching with small page size"""
        assistants = [
            make_assistant("asst-1", "Assistant 1"),
            make_assistant("asst-2", "Assistant 2"),
        ]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post(
            "/assistants/search",
            json={"limit": 2},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_search_assistants_combined_filters(self, client, mock_assistant_service):
        """Test searching with multiple filter criteria"""
        assistants = [
            make_assistant("asst-1", "Prod Assistant", "prod-graph", metadata={"env": "prod"}),
        ]
        mock_assistant_service.search_assistants.return_value = assistants

        resp = client.post(
            "/assistants/search",
            json={
                "name": "Prod Assistant",
                "graph_id": "prod-graph",
                "metadata": {"env": "prod"},
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "Prod Assistant"
        assert data[0]["graph_id"] == "prod-graph"


class TestSearchAssistantsSortAndAuth:
    """Sort params + #333 regression: auth handlers returning filters must not 500."""

    def test_search_with_sort_by_name_asc(self, client, mock_assistant_service):
        mock_assistant_service.search_assistants.return_value = []

        resp = client.post(
            "/assistants/search",
            json={"sort_by": "name", "sort_order": "asc"},
        )

        assert resp.status_code == 200
        kwargs = mock_assistant_service.search_assistants.call_args.kwargs
        assert kwargs.get("sort_asc") is True
        sort_column = kwargs.get("sort_column")
        assert getattr(sort_column, "key", None) == "name"

    def test_search_with_sort_by_only_defaults_to_desc(self, client, mock_assistant_service):
        mock_assistant_service.search_assistants.return_value = []

        resp = client.post("/assistants/search", json={"sort_by": "updated_at"})

        assert resp.status_code == 200
        kwargs = mock_assistant_service.search_assistants.call_args.kwargs
        assert kwargs.get("sort_asc") is False

    def test_invalid_sort_by_returns_422(self, client, mock_assistant_service):
        resp = client.post("/assistants/search", json={"sort_by": "; DROP TABLE"})
        assert resp.status_code == 422

    def test_invalid_sort_order_returns_422(self, client, mock_assistant_service):
        resp = client.post("/assistants/search", json={"sort_by": "name", "sort_order": "sideways"})
        assert resp.status_code == 422

    # Auth-handler filter merging (#333 regression) and read-endpoint dispatch
    # now live in the service layer; see tests/unit/test_services/
    # test_assistant_service.py::TestAuthDispatch.


class TestCountAssistants:
    """Test POST /assistants/count"""

    def test_count_assistants_no_filters(self, client, mock_assistant_service):
        """Test counting all assistants"""
        mock_assistant_service.count_assistants.return_value = 42

        resp = client.post("/assistants/count", json={})

        assert resp.status_code == 200
        assert resp.json() == 42

    def test_count_assistants_zero(self, client, mock_assistant_service):
        """Test count returning zero"""
        mock_assistant_service.count_assistants.return_value = 0

        resp = client.post("/assistants/count", json={})

        assert resp.status_code == 200
        assert resp.json() == 0

    def test_count_assistants_single(self, client, mock_assistant_service):
        """Test count returning one"""
        mock_assistant_service.count_assistants.return_value = 1

        resp = client.post(
            "/assistants/count",
            json={"name": "Unique Assistant"},
        )

        assert resp.status_code == 200
        assert resp.json() == 1

    def test_count_assistants_multiple(self, client, mock_assistant_service):
        """Test count returning multiple"""
        mock_assistant_service.count_assistants.return_value = 15

        resp = client.post(
            "/assistants/count",
            json={"graph_id": "popular-graph"},
        )

        assert resp.status_code == 200
        assert resp.json() == 15

    def test_count_assistants_by_graph_id(self, client, mock_assistant_service):
        """Test counting assistants by graph_id"""
        mock_assistant_service.count_assistants.return_value = 5

        resp = client.post(
            "/assistants/count",
            json={"graph_id": "my-graph"},
        )

        assert resp.status_code == 200
        assert resp.json() == 5

    def test_count_assistants_by_name(self, client, mock_assistant_service):
        """Test counting assistants by name"""
        mock_assistant_service.count_assistants.return_value = 2

        resp = client.post(
            "/assistants/count",
            json={"name": "Test Assistant"},
        )

        assert resp.status_code == 200
        assert resp.json() == 2

    def test_count_assistants_by_metadata(self, client, mock_assistant_service):
        """Test counting assistants by metadata"""
        mock_assistant_service.count_assistants.return_value = 8

        resp = client.post(
            "/assistants/count",
            json={"metadata": {"env": "prod"}},
        )

        assert resp.status_code == 200
        assert resp.json() == 8

    def test_count_assistants_by_multiple_filters(self, client, mock_assistant_service):
        """Test counting with multiple filter criteria"""
        mock_assistant_service.count_assistants.return_value = 3

        resp = client.post(
            "/assistants/count",
            json={
                "graph_id": "prod-graph",
                "metadata": {"env": "prod", "region": "us-east"},
            },
        )

        assert resp.status_code == 200
        assert resp.json() == 3


class TestSetLatestAssistant:
    """Test POST /assistants/{assistant_id}/latest"""

    def test_set_latest_assistant(self, client, mock_assistant_service):
        """Test setting latest version of assistant"""
        latest = make_assistant(name="Latest Version")
        latest.version = 5
        mock_assistant_service.set_assistant_latest.return_value = latest

        resp = client.post("/assistants/test-assistant-123/latest", json={"version": 5})

        assert resp.status_code == 200
        data = resp.json()
        assert data["version"] == 5
        assert data["name"] == "Latest Version"


class TestListAssistantVersions:
    """Test POST /assistants/{assistant_id}/versions"""

    def test_list_assistant_versions(self, client, mock_assistant_service):
        """Test listing all versions of assistant"""
        versions = [
            make_assistant(name="V1"),
            make_assistant(name="V2"),
            make_assistant(name="V3"),
        ]
        versions[0].version = 1
        versions[1].version = 2
        versions[2].version = 3
        mock_assistant_service.list_assistant_versions.return_value = versions

        resp = client.post("/assistants/test-assistant-123/versions")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["version"] == 1
        assert data[2]["version"] == 3

    def test_list_assistant_versions_empty(self, client, mock_assistant_service):
        """Test listing versions when there are none"""
        mock_assistant_service.list_assistant_versions.return_value = []

        resp = client.post("/assistants/test-assistant-123/versions")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 0


class TestGetAssistantSchemas:
    """Test GET /assistants/{assistant_id}/schemas"""

    def test_get_assistant_schemas(self, client, mock_assistant_service):
        """Test getting assistant schemas"""
        schemas = {
            "input_schema": {"type": "object", "properties": {}},
            "output_schema": {"type": "object", "properties": {}},
            "config_schema": {"type": "object", "properties": {}},
            "state_schema": {"type": "object", "properties": {}},
        }
        mock_assistant_service.get_assistant_schemas.return_value = schemas

        resp = client.get("/assistants/test-assistant-123/schemas")

        assert resp.status_code == 200
        data = resp.json()
        assert "input_schema" in data
        assert "output_schema" in data
        assert "config_schema" in data
        assert "state_schema" in data


class TestGetAssistantGraph:
    """Test GET /assistants/{assistant_id}/graph"""

    def test_get_assistant_graph(self, client, mock_assistant_service):
        """Test getting assistant graph definition"""
        graph = {
            "nodes": ["start", "agent", "tools", "end"],
            "edges": [
                {"from": "start", "to": "agent"},
                {"from": "agent", "to": "tools"},
                {"from": "tools", "to": "end"},
            ],
        }
        mock_assistant_service.get_assistant_graph.return_value = graph

        resp = client.get("/assistants/test-assistant-123/graph")

        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 4

    def test_get_assistant_graph_xray_format(self, client, mock_assistant_service):
        """Test getting graph in xray format"""
        graph = {"graph": {"nodes": [], "edges": []}}
        mock_assistant_service.get_assistant_graph.return_value = graph

        resp = client.get("/assistants/test-assistant-123/graph?xray=true")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)


class TestGetAssistantSubgraphs:
    """Test GET /assistants/{assistant_id}/subgraphs"""

    def test_get_assistant_subgraphs(self, client, mock_assistant_service):
        """Test getting assistant subgraphs"""
        subgraphs = {
            "subgraph_1": {
                "nodes": ["node_a", "node_b"],
                "edges": [{"from": "node_a", "to": "node_b"}],
            },
            "subgraph_2": {
                "nodes": ["node_c", "node_d"],
                "edges": [{"from": "node_c", "to": "node_d"}],
            },
        }
        mock_assistant_service.get_assistant_subgraphs.return_value = subgraphs

        resp = client.get("/assistants/test-assistant-123/subgraphs")

        assert resp.status_code == 200
        data = resp.json()
        assert "subgraph_1" in data
        assert "subgraph_2" in data
        assert len(data["subgraph_1"]["nodes"]) == 2

    def test_get_assistant_subgraphs_recurse(self, client, mock_assistant_service):
        """Test getting subgraphs with recursion"""
        subgraphs = {"nested": {"nodes": [], "edges": []}}
        mock_assistant_service.get_assistant_subgraphs.return_value = subgraphs

        resp = client.get("/assistants/test-assistant-123/subgraphs?recurse=true")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_get_assistant_subgraphs_namespace(self, client, mock_assistant_service):
        """Test getting subgraphs for specific namespace"""
        subgraphs = {"ns1": {"nodes": [], "edges": []}}
        mock_assistant_service.get_assistant_subgraphs.return_value = subgraphs

        resp = client.get("/assistants/test-assistant-123/subgraphs?namespace=ns1")

        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
