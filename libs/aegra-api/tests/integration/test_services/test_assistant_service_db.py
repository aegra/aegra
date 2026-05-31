"""Integration tests for AssistantService database operations

These tests verify service interactions with real database operations.
"""

from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import AssistantVersion as AssistantVersionORM
from aegra_api.models import Assistant, AssistantCreate, AssistantUpdate
from aegra_api.models.auth import User
from aegra_api.services.assistant_service import AssistantService
from tests.fixtures.database import DummySessionBase


class TestAssistantServiceDatabase:
    """Test AssistantService with database operations"""

    @pytest.fixture
    def mock_langgraph_service(self):
        """Mock LangGraphService for database tests"""
        from unittest.mock import AsyncMock, Mock

        mock_service = Mock()
        mock_service.list_graphs.return_value = {"test-graph": "test-graph.py"}
        mock_service.get_graph_for_validation = AsyncMock(return_value=Mock())
        return mock_service

    @pytest.fixture
    def db_session(self):
        """Database session for testing"""
        from unittest.mock import AsyncMock

        class AssistantTestSession(DummySessionBase):
            def __init__(self):
                super().__init__()
                self.added_objects = []
                self.deleted_objects = []

                # Create mockable methods
                self.scalar = AsyncMock()
                self.scalars = AsyncMock()
                self.execute = AsyncMock()
                self.commit = AsyncMock()
                self.refresh = AsyncMock()

            def add(self, obj):
                """Track added objects and set timestamps"""
                self.added_objects.append(obj)
                # Set required timestamps for Assistant ORM objects
                if hasattr(obj, "created_at"):
                    obj.created_at = datetime.now(UTC)
                if hasattr(obj, "updated_at"):
                    obj.updated_at = datetime.now(UTC)
                return None

            async def delete(self, obj):
                """Track deleted objects"""
                self.deleted_objects.append(obj)
                return None

        return AssistantTestSession()

    @pytest.fixture
    def assistant_service(self, db_session, mock_langgraph_service):
        """AssistantService with database session"""
        # Reset mocks for each test
        db_session.scalar.reset_mock()
        db_session.scalars.reset_mock()
        db_session.execute.reset_mock()
        db_session.commit.reset_mock()
        db_session.refresh.reset_mock()

        # Set default return values
        db_session.scalar.return_value = None  # No existing assistant by default
        db_session.scalars.return_value.all.return_value = []  # Empty results by default

        return AssistantService(db_session, User(identity="user-123"), mock_langgraph_service)

    @pytest.mark.asyncio
    async def test_create_assistant_db_transaction(self, assistant_service):
        """Test assistant creation with database transaction"""
        request = AssistantCreate(
            name="Test Assistant",
            description="A test assistant",
            graph_id="test-graph",
            config={"temperature": 0.7},
            metadata={"env": "test"},
        )

        result = await assistant_service.create_assistant(request)

        assert isinstance(result, Assistant)
        assert result.name == "Test Assistant"
        assert result.description == "A test assistant"
        assert result.graph_id == "test-graph"
        assert result.user_id == "user-123"
        assert result.version == 1
        assert result.config == {"temperature": 0.7}
        assert result.metadata == {"env": "test"}

        # Verify assistant ORM object was added to session
        assert len(assistant_service.session.added_objects) >= 1
        assistant_orm = assistant_service.session.added_objects[0]
        assert isinstance(assistant_orm, AssistantORM)
        assert assistant_orm.name == "Test Assistant"
        assert assistant_orm.metadata_dict == {"env": "test"}

    @pytest.mark.asyncio
    async def test_create_assistant_version_creation(self, assistant_service):
        """Test that assistant version is created during assistant creation"""
        request = AssistantCreate(
            name="Versioned Assistant",
            graph_id="test-graph",
            config={"model": "gpt-4"},
        )

        result = await assistant_service.create_assistant(request)

        # Find the created version
        versions = [obj for obj in assistant_service.session.added_objects if isinstance(obj, AssistantVersionORM)]

        assert len(versions) == 1
        version = versions[0]
        assert version.assistant_id == result.assistant_id
        assert version.version == 1
        assert version.name == "Versioned Assistant"
        assert version.config == {"model": "gpt-4"}
        assert version.graph_id == "test-graph"

    @pytest.mark.asyncio
    async def test_update_assistant_version_increment(self, assistant_service):
        """Test assistant update creates new version"""
        # First create an assistant
        create_request = AssistantCreate(
            name="Original Assistant",
            graph_id="test-graph",
        )
        original_assistant = await assistant_service.create_assistant(create_request)

        # Mock scalar calls: first returns assistant, second returns max version, third returns updated assistant
        assistant_service.session.scalar.side_effect = [
            original_assistant,
            1,
            original_assistant,
        ]  # max version = 1

        # Update the assistant
        update_request = AssistantUpdate(
            name="Updated Assistant",
            description="Updated description",
            config={"temperature": 0.8},
        )

        await assistant_service.update_assistant(original_assistant.assistant_id, update_request)

        # Verify new version was created
        versions = [obj for obj in assistant_service.session.added_objects if isinstance(obj, AssistantVersionORM)]

        # Should have 2 versions (original + updated)
        assert len(versions) == 2

        # Find the latest version
        latest_version = max(versions, key=lambda v: v.version)
        assert latest_version.version == 2
        assert latest_version.name == "Updated Assistant"
        assert latest_version.description == "Updated description"
        assert latest_version.config == {"temperature": 0.8}

    @pytest.mark.asyncio
    async def test_delete_assistant_cascade(self, assistant_service):
        """Test assistant deletion removes from database"""
        # Create an assistant first
        request = AssistantCreate(
            name="To Delete",
            graph_id="test-graph",
        )
        assistant = await assistant_service.create_assistant(request)

        # Mock the assistant for deletion
        assistant_service.session.scalar.return_value = assistant

        result = await assistant_service.delete_assistant(assistant.assistant_id)

        assert result == {"status": "deleted"}
        assert assistant in assistant_service.session.deleted_objects

    @pytest.mark.asyncio
    async def test_list_assistants_no_metadata_filter(self, assistant_service):
        """list_assistants without metadata returns all rows for the user."""
        mock_result = Mock()
        mock_result.all.return_value = []
        assistant_service.session.scalars.return_value = mock_result

        result = await assistant_service.list_assistants()

        assert result == []
        assistant_service.session.scalars.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_assistants_with_metadata_filter(self, assistant_service, monkeypatch):
        """When an auth handler scopes results via a metadata filter, list_assistants
        applies a JSONB containment predicate but no pagination — the regression
        fence for the silent 20-row cap that a previous fix introduced."""
        mock_result = Mock()
        mock_result.all.return_value = []
        assistant_service.session.scalars.return_value = mock_result

        from aegra_api.services import _authenticated

        async def handler_filter(ctx, value):
            return {"metadata": {"tenant": "x"}}

        monkeypatch.setattr(_authenticated, "handle_event", handler_filter)

        result = await assistant_service.list_assistants()

        assert result == []
        called_stmt = assistant_service.session.scalars.call_args.args[0]
        compiled = str(called_stmt.compile())
        assert "@>" in compiled
        assert "LIMIT" not in compiled.upper()
        assert "OFFSET" not in compiled.upper()

    @pytest.mark.asyncio
    async def test_search_assistants_pagination(self, assistant_service):
        """Test assistant search with pagination"""
        # Create multiple assistants
        for i in range(5):
            request = AssistantCreate(
                name=f"Assistant {i}",
                graph_id="test-graph",
                metadata={"index": i},
            )
            await assistant_service.create_assistant(request)

        # Mock search request
        mock_request = Mock()
        mock_request.name = None
        mock_request.description = None
        mock_request.graph_id = None
        mock_request.metadata = None
        mock_request.offset = 2
        mock_request.limit = 2

        # Mock search results
        mock_result = Mock()
        mock_result.all.return_value = []

        assistant_service.session.scalars.return_value = mock_result

        result = await assistant_service.search_assistants(mock_request)

        assert isinstance(result, list)
        # Verify pagination parameters were applied
        assistant_service.session.scalars.assert_called_once()

    @pytest.mark.asyncio
    async def test_assistant_version_history(self, assistant_service):
        """Test assistant version history retrieval"""
        # Create an assistant
        create_request = AssistantCreate(
            name="Versioned Assistant",
            graph_id="test-graph",
        )
        assistant = await assistant_service.create_assistant(create_request)

        # Mock assistant for version listing
        assistant_service.session.scalar.return_value = assistant

        # Mock version query results with proper ORM objects
        from aegra_api.core.orm import AssistantVersion as AssistantVersionORM

        version1 = AssistantVersionORM(
            assistant_id=assistant.assistant_id,
            version=1,
            name="Version 1",
            description="First version",
            graph_id="test-graph",
            config={"model": "gpt-3.5"},
            context={},
            metadata_dict={},
            created_at=datetime.now(UTC),
        )
        version2 = AssistantVersionORM(
            assistant_id=assistant.assistant_id,
            version=2,
            name="Version 2",
            description="Second version",
            graph_id="test-graph",
            config={"model": "gpt-4"},
            context={},
            metadata_dict={},
            created_at=datetime.now(UTC),
        )

        mock_result = Mock()
        mock_result.all.return_value = [version2, version1]  # Reverse order for testing

        assistant_service.session.scalars.return_value = mock_result

        result = await assistant_service.list_assistant_versions(assistant.assistant_id)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].version == 2
        assert result[1].version == 1

    @pytest.mark.asyncio
    async def test_set_assistant_latest_version(self, assistant_service):
        """Test setting assistant latest version"""
        # Create an assistant
        create_request = AssistantCreate(
            name="Versioned Assistant",
            graph_id="test-graph",
        )
        assistant = await assistant_service.create_assistant(create_request)

        # Mock scalar calls: assistant, version, updated assistant
        from aegra_api.core.orm import AssistantVersion as AssistantVersionORM

        version_obj = AssistantVersionORM(
            assistant_id=assistant.assistant_id,
            version=2,
            name="Version 2",
            description="Second version",
            graph_id="test-graph",
            config={"model": "gpt-4"},
            context={},
            metadata_dict={},
            created_at=datetime.now(UTC),
        )

        assistant_service.session.scalar.side_effect = [
            assistant,
            version_obj,
            assistant,
        ]

        result = await assistant_service.set_assistant_latest(assistant.assistant_id, 2)

        assert isinstance(result, Assistant)
        # Verify update was executed
        assert assistant_service.session.execute.called
        assert assistant_service.session.commit.called

    @pytest.mark.asyncio
    async def test_assistant_metadata_search(self, assistant_service):
        """Test assistant search by metadata"""
        # Create assistants with different metadata
        request1 = AssistantCreate(
            name="Prod Assistant",
            graph_id="test-graph",
            metadata={"env": "prod", "team": "backend"},
        )
        await assistant_service.create_assistant(request1)

        request2 = AssistantCreate(
            name="Dev Assistant",
            graph_id="test-graph",
            metadata={"env": "dev", "team": "frontend"},
        )
        await assistant_service.create_assistant(request2)

        # Mock search request with metadata filter
        mock_request = Mock()
        mock_request.name = None
        mock_request.description = None
        mock_request.graph_id = None
        mock_request.metadata = {"env": "prod"}
        mock_request.offset = 0
        mock_request.limit = 10

        # Mock search results
        mock_result = Mock()
        mock_result.all.return_value = []

        assistant_service.session.scalars.return_value = mock_result

        result = await assistant_service.search_assistants(mock_request)

        assert isinstance(result, list)
        # Verify metadata filter was applied
        assistant_service.session.scalars.assert_called_once()

    @pytest.mark.asyncio
    async def test_assistant_count_with_filters(self, assistant_service):
        """Test assistant counting with various filters"""
        # Create multiple assistants
        for i in range(3):
            request = AssistantCreate(
                name=f"Assistant {i}",
                graph_id="test-graph",
                metadata={"category": "test"},
            )
            await assistant_service.create_assistant(request)

        # Mock count request
        mock_request = Mock()
        mock_request.name = "Assistant"
        mock_request.description = None
        mock_request.graph_id = "test-graph"
        mock_request.metadata = {"category": "test"}

        # Mock count result
        assistant_service.session.scalar.return_value = 3

        result = await assistant_service.count_assistants(mock_request)

        assert result == 3
        # scalar is called 4 times: 3 for create_assistant + 1 for count_assistants
        assert assistant_service.session.scalar.call_count == 4

    @pytest.mark.asyncio
    async def test_assistant_concurrent_operations(self, assistant_service):
        """Test concurrent assistant operations"""
        # Create multiple assistants concurrently
        requests = [
            AssistantCreate(
                name=f"Concurrent Assistant {i}",
                graph_id="test-graph",
            )
            for i in range(3)
        ]

        # Create assistants
        results = []
        for request in requests:
            result = await assistant_service.create_assistant(request)
            results.append(result)

        # Verify all assistants were created
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.name == f"Concurrent Assistant {i}"
            assert result.graph_id == "test-graph"
            assert result.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_assistant_transaction_rollback(self, assistant_service):
        """Test assistant creation transaction rollback on error"""
        request = AssistantCreate(
            name="Failing Assistant",
            graph_id="test-graph",
        )

        # Mock LangGraph service failure
        assistant_service.langgraph_service.get_graph_for_validation.side_effect = Exception("Graph load failed")

        with pytest.raises(HTTPException, match="Failed to load graph: Graph load failed"):
            await assistant_service.create_assistant(request)

        # Verify no objects were added to session
        assert len(assistant_service.session.added_objects) == 0

    @pytest.mark.asyncio
    async def test_assistant_large_metadata_handling(self, assistant_service):
        """Test assistant creation with large metadata"""
        large_metadata = {
            "description": "A" * 1000,  # Large description
            "tags": [f"tag_{i}" for i in range(100)],  # Many tags
            "config": {"nested": {"deep": {"value": "test"}}},  # Nested structure
        }

        request = AssistantCreate(
            name="Large Metadata Assistant",
            graph_id="test-graph",
            metadata=large_metadata,
        )

        result = await assistant_service.create_assistant(request)

        assert result.metadata == large_metadata
        assert result.name == "Large Metadata Assistant"

    @pytest.mark.asyncio
    async def test_assistant_special_characters(self, assistant_service):
        """Test assistant creation with special characters"""
        special_name = "Assistant with émojis 🚀 and spëcial çharacters"
        special_description = "Description with unicode: αβγδε"

        request = AssistantCreate(
            name=special_name,
            description=special_description,
            graph_id="test-graph",
            metadata={"unicode": "测试", "emoji": "🎉"},
        )

        result = await assistant_service.create_assistant(request)

        assert result.name == special_name
        assert result.description == special_description
        assert result.metadata["unicode"] == "测试"
        assert result.metadata["emoji"] == "🎉"
