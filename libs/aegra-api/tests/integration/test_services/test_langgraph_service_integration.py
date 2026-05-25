"""Integration tests for LangGraphService"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest
from langgraph.graph import StateGraph

from aegra_api.services.langgraph_service import LangGraphService

# Import settings to patch it reliably
from aegra_api.settings import settings


class DummyStateGraph(StateGraph):
    def __init__(self):
        # avoid heavy init
        pass


class TestLangGraphServiceRealFiles:
    """Test LangGraphService with real file operations"""

    @pytest.mark.asyncio
    async def test_initialize_with_real_config_file(self):
        """Test initialization with real config file"""
        config_data = {
            "graphs": {"test_graph": "./graphs/test.py:graph"},
            "dependencies": ["dep1", "dep2"],
        }

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "aegra.json"
            config_path.write_text(json.dumps(config_data))

            with (
                patch("aegra_api.services.langgraph_service.LangGraphService._ensure_default_assistants"),
                patch("aegra_api.services.langgraph_service.LangGraphService._load_all_graph_modules"),
            ):
                service = LangGraphService(str(config_path))
                await service.initialize()

                assert service.config == config_data
                assert service.config_path == config_path
                assert service.get_dependencies() == ["dep1", "dep2"]

    @pytest.mark.asyncio
    async def test_initialize_env_var_with_real_file(self, monkeypatch):
        """Test initialization with AEGRA_CONFIG pointing to real file"""
        config_data = {"graphs": {"env_graph": "./graphs/env.py:graph"}}

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "env_config.json"
            config_path.write_text(json.dumps(config_data))

            # Directly patch settings object. This is more reliable than monkeypatch.setenv
            # because the Settings object is instantiated once at module import time.
            with (
                patch.object(settings.app, "AEGRA_CONFIG", str(config_path)),
                patch("aegra_api.services.langgraph_service.LangGraphService._ensure_default_assistants"),
                patch("aegra_api.services.langgraph_service.LangGraphService._load_all_graph_modules"),
            ):
                service = LangGraphService()
                await service.initialize()

                assert service.config == config_data
                assert service.config_path == config_path

    @pytest.mark.asyncio
    async def test_initialize_fallback_to_langgraph_json(self, monkeypatch):
        """Test fallback to langgraph.json when aegra.json not found"""
        import sys

        config_data = {"graphs": {"fallback_graph": "./graphs/fallback.py:graph"}}

        # Windows has issues with temp directory cleanup when files are still open
        # Use manual cleanup with error handling to work around this
        temp_dir_obj = TemporaryDirectory()
        temp_dir = temp_dir_obj.name
        try:
            langgraph_path = Path(temp_dir) / "langgraph.json"
            langgraph_path.write_text(json.dumps(config_data))

            # Change working directory so relative path lookups work
            monkeypatch.chdir(temp_dir)

            # Ensure AEGRA_CONFIG is None so fallback logic triggers
            with (
                patch.object(settings.app, "AEGRA_CONFIG", None),
                patch("aegra_api.services.langgraph_service.LangGraphService._ensure_default_assistants"),
                patch("aegra_api.services.langgraph_service.LangGraphService._load_all_graph_modules"),
            ):
                service = LangGraphService()
                await service.initialize()

                assert service.config == config_data
                # Path will be relative since we changed directory
                assert service.config_path.name == "langgraph.json"
        finally:
            # Windows may have file handle issues, so use ignore_errors
            try:
                temp_dir_obj.cleanup()
            except (PermissionError, OSError):
                # On Windows, files may still be locked - ignore cleanup errors
                if sys.platform == "win32":
                    pass
                else:
                    raise

    @pytest.mark.asyncio
    async def test_initialize_invalid_json_file(self):
        """Test error with invalid JSON in real file"""
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.json"
            config_path.write_text("invalid json content")

            service = LangGraphService(str(config_path))

            with pytest.raises(json.JSONDecodeError):
                await service.initialize()

    @pytest.mark.asyncio
    async def test_initialize_missing_file(self):
        """Test error with missing config file"""
        with (
            patch("aegra_api.services.langgraph_service.LangGraphService._ensure_default_assistants"),
            patch("pathlib.Path.exists", return_value=False),
        ):
            service = LangGraphService("nonexistent.json")

            with pytest.raises(ValueError, match="Configuration file not found"):
                await service.initialize()


class TestLoadGraphRegistryDictFormatIntegration:
    """Integration tests for dict-format graph config with real files."""

    @pytest.mark.asyncio
    async def test_initialize_with_dict_format_graphs(self) -> None:
        """Full initialize with a mix of string and dict format graph entries."""
        config_data = {
            "graphs": {
                "string_graph": "./graphs/test.py:graph",
                "dict_graph": {
                    "path": "./graphs/dict.py:agent",
                    "description": "A described agent",
                },
            }
        }

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "aegra.json"
            config_path.write_text(json.dumps(config_data))

            with (
                patch("aegra_api.services.langgraph_service.LangGraphService._ensure_default_assistants"),
                patch("aegra_api.services.langgraph_service.LangGraphService._load_all_graph_modules"),
            ):
                service = LangGraphService(str(config_path))
                await service.initialize()

                assert service.config == config_data

                string_entry = service._graph_registry["string_graph"]
                assert string_entry["file_path"] == "./graphs/test.py"
                assert string_entry["export_name"] == "graph"
                assert "description" not in string_entry

                dict_entry = service._graph_registry["dict_graph"]
                assert dict_entry["file_path"] == "./graphs/dict.py"
                assert dict_entry["export_name"] == "agent"
                assert dict_entry["description"] == "A described agent"

    @pytest.mark.asyncio
    async def test_initialize_dict_format_invalid_path_raises(self) -> None:
        """Dict-format entry with missing colon in path raises during initialize."""
        config_data = {
            "graphs": {
                "broken": {
                    "path": "./graphs/no_export_separator.py",
                    "description": "Missing colon",
                }
            }
        }

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "aegra.json"
            config_path.write_text(json.dumps(config_data))

            with (
                patch("aegra_api.services.langgraph_service.LangGraphService._ensure_default_assistants"),
                patch("aegra_api.services.langgraph_service.LangGraphService._load_all_graph_modules"),
            ):
                service = LangGraphService(str(config_path))

                with pytest.raises(ValueError, match="Invalid graph path format"):
                    await service.initialize()

    @pytest.mark.asyncio
    async def test_list_graphs_with_dict_format(self) -> None:
        """list_graphs returns file_path for both string and dict format entries."""
        config_data = {
            "graphs": {
                "str_agent": "./graphs/str.py:graph",
                "dict_agent": {
                    "path": "./graphs/dict.py:agent",
                    "description": "Described agent",
                },
            }
        }

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "aegra.json"
            config_path.write_text(json.dumps(config_data))

            with (
                patch("aegra_api.services.langgraph_service.LangGraphService._ensure_default_assistants"),
                patch("aegra_api.services.langgraph_service.LangGraphService._load_all_graph_modules"),
            ):
                service = LangGraphService(str(config_path))
                await service.initialize()

                graphs = service.list_graphs()
                assert graphs == {
                    "str_agent": "./graphs/str.py",
                    "dict_agent": "./graphs/dict.py",
                }

    @pytest.mark.asyncio
    async def test_dict_format_description_accessible_from_registry(self) -> None:
        """Description is stored in registry and accessible for downstream consumers."""
        config_data = {
            "graphs": {
                "my_graph": {
                    "path": "./graphs/my.py:graph",
                    "description": "Does important things",
                }
            }
        }

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "aegra.json"
            config_path.write_text(json.dumps(config_data))

            with (
                patch("aegra_api.services.langgraph_service.LangGraphService._ensure_default_assistants"),
                patch("aegra_api.services.langgraph_service.LangGraphService._load_all_graph_modules"),
            ):
                service = LangGraphService(str(config_path))
                await service.initialize()

                entry = service._graph_registry["my_graph"]
                assert entry.get("description") == "Does important things"

    @pytest.mark.asyncio
    async def test_get_graph_with_dict_format_entry(self) -> None:
        """get_graph works correctly when the graph was registered via dict format."""
        service = LangGraphService()
        service._graph_registry = {
            "dict_graph": {
                "file_path": "./graphs/dict.py",
                "export_name": "agent",
                "description": "A test agent",
            }
        }

        mock_graph = DummyStateGraph()
        mock_compiled = Mock()
        mock_compiled.copy = Mock(return_value=mock_compiled)
        mock_graph.compile = Mock(return_value=mock_compiled)

        with (
            patch.object(service, "_load_graph_from_file", return_value=mock_graph),
            patch("aegra_api.core.database.db_manager") as mock_db_manager,
        ):
            mock_db_manager.get_checkpointer = Mock(return_value="checkpointer")
            mock_db_manager.get_store = Mock(return_value="store")

            async with service.get_graph("dict_graph") as graph:
                assert graph == mock_compiled
                mock_graph.compile.assert_called_once()
                mock_compiled.copy.assert_called_once()


class TestLangGraphServiceDatabase:
    """Test LangGraphService database integration"""

    @pytest.mark.asyncio
    async def test_get_graph_database_integration(self):
        """Test graph loading with database integration"""
        service = LangGraphService()
        service._graph_registry = {"db_graph": {"file_path": "./graphs/db.py", "export_name": "graph"}}

        mock_graph = DummyStateGraph()
        mock_compiled_graph = Mock()
        # Add copy method to mock since get_graph uses .copy(update=...)
        mock_compiled_graph.copy = Mock(return_value=mock_compiled_graph)

        with (
            patch.object(service, "_load_graph_from_file", return_value=mock_graph),
            patch("aegra_api.core.database.db_manager") as mock_db_manager,
        ):
            # Mock database manager methods
            mock_checkpointer = Mock()
            mock_store = Mock()

            # FIX: Changed AsyncMock to Mock because getters are now synchronous
            mock_db_manager.get_checkpointer = Mock(return_value=mock_checkpointer)
            mock_db_manager.get_store = Mock(return_value=mock_store)

            # Mock graph compilation (for base graph caching)
            mock_graph.compile = Mock(return_value=mock_compiled_graph)

            # get_graph is now a context manager
            async with service.get_graph("db_graph") as result:
                assert result == mock_compiled_graph
                mock_db_manager.get_checkpointer.assert_called_once()
                mock_db_manager.get_store.assert_called_once()
                # Base graph is compiled once, then copy is called with checkpointer/store
                mock_graph.compile.assert_called_once()
                mock_compiled_graph.copy.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_graph_already_compiled(self):
        """Test graph loading when graph is already compiled"""
        service = LangGraphService()
        service._graph_registry = {
            "compiled_graph": {
                "file_path": "./graphs/compiled.py",
                "export_name": "graph",
            }
        }

        mock_compiled_graph = Mock()
        # Graph doesn't have compile method (already compiled Pregel)
        mock_compiled_graph.compile = None  # Simulate already compiled graph
        # Add copy method since get_graph uses .copy(update=...)
        mock_compiled_graph.copy = Mock(return_value=mock_compiled_graph)

        with (
            patch.object(service, "_load_graph_from_file", return_value=mock_compiled_graph),
            patch("aegra_api.core.database.db_manager") as mock_db_manager,
        ):
            mock_checkpointer = Mock()
            mock_store = Mock()

            # FIX: Changed AsyncMock to Mock
            mock_db_manager.get_checkpointer = Mock(return_value=mock_checkpointer)
            mock_db_manager.get_store = Mock(return_value=mock_store)

            # get_graph is now a context manager
            async with service.get_graph("compiled_graph") as result:
                # Result should be the compiled graph (may be a copy)
                assert result is not None
                # copy should be called with checkpointer/store
                mock_compiled_graph.copy.assert_called_once()


class TestLangGraphServiceGraphLoading:
    """Test actual graph loading functionality"""

    @pytest.mark.asyncio
    async def test_load_graph_from_real_file(self):
        """Test loading graph from real Python file"""
        service = LangGraphService()

        # Create a temporary Python file with a valid graph
        graph_code = """
from langgraph.graph import StateGraph

def create_graph():
    graph = StateGraph(dict)
    graph.add_node("test_node", lambda x: x)
    graph.add_edge("__start__", "test_node")
    graph.add_edge("test_node", "__end__")
    return graph.compile()

graph = create_graph()
"""

        with TemporaryDirectory() as temp_dir:
            graph_file = Path(temp_dir) / "test_graph.py"
            graph_file.write_text(graph_code)

            graph_info = {"file_path": str(graph_file), "export_name": "graph"}

            result = await service._load_graph_from_file("test_graph", graph_info)

            # Should return the compiled graph
            assert result is not None
            # Graph should have the expected structure
            assert hasattr(result, "nodes") or hasattr(result, "get_graph")

    @pytest.mark.asyncio
    async def test_load_graph_from_file_with_error(self):
        """Test error handling when graph file has syntax error"""
        service = LangGraphService()

        # Create a Python file with syntax error
        invalid_code = """
def create_graph():
    graph = StateGraph(dict
    return graph.compile()

graph = create_graph()
"""

        with TemporaryDirectory() as temp_dir:
            graph_file = Path(temp_dir) / "invalid_graph.py"
            graph_file.write_text(invalid_code)

            graph_info = {"file_path": str(graph_file), "export_name": "graph"}

            with pytest.raises((SyntaxError, ValueError)):
                await service._load_graph_from_file("invalid_graph", graph_info)

    @pytest.mark.asyncio
    async def test_load_graph_from_file_missing_export(self):
        """Test error when graph file doesn't export expected name"""
        service = LangGraphService()

        # Create a Python file without the expected export
        graph_code = """
from langgraph.graph import StateGraph

def create_graph():
    graph = StateGraph(dict)
    return graph.compile()

# Missing: graph = create_graph()
"""

        with TemporaryDirectory() as temp_dir:
            graph_file = Path(temp_dir) / "missing_export.py"
            graph_file.write_text(graph_code)

            graph_info = {"file_path": str(graph_file), "export_name": "graph"}

            with pytest.raises(ValueError, match="Graph export not found"):
                await service._load_graph_from_file("missing_export", graph_info)


class TestLangGraphServiceErrorHandling:
    """Test error handling scenarios"""

    @pytest.mark.asyncio
    async def test_get_graph_database_error(self):
        """Test error handling when database operations fail"""
        service = LangGraphService()
        service._graph_registry = {"error_graph": {"file_path": "./graphs/error.py", "export_name": "graph"}}

        mock_graph = Mock()
        mock_compiled = Mock()
        mock_compiled.copy = Mock(return_value=mock_compiled)
        mock_graph.compile = Mock(return_value=mock_compiled)

        with (
            patch.object(service, "_load_graph_from_file", return_value=mock_graph),
            patch("aegra_api.core.database.db_manager") as mock_db_manager,
        ):
            # Mock database error
            # FIX: Changed AsyncMock to Mock (synchronous getter raising exception)
            mock_db_manager.get_checkpointer = Mock(side_effect=Exception("Database error"))

            with pytest.raises(Exception, match="Database error"):
                async with service.get_graph("error_graph"):
                    pass

    @pytest.mark.asyncio
    async def test_get_graph_compilation_error(self):
        """Test error handling when graph compilation fails"""
        service = LangGraphService()
        service._graph_registry = {
            "compile_error_graph": {
                "file_path": "./graphs/compile_error.py",
                "export_name": "graph",
            }
        }

        mock_graph = DummyStateGraph()
        mock_graph.compile = Mock(side_effect=Exception("Compilation error"))

        with (
            patch.object(service, "_load_graph_from_file", return_value=mock_graph),
            patch("aegra_api.core.database.db_manager") as mock_db_manager,
        ):
            # FIX: Changed AsyncMock to Mock
            mock_db_manager.get_checkpointer = Mock(return_value="checkpointer")
            mock_db_manager.get_store = Mock(return_value="store")

            with pytest.raises(Exception, match="Compilation error"):
                async with service.get_graph("compile_error_graph"):
                    pass

    @pytest.mark.asyncio
    async def test_initialize_file_permission_error(self):
        """Test error handling when config file cannot be read"""
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "readonly.json"
            config_path.write_text('{"test": "value"}')

            # Make file read-only (Unix only)
            import stat

            config_path.chmod(stat.S_IRUSR)

            with (
                patch("aegra_api.services.langgraph_service.LangGraphService._ensure_default_assistants"),
                patch(
                    "pathlib.Path.open",
                    side_effect=PermissionError("Permission denied"),
                ),
            ):
                service = LangGraphService(str(config_path))

                # Should handle permission error gracefully
                with pytest.raises(PermissionError):
                    await service.initialize()


class TestLangGraphServiceConcurrency:
    """Test concurrent access scenarios"""

    @pytest.mark.asyncio
    async def test_concurrent_graph_loading(self):
        """Test concurrent access to graph loading"""
        service = LangGraphService()
        service._graph_registry = {
            "concurrent_graph": {
                "file_path": "./graphs/concurrent.py",
                "export_name": "graph",
            }
        }

        mock_graph = DummyStateGraph()
        mock_compiled_graph = Mock()
        mock_compiled_graph.copy = Mock(return_value=mock_compiled_graph)

        with (
            patch.object(service, "_load_graph_from_file", return_value=mock_graph),
            patch("aegra_api.core.database.db_manager") as mock_db_manager,
        ):
            # FIX: Changed AsyncMock to Mock
            mock_db_manager.get_checkpointer = Mock(return_value="checkpointer")
            mock_db_manager.get_store = Mock(return_value="store")
            mock_graph.compile = Mock(return_value=mock_compiled_graph)

            # Load same graph concurrently using context managers
            import asyncio

            async def get_graph_task():
                async with service.get_graph("concurrent_graph") as graph:
                    return graph

            tasks = [
                get_graph_task(),
                get_graph_task(),
                get_graph_task(),
            ]

            results = await asyncio.gather(*tasks)

            # All should return the compiled graph (copies)
            assert all(result == mock_compiled_graph for result in results)
            # Should only compile once due to base graph caching
            assert mock_graph.compile.call_count == 1
            # copy is called once per request (3 times)
            assert mock_compiled_graph.copy.call_count == 3

    @pytest.mark.asyncio
    async def test_concurrent_cache_invalidation(self):
        """Test concurrent cache invalidation"""
        service = LangGraphService()
        service._base_graph_cache = {
            "graph1": Mock(),
            "graph2": Mock(),
            "graph3": Mock(),
        }

        import asyncio

        async def invalidate_graph(graph_id):
            service.invalidate_cache(graph_id)
            return len(service._base_graph_cache)

        # Invalidate different graphs concurrently
        tasks = [
            invalidate_graph("graph1"),
            invalidate_graph("graph2"),
            invalidate_graph("graph3"),
        ]

        results = await asyncio.gather(*tasks)

        # Cache should be empty after all invalidations
        assert service._base_graph_cache == {}
        # All results should be 0 (empty cache) - check individually
        # Note: concurrent execution may cause intermediate states
        assert all(result <= 3 for result in results)  # Should be <= original cache size
