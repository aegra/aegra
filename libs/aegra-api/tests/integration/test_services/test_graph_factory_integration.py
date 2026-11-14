"""Integration tests for graph factory support in LangGraphService.

These tests verify the end-to-end flow from factory detection in
``_load_graph_from_file`` through ``get_graph()`` and
``get_graph_for_validation()`` with realistic graph modules.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langgraph.pregel import Pregel
from langgraph_sdk.runtime import (
    ServerRuntime,
    _ExecutionRuntime,
    _ReadRuntime,
)
from pydantic import BaseModel

from aegra_api.services.graph_factory import (
    _FACTORY_CONTEXT_TYPES,
    _FACTORY_KWARGS,
    ContextValidationError,
    classify_factory,
)
from aegra_api.services.langgraph_service import LangGraphService

# Module-level context models for integration tests.
# Defined here (not inside test methods) so that ``typing.get_type_hints()``
# can resolve ``ServerRuntime[_TestMyConfig]`` when ``from __future__ import
# annotations`` turns all annotations into strings.


class _TestMyConfig(BaseModel):
    """Pydantic config model for typed context tests."""

    model_name: str
    temperature: float


class _TestRequiredConfig(BaseModel):
    """Pydantic config model for validation fallback tests."""

    required_field: str


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_factory_state() -> Iterator[None]:
    """Ensure factory registry and dynamically loaded graph modules are clean for each test."""
    _FACTORY_KWARGS.clear()
    _FACTORY_CONTEXT_TYPES.clear()
    try:
        yield
    finally:
        _FACTORY_KWARGS.clear()
        _FACTORY_CONTEXT_TYPES.clear()
        for key in list(sys.modules.keys()):
            if key.startswith("aegra_graphs."):
                sys.modules.pop(key, None)


@pytest.fixture()
def graph_module_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for graph modules."""
    return tmp_path


def _write_graph_module(directory: Path, filename: str, code: str) -> Path:
    """Write a Python module to disk and return its path."""
    filepath = directory / filename
    filepath.write_text(code)
    return filepath


# ---------------------------------------------------------------------------
# Static graph (non-callable) — existing behavior preserved
# ---------------------------------------------------------------------------


class TestStaticGraphPreserved:
    """Ensure existing static Pregel/StateGraph exports still work."""

    @pytest.mark.asyncio
    async def test_static_non_callable_export(self, graph_module_dir: Path) -> None:
        """A module exporting a non-callable object should be returned as-is."""
        _write_graph_module(
            graph_module_dir,
            "static_graph.py",
            """\
# Simulate a static export (non-callable object)
class FakeGraph:
    '''Non-callable graph-like object.'''
    pass

graph = FakeGraph()
""",
        )

        service = LangGraphService()
        service.config_path = graph_module_dir / "aegra.json"
        service._graph_registry = {
            "static": {"file_path": str(graph_module_dir / "static_graph.py"), "export_name": "graph"}
        }

        result = await service._load_graph_from_file("static", service._graph_registry["static"])

        # Should NOT be classified as a factory (not callable)
        assert "static" not in _FACTORY_KWARGS
        # Should be the raw export
        assert result.__class__.__name__ == "FakeGraph"


# ---------------------------------------------------------------------------
# 0-arg factory (existing behavior preserved)
# ---------------------------------------------------------------------------


class TestZeroArgFactory:
    """0-arg callable factories should be called at load time, not registered."""

    @pytest.mark.asyncio
    async def test_zero_arg_sync_factory(self, graph_module_dir: Path) -> None:
        _write_graph_module(
            graph_module_dir,
            "zero_arg.py",
            """\
from unittest.mock import Mock
from langgraph.pregel import Pregel

def graph():
    g = Mock(spec=Pregel)
    g._called_factory = True
    return g
""",
        )

        service = LangGraphService()
        service.config_path = graph_module_dir / "aegra.json"
        service._graph_registry = {"zero": {"file_path": str(graph_module_dir / "zero_arg.py"), "export_name": "graph"}}

        result = await service._load_graph_from_file("zero", service._graph_registry["zero"])

        # Should NOT be registered as a factory (0-arg is called once at load)
        assert "zero" not in _FACTORY_KWARGS
        # The factory should have been called and returned its result
        assert hasattr(result, "_called_factory")

    @pytest.mark.asyncio
    async def test_zero_arg_async_factory(self, graph_module_dir: Path) -> None:
        _write_graph_module(
            graph_module_dir,
            "zero_arg_async.py",
            """\
from unittest.mock import Mock
from langgraph.pregel import Pregel

async def graph():
    g = Mock(spec=Pregel)
    g._async_factory = True
    return g
""",
        )

        service = LangGraphService()
        service.config_path = graph_module_dir / "aegra.json"
        service._graph_registry = {
            "zero_async": {
                "file_path": str(graph_module_dir / "zero_arg_async.py"),
                "export_name": "graph",
            }
        }

        result = await service._load_graph_from_file("zero_async", service._graph_registry["zero_async"])

        assert "zero_async" not in _FACTORY_KWARGS
        assert hasattr(result, "_async_factory")


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


class TestConfigFactory:
    """Config-accepting factory: ``def graph(config: dict) -> Graph``."""

    @pytest.mark.asyncio
    async def test_config_factory_detected_and_registered(self, graph_module_dir: Path) -> None:
        _write_graph_module(
            graph_module_dir,
            "config_factory.py",
            """\
from unittest.mock import Mock
from langgraph.pregel import Pregel

def graph(config):
    g = Mock(spec=Pregel)
    g._config = config
    g.copy = Mock(return_value=g)
    return g
""",
        )

        service = LangGraphService()
        service.config_path = graph_module_dir / "aegra.json"
        service._graph_registry = {
            "cfg": {
                "file_path": str(graph_module_dir / "config_factory.py"),
                "export_name": "graph",
            }
        }

        result = await service._load_graph_from_file("cfg", service._graph_registry["cfg"])

        # Should be registered as a factory
        assert "cfg" in _FACTORY_KWARGS
        # Should have stored the callable
        assert "cfg" in service._graph_factories
        # Factory graphs return None — the factory is NOT called with defaults
        # at load time; it will be invoked per-request with proper user/config.
        assert result is None


# ---------------------------------------------------------------------------
# Runtime factory
# ---------------------------------------------------------------------------


class TestRuntimeFactory:
    """Runtime-accepting factory: ``def graph(runtime: ServerRuntime) -> Graph``."""

    @pytest.mark.asyncio
    async def test_runtime_factory_detected_and_registered(self, graph_module_dir: Path) -> None:
        _write_graph_module(
            graph_module_dir,
            "runtime_factory.py",
            """\
from unittest.mock import Mock
from langgraph.pregel import Pregel
from langgraph_sdk.runtime import ServerRuntime

def graph(runtime: ServerRuntime):
    g = Mock(spec=Pregel)
    g._runtime = runtime
    g.copy = Mock(return_value=g)
    return g
""",
        )

        service = LangGraphService()
        service.config_path = graph_module_dir / "aegra.json"
        service._graph_registry = {
            "rt": {
                "file_path": str(graph_module_dir / "runtime_factory.py"),
                "export_name": "graph",
            }
        }

        result = await service._load_graph_from_file("rt", service._graph_registry["rt"])

        assert "rt" in _FACTORY_KWARGS
        assert "rt" in service._graph_factories
        # Factory graphs return None — no default-args call at load time.
        assert result is None


# ---------------------------------------------------------------------------
# get_graph() with factory — per-request invocation
# ---------------------------------------------------------------------------


class TestGetGraphWithFactory:
    """Test that get_graph() invokes factories per-request."""

    @pytest.mark.asyncio
    async def test_factory_called_per_request(self) -> None:
        """Each call to get_graph() should invoke the factory, not return cached."""
        call_count = 0

        def factory(config: dict[str, Any]) -> Mock:
            nonlocal call_count
            call_count += 1
            g = Mock(spec=Pregel)
            g._call_number = call_count
            g.copy = Mock(return_value=g)
            return g

        service = LangGraphService()
        service._graph_registry = {"f": {"file_path": "f.py", "export_name": "graph"}}
        service._graph_factories = {"f": factory}

        # Manually register factory dispatch hook

        classify_factory(factory, "f")

        with patch("aegra_api.core.database.db_manager") as mock_db:
            mock_db.get_checkpointer = Mock(return_value=Mock())
            mock_db.get_store = Mock(return_value=Mock())

            async with service.get_graph("f", config={"configurable": {}}) as _g1:
                pass
            async with service.get_graph("f", config={"configurable": {}}) as _g2:
                pass

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_factory_receives_config(self) -> None:
        """Factory should receive the request config."""
        received_config: dict[str, Any] = {}

        def factory(config: dict[str, Any]) -> Mock:
            received_config.update(config)
            g = Mock(spec=Pregel)
            g.copy = Mock(return_value=g)
            return g

        service = LangGraphService()
        service._graph_registry = {"fc": {"file_path": "f.py", "export_name": "graph"}}
        service._graph_factories = {"fc": factory}

        classify_factory(factory, "fc")

        with patch("aegra_api.core.database.db_manager") as mock_db:
            mock_db.get_checkpointer = Mock(return_value=Mock())
            mock_db.get_store = Mock(return_value=Mock())

            test_config = {"configurable": {"thread_id": "t1", "model": "gpt-4"}}
            async with service.get_graph("fc", config=test_config) as _graph:
                pass

        assert received_config.get("configurable", {}).get("thread_id") == "t1"

    @pytest.mark.asyncio
    async def test_factory_receives_execution_runtime(self) -> None:
        """Factory with ServerRuntime param receives _ExecutionRuntime for create_run."""
        received_runtime = None

        def factory(runtime: ServerRuntime) -> Mock:
            nonlocal received_runtime
            received_runtime = runtime
            g = Mock(spec=Pregel)
            g.copy = Mock(return_value=g)
            return g

        service = LangGraphService()
        service._graph_registry = {"frt": {"file_path": "f.py", "export_name": "graph"}}
        service._graph_factories = {"frt": factory}

        classify_factory(factory, "frt")

        with patch("aegra_api.core.database.db_manager") as mock_db:
            mock_db.get_checkpointer = Mock(return_value=Mock())
            mock_db.get_store = Mock(return_value=Mock())

            async with service.get_graph(
                "frt",
                access_context="threads.create_run",
            ) as _graph:
                pass

        assert isinstance(received_runtime, _ExecutionRuntime)
        assert received_runtime.access_context == "threads.create_run"

    @pytest.mark.asyncio
    async def test_factory_receives_read_runtime_for_schema(self) -> None:
        """Factory with ServerRuntime param receives _ReadRuntime for assistants.read."""
        received_runtime = None

        def factory(runtime: ServerRuntime) -> Mock:
            nonlocal received_runtime
            received_runtime = runtime
            g = Mock(spec=Pregel)
            g.copy = Mock(return_value=g)
            return g

        service = LangGraphService()
        service._graph_registry = {"frr": {"file_path": "f.py", "export_name": "graph"}}
        service._graph_factories = {"frr": factory}

        classify_factory(factory, "frr")

        with patch("aegra_api.core.database.db_manager") as mock_db:
            mock_db.get_checkpointer = Mock(return_value=Mock())
            mock_db.get_store = Mock(return_value=Mock())

            async with service.get_graph(
                "frr",
                access_context="assistants.read",
            ) as _graph:
                pass

        assert isinstance(received_runtime, _ReadRuntime)
        assert received_runtime.access_context == "assistants.read"


# ---------------------------------------------------------------------------
# get_graph_for_validation() with factory
# ---------------------------------------------------------------------------


class TestGetGraphForValidationWithFactory:
    """Test that get_graph_for_validation() also invokes factories."""

    @pytest.mark.asyncio
    async def test_validation_invokes_factory(self) -> None:
        """get_graph_for_validation should call factory with assistants.read context."""
        received_runtime = None

        def factory(runtime: ServerRuntime) -> Mock:
            nonlocal received_runtime
            received_runtime = runtime
            g = Mock(spec=Pregel)
            return g

        service = LangGraphService()
        service._graph_registry = {"fv": {"file_path": "f.py", "export_name": "graph"}}
        service._graph_factories = {"fv": factory}

        classify_factory(factory, "fv")

        result = await service.get_graph_for_validation("fv")

        assert isinstance(result, Pregel)
        assert isinstance(received_runtime, _ReadRuntime)
        assert received_runtime.access_context == "assistants.read"

    @pytest.mark.asyncio
    async def test_validation_non_factory_returns_cached(self) -> None:
        """Non-factory graphs still return cached base graph from validation."""
        mock_base = Mock(spec=Pregel)

        service = LangGraphService()
        service._graph_registry = {"s": {"file_path": "s.py", "export_name": "graph"}}
        service._base_graph_cache = {"s": mock_base}

        result = await service.get_graph_for_validation("s")

        assert result is mock_base


# ---------------------------------------------------------------------------
# Cache invalidation with factory
# ---------------------------------------------------------------------------


class TestCacheInvalidationWithFactory:
    """Test that invalidate_cache clears both base cache and factory registry."""

    def test_invalidate_specific_clears_factory(self) -> None:
        service = LangGraphService()
        service._base_graph_cache = {"g": Mock()}
        service._graph_factories = {"g": Mock()}
        _FACTORY_KWARGS["g"] = lambda c, r: {}

        service.invalidate_cache("g")

        assert "g" not in service._base_graph_cache
        assert "g" not in service._graph_factories
        assert "g" not in _FACTORY_KWARGS

    def test_invalidate_all_clears_factories(self) -> None:
        service = LangGraphService()
        service._base_graph_cache = {"g1": Mock(), "g2": Mock()}
        service._graph_factories = {"g1": Mock()}
        _FACTORY_KWARGS["g1"] = lambda c, r: {}

        service.invalidate_cache()

        assert len(service._base_graph_cache) == 0
        assert len(service._graph_factories) == 0
        assert len(_FACTORY_KWARGS) == 0


# ---------------------------------------------------------------------------
# Non-factory get_graph() — existing behavior preserved
# ---------------------------------------------------------------------------


class TestNonFactoryGetGraphPreserved:
    """Ensure that non-factory graphs still work through get_graph()."""

    @pytest.mark.asyncio
    async def test_static_graph_uses_cached_base(self) -> None:
        """Non-factory graph should use the cached base graph, not invoke factory."""
        mock_base = Mock(spec=Pregel)
        mock_copy = Mock(spec=Pregel)
        mock_base.copy = Mock(return_value=mock_copy)

        service = LangGraphService()
        service._graph_registry = {"s": {"file_path": "s.py", "export_name": "graph"}}
        service._base_graph_cache = {"s": mock_base}

        with patch("aegra_api.core.database.db_manager") as mock_db:
            mock_db.get_checkpointer = Mock(return_value="cp")
            mock_db.get_store = Mock(return_value="st")

            async with service.get_graph("s") as graph:
                assert graph is mock_copy

            mock_base.copy.assert_called_once()
            call_kwargs = mock_base.copy.call_args[1]["update"]
            assert call_kwargs["checkpointer"] == "cp"
            assert call_kwargs["store"] == "st"
            assert "config" in call_kwargs


# ---------------------------------------------------------------------------
# Context passthrough via get_graph()
# ---------------------------------------------------------------------------


class TestGetGraphWithContext:
    """Test that get_graph() passes context through to factory runtime."""

    @pytest.mark.asyncio
    async def test_typed_context_coerced_to_pydantic(self) -> None:
        """Factory with ServerRuntime[_TestMyConfig] → runtime.context is coerced."""
        received_runtime = None

        def factory(runtime: ServerRuntime[_TestMyConfig]) -> Mock:
            nonlocal received_runtime
            received_runtime = runtime
            g = Mock(spec=Pregel)
            g.copy = Mock(return_value=g)
            return g

        service = LangGraphService()
        service._graph_registry = {"typed": {"file_path": "f.py", "export_name": "graph"}}
        service._graph_factories = {"typed": factory}
        classify_factory(factory, "typed")

        with patch("aegra_api.core.database.db_manager") as mock_db:
            mock_db.get_checkpointer = Mock(return_value=Mock())
            mock_db.get_store = Mock(return_value=Mock())

            async with service.get_graph(
                "typed",
                access_context="threads.create_run",
                context={"model_name": "gpt-4", "temperature": 0.7},
            ) as _graph:
                pass

        assert isinstance(received_runtime, _ExecutionRuntime)
        assert isinstance(received_runtime.context, _TestMyConfig)
        assert received_runtime.context.model_name == "gpt-4"
        assert received_runtime.context.temperature == 0.7

    @pytest.mark.asyncio
    async def test_plain_runtime_passes_raw_dict(self) -> None:
        """Factory with plain ServerRuntime → runtime.context is the raw dict."""
        received_runtime = None

        def factory(runtime: ServerRuntime) -> Mock:
            nonlocal received_runtime
            received_runtime = runtime
            g = Mock(spec=Pregel)
            g.copy = Mock(return_value=g)
            return g

        service = LangGraphService()
        service._graph_registry = {"plain": {"file_path": "f.py", "export_name": "graph"}}
        service._graph_factories = {"plain": factory}
        classify_factory(factory, "plain")

        ctx = {"key": "value", "number": 42}
        with patch("aegra_api.core.database.db_manager") as mock_db:
            mock_db.get_checkpointer = Mock(return_value=Mock())
            mock_db.get_store = Mock(return_value=Mock())

            async with service.get_graph(
                "plain",
                access_context="threads.create_run",
                context=ctx,
            ) as _graph:
                pass

        assert isinstance(received_runtime, _ExecutionRuntime)
        assert received_runtime.context is ctx

    @pytest.mark.asyncio
    async def test_invalid_context_never_reaches_the_factory(self) -> None:
        """An invalid context fails the run rather than reaching the factory as a raw dict."""
        called = False

        def factory(runtime: ServerRuntime[_TestRequiredConfig]) -> Mock:
            nonlocal called
            called = True
            g = Mock(spec=Pregel)
            g.copy = Mock(return_value=g)
            return g

        service = LangGraphService()
        service._graph_registry = {"strict": {"file_path": "f.py", "export_name": "graph"}}
        service._graph_factories = {"strict": factory}
        classify_factory(factory, "strict")

        with (
            patch("aegra_api.core.database.db_manager") as mock_db,
            pytest.raises(ContextValidationError) as exc_info,
        ):
            mock_db.get_checkpointer = Mock(return_value=Mock())
            mock_db.get_store = Mock(return_value=Mock())

            async with service.get_graph(
                "strict",
                access_context="threads.create_run",
                context={"wrong_field": "oops"},
            ) as _graph:
                pass

        assert not called
        assert exc_info.value.errors == [
            {"loc": ["context", "required_field"], "msg": "Field required", "type": "missing"}
        ]


# ---------------------------------------------------------------------------
# Regression: factory NOT called with user=None at startup (issue #245)
# ---------------------------------------------------------------------------


class TestFactoryNotCalledAtStartup:
    """Verify that factory graphs are classified but NOT invoked at load time.

    Regression test for https://github.com/aegra/aegra/issues/245:
    dynamic graph factories were called with ``user=None`` during server
    startup for schema extraction, causing the first request to get a graph
    compiled without user context.
    """

    @pytest.mark.asyncio
    async def test_factory_not_invoked_during_load(self, graph_module_dir: Path) -> None:
        """_load_all_graph_modules must NOT call the factory for config/runtime factories."""
        _write_graph_module(
            graph_module_dir,
            "tracked_factory.py",
            """\
from unittest.mock import Mock
from langgraph.pregel import Pregel
from langgraph_sdk.runtime import ServerRuntime

call_log: list[dict] = []

def graph(config: dict, runtime: ServerRuntime):
    call_log.append({"config": config, "user": runtime.user})
    g = Mock(spec=Pregel)
    g.copy = Mock(return_value=g)
    return g
""",
        )

        service = LangGraphService()
        service.config_path = graph_module_dir / "aegra.json"
        service._graph_registry = {
            "tracked": {
                "file_path": str(graph_module_dir / "tracked_factory.py"),
                "export_name": "graph",
            }
        }

        await service._load_all_graph_modules()

        # Factory should be registered but NOT called
        assert "tracked" in service._graph_factories
        assert "tracked" not in service._base_graph_cache

        # Verify the factory was never invoked (no calls logged)
        mod = importlib.import_module("aegra_graphs.tracked")
        assert mod.call_log == [], (
            "Factory should not be called at load time; "
            f"got {len(mod.call_log)} call(s) with user={mod.call_log[0]['user'] if mod.call_log else '?'}"
        )

    @pytest.mark.asyncio
    async def test_first_request_receives_user(self) -> None:
        """First get_graph() call should pass user and config through to factory."""
        received_calls: list[dict[str, Any]] = []

        def factory(config: dict[str, Any], runtime: ServerRuntime) -> Mock:
            received_calls.append({"config": config, "user": runtime.user})
            g = Mock(spec=Pregel)
            g.copy = Mock(return_value=g)
            return g

        service = LangGraphService()
        service._graph_registry = {"u": {"file_path": "u.py", "export_name": "graph"}}
        service._graph_factories = {"u": factory}
        classify_factory(factory, "u")

        mock_user = Mock()
        mock_user.identity = "test-user"
        request_config: dict[str, Any] = {"configurable": {"thread_id": "t-1", "run_id": "r-1"}}

        with patch("aegra_api.core.database.db_manager") as mock_db:
            mock_db.get_checkpointer = Mock(return_value=Mock())
            mock_db.get_store = Mock(return_value=Mock())

            async with service.get_graph(
                "u",
                config=request_config,
                access_context="threads.create_run",
                user=mock_user,
            ) as _graph:
                pass

        assert len(received_calls) == 1
        assert received_calls[0]["user"] is mock_user
        assert received_calls[0]["config"]["configurable"]["thread_id"] == "t-1"

    @pytest.mark.asyncio
    async def test_post_invalidation_request_receives_user(self, graph_module_dir: Path) -> None:
        """After invalidate_cache(), the first get_graph() must still invoke the factory with the real user."""
        _write_graph_module(
            graph_module_dir,
            "post_invalidation.py",
            """\
from unittest.mock import Mock
from langgraph.pregel import Pregel
from langgraph_sdk.runtime import ServerRuntime

received_calls: list[dict] = []

def graph(config: dict, runtime: ServerRuntime):
    received_calls.append({"config": config, "user": runtime.user})
    g = Mock(spec=Pregel)
    g.copy = Mock(return_value=g)
    return g
""",
        )

        service = LangGraphService()
        service.config_path = graph_module_dir / "aegra.json"
        service._graph_registry = {
            "pi": {
                "file_path": str(graph_module_dir / "post_invalidation.py"),
                "export_name": "graph",
            }
        }

        # Initial load — factory is classified
        await service._load_all_graph_modules()
        assert "pi" in service._graph_factories

        # Simulate hot-reload
        service.invalidate_cache("pi")
        assert "pi" not in service._graph_factories

        mock_user = Mock()
        mock_user.identity = "post-invalidation-user"
        request_config: dict[str, Any] = {"configurable": {"thread_id": "t-2", "run_id": "r-2"}}

        with patch("aegra_api.core.database.db_manager") as mock_db:
            mock_db.get_checkpointer = Mock(return_value=Mock())
            mock_db.get_store = Mock(return_value=Mock())

            async with service.get_graph(
                "pi",
                config=request_config,
                access_context="threads.create_run",
                user=mock_user,
            ) as _graph:
                pass

        mod = importlib.import_module("aegra_graphs.pi")
        # The factory must be called exactly once — with the real user only.
        # No spurious user=None invocation from _call_factory_with_defaults.
        assert len(mod.received_calls) == 1, (
            f"Expected 1 factory call, got {len(mod.received_calls)}: "
            f"{[c['user'].identity if hasattr(c['user'], 'identity') else c['user'] for c in mod.received_calls]}"
        )
        assert mod.received_calls[0]["user"] is mock_user
        assert mod.received_calls[0]["config"]["configurable"]["thread_id"] == "t-2"
