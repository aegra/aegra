"""Unit tests for graph factory classification, runtime construction, and dispatch."""

import dataclasses
import inspect
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import Mock, patch

import pytest
from langgraph.graph import StateGraph
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
    AegraExecutionRuntime,
    AegraReadRuntime,
    _classify_factory,
    _extract_context_type,
    _is_runtime_annotation,
    build_server_runtime,
    classify_factory,
    clear_factory_registry,
    coerce_context,
    generate_graph,
    invoke_factory,
    is_factory,
    is_for_execution,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_factory_registry() -> Iterator[None]:
    """Ensure the factory registry is clean before and after each test."""
    _FACTORY_KWARGS.clear()
    _FACTORY_CONTEXT_TYPES.clear()
    try:
        yield
    finally:
        _FACTORY_KWARGS.clear()
        _FACTORY_CONTEXT_TYPES.clear()


# ---------------------------------------------------------------------------
# _is_runtime_annotation
# ---------------------------------------------------------------------------


class TestIsRuntimeAnnotation:
    """Test runtime annotation detection."""

    def test_empty_annotation(self) -> None:
        assert _is_runtime_annotation(inspect.Parameter.empty) is False

    def test_server_runtime_type_alias(self) -> None:
        assert _is_runtime_annotation(ServerRuntime) is True

    def test_execution_runtime_class(self) -> None:
        assert _is_runtime_annotation(_ExecutionRuntime) is True

    def test_read_runtime_class(self) -> None:
        assert _is_runtime_annotation(_ReadRuntime) is True

    def test_unrelated_type(self) -> None:
        assert _is_runtime_annotation(str) is False
        assert _is_runtime_annotation(dict) is False

    def test_none_type(self) -> None:
        assert _is_runtime_annotation(None) is False

    def test_non_type_object(self) -> None:
        assert _is_runtime_annotation(42) is False
        assert _is_runtime_annotation("ServerRuntime") is False


# ---------------------------------------------------------------------------
# classify_factory / _classify_factory
# ---------------------------------------------------------------------------


class TestClassifyFactory:
    """Test factory classification by parameter signature."""

    def test_classify_no_params(self) -> None:
        """0-arg factory — no dispatch hook registered."""

        def make_graph() -> None:
            pass

        hook, ctx_type = _classify_factory(make_graph)
        assert hook is None
        assert ctx_type is None

    def test_classify_config_param_no_annotation(self) -> None:
        """1-param factory without annotation → treated as config factory."""

        def make_graph(config) -> None:
            pass

        hook, ctx_type = _classify_factory(make_graph)
        assert hook is not None
        assert ctx_type is None
        kwargs = hook({"key": "val"}, Mock())
        assert kwargs == {"config": {"key": "val"}}

    def test_classify_config_param_dict_annotation(self) -> None:
        """1-param factory with dict annotation → treated as config factory."""

        def make_graph(config: dict[str, Any]) -> None:
            pass

        hook, ctx_type = _classify_factory(make_graph)
        assert hook is not None
        assert ctx_type is None
        mock_runtime = Mock()
        kwargs = hook({"configurable": {}}, mock_runtime)
        assert kwargs == {"config": {"configurable": {}}}

    def test_classify_runtime_param(self) -> None:
        """1-param factory with ServerRuntime annotation → runtime factory."""

        def make_graph(runtime: ServerRuntime) -> None:
            pass

        hook, ctx_type = _classify_factory(make_graph)
        assert hook is not None
        assert ctx_type is None  # plain ServerRuntime, no T
        mock_runtime = Mock()
        kwargs = hook({}, mock_runtime)
        assert kwargs == {"runtime": mock_runtime}

    def test_classify_runtime_param_execution(self) -> None:
        """1-param factory with _ExecutionRuntime annotation → runtime factory."""

        def make_graph(rt: _ExecutionRuntime) -> None:
            pass

        hook, _ctx_type = _classify_factory(make_graph)
        assert hook is not None

    def test_classify_runtime_param_read(self) -> None:
        """1-param factory with _ReadRuntime annotation → runtime factory."""

        def make_graph(rt: _ReadRuntime) -> None:
            pass

        hook, _ctx_type = _classify_factory(make_graph)
        assert hook is not None

    def test_classify_both_params_runtime_first(self) -> None:
        """2-param factory (runtime, config) — both are passed as kwargs."""

        def make_graph(runtime: ServerRuntime, config: dict) -> None:
            pass

        hook, ctx_type = _classify_factory(make_graph)
        assert hook is not None
        assert ctx_type is None
        mock_runtime = Mock()
        config = {"configurable": {}}
        kwargs = hook(config, mock_runtime)
        assert kwargs == {"runtime": mock_runtime, "config": config}

    def test_classify_both_params_config_first(self) -> None:
        """2-param factory (config, runtime) — order doesn't matter for dispatch."""

        def make_graph(config: dict, runtime: ServerRuntime) -> None:
            pass

        hook, ctx_type = _classify_factory(make_graph)
        assert hook is not None
        assert ctx_type is None
        mock_runtime = Mock()
        config = {"configurable": {}}
        kwargs = hook(config, mock_runtime)
        assert kwargs == {"runtime": mock_runtime, "config": config}

    def test_classify_three_params_raises(self) -> None:
        """3+ params → ValueError."""

        def make_graph(a: ServerRuntime, b: dict, c: str) -> None:
            pass

        with pytest.raises(ValueError, match="must take 0, 1, or 2 arguments"):
            _classify_factory(make_graph)

    def test_classify_two_runtime_params_raises(self) -> None:
        """2 params both annotated as ServerRuntime → ValueError."""

        def make_graph(rt1: ServerRuntime, rt2: _ExecutionRuntime) -> None:
            pass

        with pytest.raises(ValueError, match="both annotated as ServerRuntime"):
            _classify_factory(make_graph)

    def test_classify_two_unannotated_params_raises(self) -> None:
        """2 params with no annotations → ValueError (neither is ServerRuntime)."""

        def make_graph(a, b) -> None:
            pass

        with pytest.raises(ValueError, match="neither is annotated as ServerRuntime"):
            _classify_factory(make_graph)


class TestClassifyFactoryRegistration:
    """Test the public classify_factory (with graph_id registration)."""

    def test_classify_registers_factory(self) -> None:
        def make_graph(config: dict) -> None:
            pass

        classify_factory(make_graph, "my_graph")
        assert is_factory("my_graph") is True

    def test_classify_no_arg_not_registered(self) -> None:
        def make_graph() -> None:
            pass

        classify_factory(make_graph, "my_graph")
        assert is_factory("my_graph") is False

    def test_classify_idempotent(self) -> None:
        """Calling classify_factory twice with the same graph_id is a no-op."""
        call_count = 0

        def make_graph(config: dict) -> None:
            nonlocal call_count
            call_count += 1

        classify_factory(make_graph, "my_graph")
        classify_factory(make_graph, "my_graph")  # second call should be a no-op
        assert is_factory("my_graph") is True


# ---------------------------------------------------------------------------
# is_factory / is_for_execution
# ---------------------------------------------------------------------------


class TestFactoryStateQueries:
    """Test is_factory and is_for_execution helpers."""

    def test_is_factory_registered(self) -> None:
        _FACTORY_KWARGS["test_graph"] = lambda c, r: {}
        assert is_factory("test_graph") is True

    def test_is_factory_not_registered(self) -> None:
        assert is_factory("unknown_graph") is False

    def test_is_for_execution_create_run(self) -> None:
        assert is_for_execution("threads.create_run") is True

    def test_is_for_execution_threads_update(self) -> None:
        assert is_for_execution("threads.update") is False

    def test_is_for_execution_threads_read(self) -> None:
        assert is_for_execution("threads.read") is False

    def test_is_for_execution_assistants_read(self) -> None:
        assert is_for_execution("assistants.read") is False


# ---------------------------------------------------------------------------
# build_server_runtime
# ---------------------------------------------------------------------------


class TestBuildServerRuntime:
    """Test ServerRuntime construction."""

    def test_build_execution_runtime(self) -> None:
        mock_store = Mock()
        mock_user = Mock()

        runtime = build_server_runtime(
            access_context="threads.create_run",
            store=mock_store,
            user=mock_user,
        )

        assert isinstance(runtime, _ExecutionRuntime)
        assert runtime.access_context == "threads.create_run"
        assert runtime.user is mock_user
        assert runtime.store is mock_store

    def test_build_read_runtime_threads_read(self) -> None:
        mock_store = Mock()
        mock_user = Mock()

        runtime = build_server_runtime(
            access_context="threads.read",
            store=mock_store,
            user=mock_user,
        )

        assert isinstance(runtime, _ReadRuntime)
        assert runtime.access_context == "threads.read"
        assert runtime.user is mock_user
        assert runtime.store is mock_store

    def test_build_read_runtime_threads_update(self) -> None:
        runtime = build_server_runtime(
            access_context="threads.update",
            store=Mock(),
        )
        assert isinstance(runtime, _ReadRuntime)

    def test_build_read_runtime_assistants_read(self) -> None:
        runtime = build_server_runtime(
            access_context="assistants.read",
            store=None,
        )
        assert isinstance(runtime, _ReadRuntime)
        assert runtime.store is None

    def test_build_runtime_no_user_no_auth_ctx(self) -> None:
        """When no user is passed and no auth context exists, user should be None."""
        with patch("aegra_api.services.graph_factory.get_auth_ctx", return_value=None):
            runtime = build_server_runtime(
                access_context="threads.create_run",
                store=Mock(),
            )
            assert runtime.user is None

    def test_build_runtime_no_user_falls_back_to_auth_ctx(self) -> None:
        """When no user is passed, falls back to auth context."""
        mock_auth_ctx = Mock()
        mock_auth_ctx.user = Mock(identity="ctx_user")

        with patch("aegra_api.services.graph_factory.get_auth_ctx", return_value=mock_auth_ctx):
            runtime = build_server_runtime(
                access_context="threads.create_run",
                store=Mock(),
            )
            assert runtime.user is mock_auth_ctx.user


# ---------------------------------------------------------------------------
# invoke_factory
# ---------------------------------------------------------------------------


class TestInvokeFactory:
    """Test factory invocation."""

    def test_invoke_no_args_factory(self) -> None:
        """Factory with no dispatch hook → called with no args."""
        mock_graph = Mock()
        factory = Mock(return_value=mock_graph)

        result = invoke_factory(factory, "unregistered_graph", {}, Mock())

        factory.assert_called_once_with()
        assert result is mock_graph

    def test_invoke_config_factory(self) -> None:
        """Config factory → called with config kwarg."""

        def make_graph(config: dict) -> str:
            return f"graph_with_{config.get('key')}"

        classify_factory(make_graph, "cfg_graph")
        config = {"key": "value"}

        result = invoke_factory(make_graph, "cfg_graph", config, Mock())

        assert result == "graph_with_value"

    def test_invoke_runtime_factory(self) -> None:
        """Runtime factory → called with runtime kwarg."""
        received_runtime = None

        def make_graph(runtime: ServerRuntime) -> str:
            nonlocal received_runtime
            received_runtime = runtime
            return "runtime_graph"

        classify_factory(make_graph, "rt_graph")
        mock_runtime = Mock()

        result = invoke_factory(make_graph, "rt_graph", {}, mock_runtime)

        assert result == "runtime_graph"
        assert received_runtime is mock_runtime

    def test_invoke_both_factory(self) -> None:
        """Factory with both config and runtime → called with both kwargs."""
        received: dict[str, Any] = {}

        def make_graph(config: dict, runtime: ServerRuntime) -> str:
            received["config"] = config
            received["runtime"] = runtime
            return "both_graph"

        classify_factory(make_graph, "both_graph")
        config = {"configurable": {"thread_id": "t1"}}
        mock_runtime = Mock()

        result = invoke_factory(make_graph, "both_graph", config, mock_runtime)

        assert result == "both_graph"
        assert received["config"] is config
        assert received["runtime"] is mock_runtime


# ---------------------------------------------------------------------------
# generate_graph
# ---------------------------------------------------------------------------


class TestGenerateGraph:
    """Test generate_graph helper for resolving factory return types."""

    @pytest.mark.asyncio
    async def test_pregel_passthrough(self) -> None:
        """Pregel instances are yielded directly."""
        mock_pregel = Mock(spec=Pregel)

        async with generate_graph(mock_pregel, "test") as result:
            assert result is mock_pregel

    @pytest.mark.asyncio
    async def test_stategraph_passthrough(self) -> None:
        """StateGraph instances are yielded directly (compilation happens upstream)."""
        mock_sg = Mock(spec=StateGraph)

        async with generate_graph(mock_sg, "test") as result:
            assert result is mock_sg

    @pytest.mark.asyncio
    async def test_async_context_manager(self) -> None:
        """Async context managers are entered and yielded."""
        sentinel = object()

        @asynccontextmanager
        async def factory_result() -> AsyncIterator[object]:
            yield sentinel

        async with generate_graph(factory_result(), "test") as result:
            assert result is sentinel

    @pytest.mark.asyncio
    async def test_coroutine(self) -> None:
        """Coroutines are awaited and yielded."""
        sentinel = object()

        async def coro() -> object:
            return sentinel

        async with generate_graph(coro(), "test") as result:
            assert result is sentinel

    @pytest.mark.asyncio
    async def test_plain_object(self) -> None:
        """Plain objects (e.g., already compiled graphs) are yielded as-is."""
        sentinel = {"i_am": "a_graph"}

        async with generate_graph(sentinel, "test") as result:
            assert result is sentinel


# ---------------------------------------------------------------------------
# clear_factory_registry
# ---------------------------------------------------------------------------


class TestClearFactoryRegistry:
    """Test factory registry cleanup."""

    def test_clear_specific(self) -> None:
        _FACTORY_KWARGS["g1"] = lambda c, r: {}
        _FACTORY_KWARGS["g2"] = lambda c, r: {}
        _FACTORY_CONTEXT_TYPES["g1"] = str
        _FACTORY_CONTEXT_TYPES["g2"] = int

        clear_factory_registry("g1")

        assert "g1" not in _FACTORY_KWARGS
        assert "g2" in _FACTORY_KWARGS
        assert "g1" not in _FACTORY_CONTEXT_TYPES
        assert "g2" in _FACTORY_CONTEXT_TYPES

    def test_clear_all(self) -> None:
        _FACTORY_KWARGS["g1"] = lambda c, r: {}
        _FACTORY_KWARGS["g2"] = lambda c, r: {}
        _FACTORY_CONTEXT_TYPES["g1"] = str
        _FACTORY_CONTEXT_TYPES["g2"] = int

        clear_factory_registry()

        assert len(_FACTORY_KWARGS) == 0
        assert len(_FACTORY_CONTEXT_TYPES) == 0

    def test_clear_nonexistent(self) -> None:
        """Clearing a non-existent graph_id is a no-op."""
        clear_factory_registry("nonexistent")

    def test_clear_empty_registry(self) -> None:
        """Clearing an empty registry is a no-op."""
        clear_factory_registry()


# ---------------------------------------------------------------------------
# _extract_context_type
# ---------------------------------------------------------------------------


class TestExtractContextType:
    """Test extraction of T from ServerRuntime[T] annotations."""

    def test_plain_server_runtime_returns_none(self) -> None:
        """Plain ``ServerRuntime`` (no parameterization) → None."""
        assert _extract_context_type(ServerRuntime) is None

    def test_parameterized_server_runtime(self) -> None:
        """``ServerRuntime[MyCtx]`` → MyCtx."""

        class MyCtx:
            pass

        assert _extract_context_type(ServerRuntime[MyCtx]) is MyCtx

    def test_union_with_none(self) -> None:
        """``None | ServerRuntime[MyCtx]`` → MyCtx."""

        class MyCtx:
            pass

        ann = None | ServerRuntime[MyCtx]
        assert _extract_context_type(ann) is MyCtx

    def test_empty_annotation(self) -> None:
        """Empty parameter annotation → None."""
        assert _extract_context_type(inspect.Parameter.empty) is None

    def test_non_runtime_annotation(self) -> None:
        """Non-runtime types → None."""
        assert _extract_context_type(str) is None
        assert _extract_context_type(dict[str, Any]) is None
        assert _extract_context_type(int) is None


# ---------------------------------------------------------------------------
# coerce_context
# ---------------------------------------------------------------------------


class _PydanticCtx(BaseModel):
    """Pydantic model for testing context coercion."""

    name: str
    value: int


@dataclasses.dataclass
class _DataclassCtx:
    """Dataclass for testing context coercion."""

    name: str
    value: int


class TestCoerceContext:
    """Test context coercion from raw dict to T."""

    def test_none_context_passthrough(self) -> None:
        """None context → None, regardless of registered type."""
        _FACTORY_CONTEXT_TYPES["g"] = _PydanticCtx
        assert coerce_context(None, "g") is None

    def test_no_registered_type_passthrough(self) -> None:
        """Unregistered graph_id → raw dict passthrough."""
        ctx = {"name": "test", "value": 42}
        result = coerce_context(ctx, "unregistered")
        assert result is ctx

    def test_registered_none_type_passthrough(self) -> None:
        """Registered with None type (plain ServerRuntime) → raw dict passthrough."""
        _FACTORY_CONTEXT_TYPES["g"] = None
        ctx = {"name": "test", "value": 42}
        result = coerce_context(ctx, "g")
        assert result is ctx

    def test_pydantic_coercion(self) -> None:
        """Pydantic BaseModel → model_validate(dict)."""
        _FACTORY_CONTEXT_TYPES["g"] = _PydanticCtx
        ctx = {"name": "test", "value": 42}

        result = coerce_context(ctx, "g")

        assert isinstance(result, _PydanticCtx)
        assert result.name == "test"
        assert result.value == 42

    def test_dataclass_coercion(self) -> None:
        """Dataclass → T(**dict)."""
        _FACTORY_CONTEXT_TYPES["g"] = _DataclassCtx
        ctx = {"name": "test", "value": 42}

        result = coerce_context(ctx, "g")

        assert isinstance(result, _DataclassCtx)
        assert result.name == "test"
        assert result.value == 42

    def test_pydantic_validation_error_falls_back(self) -> None:
        """Invalid data for Pydantic model → graceful fallback to raw dict."""
        _FACTORY_CONTEXT_TYPES["g"] = _PydanticCtx
        ctx = {"wrong_field": "nope"}

        result = coerce_context(ctx, "g")

        assert result is ctx  # raw dict fallback

    def test_dataclass_missing_field_falls_back(self) -> None:
        """Missing fields for dataclass → graceful fallback to raw dict."""
        _FACTORY_CONTEXT_TYPES["g"] = _DataclassCtx
        ctx = {"name": "test"}  # missing 'value'

        result = coerce_context(ctx, "g")

        assert result is ctx  # raw dict fallback


# ---------------------------------------------------------------------------
# build_server_runtime — context support
# ---------------------------------------------------------------------------


class TestBuildServerRuntimeWithContext:
    """Test that build_server_runtime passes context to _ExecutionRuntime."""

    def test_execution_runtime_with_context(self) -> None:
        """Execution runtime should have context set."""
        ctx = {"key": "value"}
        runtime = build_server_runtime(
            access_context="threads.create_run",
            store=Mock(),
            user=Mock(),
            context=ctx,
        )

        assert isinstance(runtime, _ExecutionRuntime)
        assert runtime.context is ctx

    def test_execution_runtime_without_context(self) -> None:
        """Execution runtime without context → context is None."""
        runtime = build_server_runtime(
            access_context="threads.create_run",
            store=Mock(),
            user=Mock(),
        )

        assert isinstance(runtime, _ExecutionRuntime)
        assert runtime.context is None

    def test_read_runtime_ignores_context(self) -> None:
        """Read runtime should not have a context field."""
        runtime = build_server_runtime(
            access_context="threads.read",
            store=Mock(),
            user=Mock(),
            context={"key": "value"},
        )

        assert isinstance(runtime, _ReadRuntime)
        assert not hasattr(runtime, "context")


# ---------------------------------------------------------------------------
# build_server_runtime — assistant identity
# ---------------------------------------------------------------------------


class TestBuildServerRuntimeAssistantId:
    """The runtime carries the assistant a run executes as."""

    def test_execution_runtime_carries_assistant_id(self) -> None:
        runtime = build_server_runtime(
            access_context="threads.create_run",
            store=Mock(),
            user=Mock(),
            assistant_id="asst-1",
        )

        assert isinstance(runtime, AegraExecutionRuntime)
        assert isinstance(runtime, _ExecutionRuntime)
        assert runtime.assistant_id == "asst-1"

    def test_execution_runtime_without_assistant_id(self) -> None:
        runtime = build_server_runtime(
            access_context="threads.create_run",
            store=Mock(),
            user=Mock(),
        )

        assert runtime.assistant_id is None

    @pytest.mark.parametrize("access_context", ["threads.read", "threads.update", "assistants.read"])
    def test_read_runtime_assistant_id_is_none(self, access_context: str) -> None:
        """Read paths have no run and therefore no assistant."""
        runtime = build_server_runtime(
            access_context=access_context,  # type: ignore[arg-type]
            store=None,
            user=Mock(),
            assistant_id="asst-1",
        )

        assert isinstance(runtime, AegraReadRuntime)
        assert isinstance(runtime, _ReadRuntime)
        assert runtime.assistant_id is None

    def test_runtime_only_factory_reads_assistant_id(self) -> None:
        """The zero-config factory signature can read the assistant without a config param."""
        seen: dict[str, Any] = {}

        def make_graph(runtime: ServerRuntime) -> str:
            seen["assistant_id"] = runtime.assistant_id  # type: ignore[union-attr]
            return "graph"

        classify_factory(make_graph, "assistant_rt_graph")
        runtime = build_server_runtime(
            access_context="threads.create_run",
            store=Mock(),
            user=Mock(),
            assistant_id="asst-1",
        )

        assert invoke_factory(make_graph, "assistant_rt_graph", {}, runtime) == "graph"
        assert seen["assistant_id"] == "asst-1"

    def test_read_path_factory_does_not_raise(self) -> None:
        """A factory reading assistant_id still works for schema extraction."""

        def make_graph(runtime: ServerRuntime) -> str | None:
            return runtime.assistant_id  # type: ignore[union-attr]

        classify_factory(make_graph, "assistant_read_graph")
        runtime = build_server_runtime(access_context="assistants.read", store=None, user=None)

        assert invoke_factory(make_graph, "assistant_read_graph", {}, runtime) is None


# ---------------------------------------------------------------------------
# classify_factory — context type registration
# ---------------------------------------------------------------------------


class TestClassifyFactoryContextType:
    """Test that classify_factory populates _FACTORY_CONTEXT_TYPES."""

    def test_parameterized_runtime_stores_type(self) -> None:
        """ServerRuntime[MyCtx] → stores MyCtx in _FACTORY_CONTEXT_TYPES."""

        class MyCtx:
            pass

        def make_graph(runtime: ServerRuntime[MyCtx]) -> None:
            pass

        classify_factory(make_graph, "typed_graph")

        assert "typed_graph" in _FACTORY_CONTEXT_TYPES
        assert _FACTORY_CONTEXT_TYPES["typed_graph"] is MyCtx

    def test_plain_runtime_stores_none(self) -> None:
        """Plain ServerRuntime → stores None in _FACTORY_CONTEXT_TYPES."""

        def make_graph(runtime: ServerRuntime) -> None:
            pass

        classify_factory(make_graph, "plain_graph")

        assert "plain_graph" in _FACTORY_CONTEXT_TYPES
        assert _FACTORY_CONTEXT_TYPES["plain_graph"] is None

    def test_config_only_factory_no_context_type(self) -> None:
        """Config-only factory → stores None context type."""

        def make_graph(config: dict) -> None:
            pass

        classify_factory(make_graph, "cfg_graph")

        assert "cfg_graph" in _FACTORY_CONTEXT_TYPES
        assert _FACTORY_CONTEXT_TYPES["cfg_graph"] is None

    def test_two_param_with_typed_runtime(self) -> None:
        """2-param factory with ServerRuntime[MyCtx] → stores MyCtx."""

        class MyCtx:
            pass

        def make_graph(config: dict, runtime: ServerRuntime[MyCtx]) -> None:
            pass

        classify_factory(make_graph, "both_graph")

        assert _FACTORY_CONTEXT_TYPES["both_graph"] is MyCtx
