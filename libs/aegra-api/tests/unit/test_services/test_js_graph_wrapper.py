"""Unit tests for the JS graph wrapper.

Tests the JSGraphWrapper class with a mocked JSBridge — no Node.js needed.
Checkpoint stub tests are removed — checkpointing is now handled natively
on the JS side via @langchain/langgraph-checkpoint-postgres.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aegra_api.services.js_graph_wrapper import JSGraphWrapper, _serialize_input


@pytest.fixture
def mock_bridge() -> MagicMock:
    """Create a mock JSBridge."""
    bridge = MagicMock()
    bridge.invoke = AsyncMock(return_value={"state": {"messages": ["response"]}})
    bridge.stream = MagicMock()  # configured per test
    bridge.load_graph = AsyncMock()
    bridge.get_schema = AsyncMock()
    return bridge


@pytest.fixture
def graph_info() -> dict:
    """Sample graph info returned by load_graph."""
    return {
        "graphId": "test_graph",
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {"type": "array"},
            },
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "messages": {"type": "array"},
            },
        },
    }


@pytest.fixture
def wrapper(mock_bridge: MagicMock, graph_info: dict) -> JSGraphWrapper:
    """Create a JSGraphWrapper instance."""
    return JSGraphWrapper(mock_bridge, "test_graph", graph_info)


class TestJSGraphWrapperSchema:
    """Tests for schema-related methods."""

    def test_input_schema(self, wrapper: JSGraphWrapper, graph_info: dict) -> None:
        """get_input_jsonschema returns the stored input schema."""
        assert wrapper.get_input_jsonschema() == graph_info["inputSchema"]

    def test_input_schema_returns_copy(self, wrapper: JSGraphWrapper) -> None:
        """get_input_jsonschema returns a deep copy, not a reference."""
        schema = wrapper.get_input_jsonschema()
        schema["modified"] = True
        assert "modified" not in wrapper.get_input_jsonschema()

    def test_output_schema(self, wrapper: JSGraphWrapper, graph_info: dict) -> None:
        assert wrapper.get_output_jsonschema() == graph_info["outputSchema"]

    def test_output_schema_returns_copy(self, wrapper: JSGraphWrapper) -> None:
        schema = wrapper.get_output_jsonschema()
        schema["modified"] = True
        assert "modified" not in wrapper.get_output_jsonschema()

    def test_context_schema_empty(self, wrapper: JSGraphWrapper) -> None:
        """JS graphs return empty context schema."""
        assert wrapper.get_context_jsonschema() == {}

    def test_graph_state_schema(self, wrapper: JSGraphWrapper, graph_info: dict) -> None:
        """Graph state schema matches the output schema."""
        assert wrapper.get_graph_state_jsonschema() == graph_info["outputSchema"]

    def test_missing_schemas_default_empty(self, mock_bridge: MagicMock) -> None:
        """Missing schemas in graph_info default to empty dicts."""
        w = JSGraphWrapper(mock_bridge, "g", {})
        assert w.get_input_jsonschema() == {}
        assert w.get_output_jsonschema() == {}


class TestJSGraphWrapperCopy:
    """Tests for the copy/clone mechanism."""

    def test_copy_creates_new_instance(self, wrapper: JSGraphWrapper) -> None:
        """copy() returns a new wrapper with same bridge."""
        clone = wrapper.copy()
        assert clone is not wrapper
        assert clone._bridge is wrapper._bridge
        assert clone._graph_id == wrapper._graph_id

    def test_copy_preserves_graph_info(self, wrapper: JSGraphWrapper) -> None:
        clone = wrapper.copy()
        assert clone._graph_info is wrapper._graph_info

    def test_copy_with_checkpointer(self, wrapper: JSGraphWrapper) -> None:
        """copy(update=...) applies attribute updates."""
        mock_checkpointer = MagicMock()
        clone = wrapper.copy(update={"checkpointer": mock_checkpointer})
        assert clone.checkpointer is mock_checkpointer
        assert wrapper.checkpointer is None  # original unchanged

    def test_copy_with_store(self, wrapper: JSGraphWrapper) -> None:
        mock_store = MagicMock()
        clone = wrapper.copy(update={"store": mock_store})
        assert clone.store is mock_store
        assert wrapper.store is None

    def test_copy_with_config(self, wrapper: JSGraphWrapper) -> None:
        wrapper.config = {"configurable": {"thread_id": "base"}}
        clone = wrapper.copy()
        assert clone.config == wrapper.config
        clone.config["configurable"]["thread_id"] = "changed"
        assert wrapper.config["configurable"]["thread_id"] == "base"

    def test_copy_ignores_unknown_attrs(self, wrapper: JSGraphWrapper) -> None:
        """Unknown keys in update are silently ignored."""
        clone = wrapper.copy(update={"nonexistent_attr": "value"})
        assert not hasattr(clone, "nonexistent_attr")


class TestSerializeInput:
    """Tests for the _serialize_input helper."""

    def test_dict_passthrough(self) -> None:
        """Regular dicts pass through unchanged."""
        data = {"messages": [{"role": "user", "content": "hi"}]}
        assert _serialize_input(data) == data

    def test_non_dict_wrapped(self) -> None:
        """Non-dict inputs are wrapped in an 'input' key."""
        assert _serialize_input("hello") == {"input": "hello"}

    def test_command_serialized(self) -> None:
        """LangGraph Command objects are serialized to wire format."""
        from langgraph.types import Command

        cmd = Command(resume="yes")
        result = _serialize_input(cmd)
        assert "__command__" in result
        assert result["__command__"]["resume"] == "yes"

    def test_command_with_goto(self) -> None:
        """Command with goto is serialized correctly."""
        from langgraph.types import Command

        cmd = Command(goto=["node_a", "node_b"])
        result = _serialize_input(cmd)
        assert result["__command__"]["goto"] == ["node_a", "node_b"]

    def test_command_with_update(self) -> None:
        """Command with update is serialized correctly."""
        from langgraph.types import Command

        cmd = Command(update={"key": "value"})
        result = _serialize_input(cmd)
        assert result["__command__"]["update"] == {"key": "value"}


class TestJSGraphWrapperInvoke:
    """Tests for ainvoke execution."""

    async def test_ainvoke(self, wrapper: JSGraphWrapper, mock_bridge: MagicMock) -> None:
        """ainvoke delegates to bridge.invoke and returns final state."""
        result = await wrapper.ainvoke(
            {"messages": [{"role": "user", "content": "hi"}]},
            {"configurable": {"thread_id": "t1"}},
        )
        mock_bridge.invoke.assert_called_once()
        assert result == {"messages": ["response"]}

    async def test_ainvoke_without_config(self, wrapper: JSGraphWrapper, mock_bridge: MagicMock) -> None:
        """ainvoke works without config."""
        await wrapper.ainvoke({"messages": []})
        mock_bridge.invoke.assert_called_once()

    async def test_ainvoke_passes_graph_id(self, wrapper: JSGraphWrapper, mock_bridge: MagicMock) -> None:
        """ainvoke sends the correct graph_id to the bridge."""
        await wrapper.ainvoke({"messages": []})
        call_args = mock_bridge.invoke.call_args
        assert call_args[0][0] == "test_graph"

    async def test_ainvoke_extracts_state(self, wrapper: JSGraphWrapper, mock_bridge: MagicMock) -> None:
        """ainvoke extracts 'state' key from bridge result."""
        mock_bridge.invoke.return_value = {
            "state": {"messages": ["extracted"]},
        }
        result = await wrapper.ainvoke({"messages": []})
        assert result == {"messages": ["extracted"]}

    async def test_ainvoke_falls_back_to_raw_result(self, wrapper: JSGraphWrapper, mock_bridge: MagicMock) -> None:
        """When bridge result has no 'state', returns the entire result."""
        mock_bridge.invoke.return_value = {"messages": ["raw"]}
        result = await wrapper.ainvoke({"messages": []})
        assert result == {"messages": ["raw"]}

    async def test_ainvoke_does_not_touch_checkpoints(self, wrapper: JSGraphWrapper, mock_bridge: MagicMock) -> None:
        """ainvoke does not load or save checkpoints on the Python side."""
        mock_cp = MagicMock()
        mock_cp.aget_tuple = AsyncMock()
        mock_cp.aput = AsyncMock()
        wrapper.checkpointer = mock_cp

        await wrapper.ainvoke(
            {"messages": []},
            {"configurable": {"thread_id": "t1"}},
        )

        mock_cp.aget_tuple.assert_not_called()
        mock_cp.aput.assert_not_called()


class TestJSGraphWrapperConfig:
    """Tests for config merging."""

    async def test_merge_config_empty(self, wrapper: JSGraphWrapper) -> None:
        """Merging with None config returns base config."""
        merged = wrapper._merge_config(None)
        assert merged == {}

    async def test_merge_config_with_instance_config(self, wrapper: JSGraphWrapper) -> None:
        """Instance config is used as base."""
        wrapper.config = {"configurable": {"thread_id": "base"}}
        merged = wrapper._merge_config(None)
        assert merged["configurable"]["thread_id"] == "base"

    async def test_merge_config_combines(self, wrapper: JSGraphWrapper) -> None:
        """Config merging combines instance and call configs."""
        wrapper.config = {"configurable": {"thread_id": "base"}}
        merged = wrapper._merge_config({"configurable": {"run_id": "r1"}})
        assert merged["configurable"]["thread_id"] == "base"
        assert merged["configurable"]["run_id"] == "r1"

    async def test_merge_config_call_overrides(self, wrapper: JSGraphWrapper) -> None:
        """Call config overrides instance config for same keys."""
        wrapper.config = {"configurable": {"thread_id": "base"}}
        merged = wrapper._merge_config({"configurable": {"thread_id": "override"}})
        assert merged["configurable"]["thread_id"] == "override"


class TestJSGraphWrapperMisc:
    """Tests for repr, name, and other basics."""

    def test_repr(self, wrapper: JSGraphWrapper) -> None:
        assert "test_graph" in repr(wrapper)

    def test_name(self, wrapper: JSGraphWrapper) -> None:
        assert wrapper.name == "test_graph"

    def test_name_falls_back_to_graph_id(self, mock_bridge: MagicMock) -> None:
        """When graphId missing from info, name falls back to constructor arg."""
        w = JSGraphWrapper(mock_bridge, "fallback_id", {})
        assert w.name == "fallback_id"

    def test_initial_checkpointer_none(self, wrapper: JSGraphWrapper) -> None:
        assert wrapper.checkpointer is None

    def test_initial_store_none(self, wrapper: JSGraphWrapper) -> None:
        assert wrapper.store is None

    def test_initial_config_empty(self, wrapper: JSGraphWrapper) -> None:
        assert wrapper.config == {}


class TestJSGraphWrapperStream:
    """Tests for astream execution."""

    async def test_astream_yields_mode_data_tuples(self, wrapper: JSGraphWrapper, mock_bridge: MagicMock) -> None:
        """astream yields (mode, data) tuples from the bridge."""

        async def mock_stream(*args: object, **kwargs: object) -> None:
            for event in [
                {"mode": "values", "data": {"messages": ["hello"]}},
                {"mode": "updates", "data": {"node": "chatbot"}},
            ]:
                yield event

        mock_bridge.stream = MagicMock(return_value=mock_stream())

        results = []
        async for item in wrapper.astream(
            {"messages": [{"role": "user", "content": "hi"}]},
            {"configurable": {"thread_id": "t1"}},
        ):
            results.append(item)

        assert len(results) == 2
        assert results[0] == ("values", {"messages": ["hello"]})
        assert results[1] == ("updates", {"node": "chatbot"})

    async def test_astream_does_not_touch_checkpoints(self, wrapper: JSGraphWrapper, mock_bridge: MagicMock) -> None:
        """astream does not load or save checkpoints on the Python side."""

        async def mock_stream(*args: object, **kwargs: object) -> None:
            yield {"mode": "values", "data": {"messages": ["final"]}}

        mock_bridge.stream = MagicMock(return_value=mock_stream())

        mock_cp = MagicMock()
        mock_cp.aget_tuple = AsyncMock()
        mock_cp.aput = AsyncMock()
        wrapper.checkpointer = mock_cp

        results = []
        async for item in wrapper.astream(
            {"messages": []},
            {"configurable": {"thread_id": "t1"}},
        ):
            results.append(item)

        mock_cp.aget_tuple.assert_not_called()
        mock_cp.aput.assert_not_called()


class TestJSGraphWrapperStreamEvents:
    """Tests for astream_events execution."""

    async def test_astream_events_wraps_in_v2_format(self, wrapper: JSGraphWrapper, mock_bridge: MagicMock) -> None:
        """astream_events wraps output in LangChain v2 event format."""

        async def mock_stream(*args: object, **kwargs: object) -> None:
            yield {"mode": "values", "data": {"messages": ["hello"]}}

        mock_bridge.stream = MagicMock(return_value=mock_stream())

        results = []
        async for event in wrapper.astream_events(
            {"messages": []},
            {"configurable": {"thread_id": "t1", "run_id": "run-123"}},
        ):
            results.append(event)

        assert len(results) == 1
        assert results[0]["event"] == "on_chain_stream"
        assert results[0]["run_id"] == "run-123"
        assert results[0]["data"]["chunk"] == ("values", {"messages": ["hello"]})
        assert results[0]["tags"] == []

    async def test_astream_events_multiple_events(self, wrapper: JSGraphWrapper, mock_bridge: MagicMock) -> None:
        """astream_events yields one event per stream chunk."""

        async def mock_stream(*args: object, **kwargs: object) -> None:
            yield {"mode": "values", "data": {"messages": ["a"]}}
            yield {"mode": "updates", "data": {"node": "chatbot"}}
            yield {"mode": "values", "data": {"messages": ["a", "b"]}}

        mock_bridge.stream = MagicMock(return_value=mock_stream())

        results = []
        async for event in wrapper.astream_events(
            {"messages": []},
            {"configurable": {"thread_id": "t1"}},
        ):
            results.append(event)

        assert len(results) == 3
        assert results[0]["data"]["chunk"][0] == "values"
        assert results[1]["data"]["chunk"][0] == "updates"
        assert results[2]["data"]["chunk"][0] == "values"
