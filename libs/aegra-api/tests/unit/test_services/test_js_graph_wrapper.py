"""Unit tests for the JS graph wrapper.

Tests the JSGraphWrapper class with a mocked JSBridge — no Node.js needed.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aegra_api.services.js_graph_wrapper import JSGraphWrapper


@pytest.fixture
def mock_bridge():
    """Create a mock JSBridge."""
    bridge = MagicMock()
    bridge.invoke = AsyncMock(return_value={"state": {"messages": ["response"]}})
    bridge.stream = MagicMock()  # configured per test
    bridge.load_graph = AsyncMock()
    bridge.get_schema = AsyncMock()
    return bridge


@pytest.fixture
def graph_info():
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
def wrapper(mock_bridge, graph_info):
    """Create a JSGraphWrapper instance."""
    return JSGraphWrapper(mock_bridge, "test_graph", graph_info)


class TestJSGraphWrapperSchema:
    """Tests for schema-related methods."""

    def test_input_schema(self, wrapper, graph_info):
        """get_input_jsonschema returns the stored input schema."""
        assert wrapper.get_input_jsonschema() == graph_info["inputSchema"]

    def test_input_schema_returns_copy(self, wrapper):
        """get_input_jsonschema returns a deep copy, not a reference."""
        schema = wrapper.get_input_jsonschema()
        schema["modified"] = True
        assert "modified" not in wrapper.get_input_jsonschema()

    def test_output_schema(self, wrapper, graph_info):
        assert wrapper.get_output_jsonschema() == graph_info["outputSchema"]

    def test_output_schema_returns_copy(self, wrapper):
        schema = wrapper.get_output_jsonschema()
        schema["modified"] = True
        assert "modified" not in wrapper.get_output_jsonschema()

    def test_context_schema_empty(self, wrapper):
        """JS graphs return empty context schema."""
        assert wrapper.get_context_jsonschema() == {}

    def test_graph_state_schema(self, wrapper, graph_info):
        """Graph state schema matches the output schema."""
        assert wrapper.get_graph_state_jsonschema() == graph_info["outputSchema"]

    def test_missing_schemas_default_empty(self, mock_bridge):
        """Missing schemas in graph_info default to empty dicts."""
        w = JSGraphWrapper(mock_bridge, "g", {})
        assert w.get_input_jsonschema() == {}
        assert w.get_output_jsonschema() == {}


class TestJSGraphWrapperCopy:
    """Tests for the copy/clone mechanism."""

    def test_copy_creates_new_instance(self, wrapper):
        """copy() returns a new wrapper with same bridge."""
        clone = wrapper.copy()
        assert clone is not wrapper
        assert clone._bridge is wrapper._bridge
        assert clone._graph_id == wrapper._graph_id

    def test_copy_preserves_graph_info(self, wrapper):
        clone = wrapper.copy()
        assert clone._graph_info is wrapper._graph_info

    def test_copy_with_checkpointer(self, wrapper):
        """copy(update=...) applies attribute updates."""
        mock_checkpointer = MagicMock()
        clone = wrapper.copy(update={"checkpointer": mock_checkpointer})
        assert clone.checkpointer is mock_checkpointer
        assert wrapper.checkpointer is None  # original unchanged

    def test_copy_with_store(self, wrapper):
        mock_store = MagicMock()
        clone = wrapper.copy(update={"store": mock_store})
        assert clone.store is mock_store
        assert wrapper.store is None

    def test_copy_with_config(self, wrapper):
        wrapper.config = {"configurable": {"thread_id": "base"}}
        clone = wrapper.copy()
        assert clone.config == wrapper.config
        # Deep copy — mutating clone config doesn't affect original
        clone.config["configurable"]["thread_id"] = "changed"
        assert wrapper.config["configurable"]["thread_id"] == "base"

    def test_copy_ignores_unknown_attrs(self, wrapper):
        """Unknown keys in update are silently ignored."""
        clone = wrapper.copy(update={"nonexistent_attr": "value"})
        assert not hasattr(clone, "nonexistent_attr")


class TestJSGraphWrapperInvoke:
    """Tests for ainvoke execution."""

    async def test_ainvoke(self, wrapper, mock_bridge):
        """ainvoke delegates to bridge.invoke and returns final state."""
        result = await wrapper.ainvoke(
            {"messages": [{"role": "user", "content": "hi"}]},
            {"configurable": {"thread_id": "t1"}},
        )
        mock_bridge.invoke.assert_called_once()
        assert result == {"messages": ["response"]}

    async def test_ainvoke_without_config(self, wrapper, mock_bridge):
        """ainvoke works without config."""
        await wrapper.ainvoke({"messages": []})
        mock_bridge.invoke.assert_called_once()

    async def test_ainvoke_passes_graph_id(self, wrapper, mock_bridge):
        """ainvoke sends the correct graph_id to the bridge."""
        await wrapper.ainvoke({"messages": []})
        call_args = mock_bridge.invoke.call_args
        assert call_args[0][0] == "test_graph"

    async def test_ainvoke_extracts_state(self, wrapper, mock_bridge):
        """ainvoke extracts 'state' key from bridge result."""
        mock_bridge.invoke.return_value = {
            "state": {"messages": ["extracted"]},
        }
        result = await wrapper.ainvoke({"messages": []})
        assert result == {"messages": ["extracted"]}

    async def test_ainvoke_falls_back_to_raw_result(self, wrapper, mock_bridge):
        """When bridge result has no 'state', returns the entire result."""
        mock_bridge.invoke.return_value = {"messages": ["raw"]}
        result = await wrapper.ainvoke({"messages": []})
        assert result == {"messages": ["raw"]}


class TestJSGraphWrapperConfig:
    """Tests for config merging."""

    async def test_merge_config_empty(self, wrapper):
        """Merging with None config returns base config."""
        merged = wrapper._merge_config(None)
        assert merged == {}

    async def test_merge_config_with_instance_config(self, wrapper):
        """Instance config is used as base."""
        wrapper.config = {"configurable": {"thread_id": "base"}}
        merged = wrapper._merge_config(None)
        assert merged["configurable"]["thread_id"] == "base"

    async def test_merge_config_combines(self, wrapper):
        """Config merging combines instance and call configs."""
        wrapper.config = {"configurable": {"thread_id": "base"}}
        merged = wrapper._merge_config({"configurable": {"run_id": "r1"}})
        assert merged["configurable"]["thread_id"] == "base"
        assert merged["configurable"]["run_id"] == "r1"

    async def test_merge_config_call_overrides(self, wrapper):
        """Call config overrides instance config for same keys."""
        wrapper.config = {"configurable": {"thread_id": "base"}}
        merged = wrapper._merge_config({"configurable": {"thread_id": "override"}})
        assert merged["configurable"]["thread_id"] == "override"


class TestJSGraphWrapperCheckpoint:
    """Tests for checkpoint helpers."""

    async def test_load_checkpoint_no_checkpointer(self, wrapper):
        """No checkpoint loaded when checkpointer is None."""
        result = await wrapper._load_checkpoint({"configurable": {"thread_id": "t1"}})
        assert result is None

    async def test_load_checkpoint_no_thread_id(self, wrapper):
        """No checkpoint loaded when thread_id is missing."""
        wrapper.checkpointer = MagicMock()
        result = await wrapper._load_checkpoint({"configurable": {}})
        assert result is None

    async def test_load_checkpoint_returns_channel_values(self, wrapper):
        """Checkpoint with channel_values returns them."""
        mock_cp = MagicMock()
        mock_cp.checkpoint = {"channel_values": {"messages": ["saved"]}}
        wrapper.checkpointer = MagicMock()
        wrapper.checkpointer.aget_tuple = AsyncMock(return_value=mock_cp)

        result = await wrapper._load_checkpoint({"configurable": {"thread_id": "t1"}})
        assert result == {"messages": ["saved"]}

    async def test_load_checkpoint_empty_channel_values(self, wrapper):
        """Empty channel_values returns None."""
        mock_cp = MagicMock()
        mock_cp.checkpoint = {"channel_values": {}}
        wrapper.checkpointer = MagicMock()
        wrapper.checkpointer.aget_tuple = AsyncMock(return_value=mock_cp)

        result = await wrapper._load_checkpoint({"configurable": {"thread_id": "t1"}})
        assert result is None

    async def test_load_checkpoint_exception_returns_none(self, wrapper):
        """Exceptions during checkpoint loading return None gracefully."""
        wrapper.checkpointer = MagicMock()
        wrapper.checkpointer.aget_tuple = AsyncMock(side_effect=RuntimeError("db error"))

        result = await wrapper._load_checkpoint({"configurable": {"thread_id": "t1"}})
        assert result is None

    async def test_save_checkpoint_no_checkpointer(self, wrapper):
        """Save is a no-op when checkpointer is None."""
        # Should not raise
        await wrapper._save_checkpoint(
            {"configurable": {"thread_id": "t1"}},
            {"messages": ["test"]},
        )

    async def test_save_checkpoint_no_thread_id(self, wrapper):
        """Save is a no-op when thread_id is missing."""
        wrapper.checkpointer = MagicMock()
        wrapper.checkpointer.aput = AsyncMock()
        await wrapper._save_checkpoint({"configurable": {}}, {"messages": []})
        wrapper.checkpointer.aput.assert_not_called()

    async def test_save_checkpoint_calls_aput(self, wrapper):
        """Save delegates to checkpointer.aput with correct structure."""
        wrapper.checkpointer = MagicMock()
        wrapper.checkpointer.aput = AsyncMock()

        state = {"messages": ["saved"]}
        await wrapper._save_checkpoint({"configurable": {"thread_id": "t1"}}, state)

        wrapper.checkpointer.aput.assert_called_once()
        call_args = wrapper.checkpointer.aput.call_args[0]
        checkpoint = call_args[1]
        assert checkpoint["channel_values"] == state
        assert checkpoint["v"] == 1
        assert checkpoint["id"] != ""
        assert checkpoint["ts"] != ""

    async def test_save_checkpoint_exception_no_raise(self, wrapper):
        """Exceptions during checkpoint saving don't propagate."""
        wrapper.checkpointer = MagicMock()
        wrapper.checkpointer.aput = AsyncMock(side_effect=RuntimeError("db fail"))

        # Should not raise
        await wrapper._save_checkpoint(
            {"configurable": {"thread_id": "t1"}},
            {"messages": ["test"]},
        )


class TestJSGraphWrapperMisc:
    """Tests for repr, name, and other basics."""

    def test_repr(self, wrapper):
        assert "test_graph" in repr(wrapper)

    def test_name(self, wrapper):
        assert wrapper.name == "test_graph"

    def test_name_falls_back_to_graph_id(self, mock_bridge):
        """When graphId missing from info, name falls back to constructor arg."""
        w = JSGraphWrapper(mock_bridge, "fallback_id", {})
        assert w.name == "fallback_id"

    def test_initial_checkpointer_none(self, wrapper):
        assert wrapper.checkpointer is None

    def test_initial_store_none(self, wrapper):
        assert wrapper.store is None

    def test_initial_config_empty(self, wrapper):
        assert wrapper.config == {}


class TestJSGraphWrapperStream:
    """Tests for astream execution."""

    async def test_astream_yields_mode_data_tuples(self, wrapper, mock_bridge):
        """astream yields (mode, data) tuples from the bridge."""

        async def mock_stream(*args, **kwargs):
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

    async def test_astream_saves_final_checkpoint(self, wrapper, mock_bridge):
        """astream saves the final values state as a checkpoint."""

        async def mock_stream(*args, **kwargs):
            yield {"mode": "values", "data": {"messages": ["final"]}}

        mock_bridge.stream = MagicMock(return_value=mock_stream())

        mock_cp = MagicMock()
        mock_cp.aput = AsyncMock()
        wrapper.checkpointer = mock_cp

        results = []
        async for item in wrapper.astream(
            {"messages": []},
            {"configurable": {"thread_id": "t1"}},
        ):
            results.append(item)

        mock_cp.aput.assert_called_once()
        call_args = mock_cp.aput.call_args[0]
        checkpoint = call_args[1]
        assert checkpoint["channel_values"] == {"messages": ["final"]}

    async def test_astream_no_checkpoint_without_values(self, wrapper, mock_bridge):
        """astream does not save a checkpoint if no values events are received."""

        async def mock_stream(*args, **kwargs):
            yield {"mode": "updates", "data": {"node": "chatbot"}}

        mock_bridge.stream = MagicMock(return_value=mock_stream())

        mock_cp = MagicMock()
        mock_cp.aput = AsyncMock()
        wrapper.checkpointer = mock_cp

        results = []
        async for item in wrapper.astream(
            {"messages": []},
            {"configurable": {"thread_id": "t1"}},
        ):
            results.append(item)

        mock_cp.aput.assert_not_called()


class TestJSGraphWrapperStreamEvents:
    """Tests for astream_events execution."""

    async def test_astream_events_wraps_in_v2_format(self, wrapper, mock_bridge):
        """astream_events wraps output in LangChain v2 event format."""

        async def mock_stream(*args, **kwargs):
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

    async def test_astream_events_multiple_events(self, wrapper, mock_bridge):
        """astream_events yields one event per stream chunk."""

        async def mock_stream(*args, **kwargs):
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
