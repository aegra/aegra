"""Unit tests for ThreadStateService."""

from datetime import UTC, datetime
from unittest.mock import patch

from langchain_core.messages import AIMessage

from aegra_api.services.thread_state_service import ThreadStateService
from tests.fixtures.langgraph import make_interrupt, make_snapshot, make_task

TEST_INTERRUPT_ID = "5a2c7e24fdc253a5c72d91717e662021"


def test_convert_snapshot_to_thread_state_basic():
    service = ThreadStateService()
    default_interrupt = make_interrupt(interrupt_id=TEST_INTERRUPT_ID)
    snapshot = make_snapshot(
        {},
        {"configurable": {"checkpoint_id": "checkpoint-1", "checkpoint_ns": ""}},
        created_at="2024-01-01T00:00:00Z",
        next_nodes=("node_1",),
        metadata={
            "step": 0,
            "user_id": "anonymous",
            "user_display_name": "Anonymous User",
        },
        tasks=(make_task(interrupts=(default_interrupt,)),),
        interrupts=(default_interrupt,),
    )

    result = service.convert_snapshot_to_thread_state(snapshot, "thread-123")

    assert result.values == {}
    assert result.next == ["node_1"]
    assert result.metadata["user_id"] == "anonymous"
    assert result.created_at == datetime(2024, 1, 1, tzinfo=UTC)
    assert result.checkpoint.checkpoint_id == "checkpoint-1"
    assert result.checkpoint.checkpoint_ns == ""
    assert result.checkpoint.thread_id == "thread-123"
    assert result.parent_checkpoint is None
    assert result.checkpoint_id == "checkpoint-1"
    assert result.parent_checkpoint_id is None

    assert len(result.tasks) == 1
    task = result.tasks[0]
    assert task["id"] == "task-1"
    assert task["name"] == "node_1"
    assert task["state"] is None
    assert task["interrupts"] == [{"value": "Provide value:", "id": TEST_INTERRUPT_ID}]

    assert result.interrupts == [{"value": "Provide value:", "id": TEST_INTERRUPT_ID}]


def test_convert_snapshot_to_thread_state_subgraphs_recurses():
    service = ThreadStateService()

    child_interrupt = make_interrupt(interrupt_id="child-interrupt")
    child_task = make_task(
        id="child-task",
        name="subgraph_node",
        path=("subgraph_node",),
        interrupts=(child_interrupt,),
        state=None,
    )
    child_snapshot = make_snapshot(
        {"foo": "Initial subgraph value."},
        {
            "configurable": {
                "checkpoint_id": "checkpoint-child",
                "checkpoint_ns": "child",
            }
        },
        created_at="2025-11-10T16:53:21.706336Z",
        next_nodes=("subgraph_node",),
        metadata={"step": 1, "user_id": "anonymous"},
        parent_config={
            "configurable": {
                "checkpoint_id": "checkpoint-parent",
                "checkpoint_ns": "child",
            }
        },
        tasks=(child_task,),
        interrupts=(child_interrupt,),
    )

    top_interrupt = make_interrupt(interrupt_id="top-interrupt")
    top_task = make_task(
        id="top-task",
        interrupts=(top_interrupt,),
        state=child_snapshot,
    )
    snapshot = make_snapshot(
        {},
        {
            "configurable": {
                "checkpoint_id": "checkpoint-1",
                "checkpoint_ns": "",
            }
        },
        created_at="2025-11-10T16:53:21.701708Z",
        next_nodes=("node_1",),
        metadata={"step": 0, "user_id": "anonymous"},
        tasks=(top_task,),
        interrupts=(top_interrupt,),
    )

    result = service.convert_snapshot_to_thread_state(snapshot, "thread-123", subgraphs=True)

    nested_state = result.tasks[0]["state"]
    assert nested_state.values == {"foo": "Initial subgraph value."}
    assert nested_state.next == ["subgraph_node"]
    assert nested_state.checkpoint.checkpoint_id == "checkpoint-child"
    assert nested_state.checkpoint.checkpoint_ns == "child"
    assert nested_state.parent_checkpoint is not None
    assert nested_state.parent_checkpoint.checkpoint_id == "checkpoint-parent"
    assert nested_state.interrupts == [{"value": "Provide value:", "id": "child-interrupt"}]


def test_convert_snapshots_to_thread_states_skips_failures():
    service = ThreadStateService()

    snapshots = ["good", "bad"]

    with patch.object(
        service,
        "convert_snapshot_to_thread_state",
        side_effect=["converted", Exception("boom")],
    ) as mock_convert:
        result = service.convert_snapshots_to_thread_states(snapshots, "thread-123")

    assert result == ["converted"]
    assert mock_convert.call_count == 2


class TestExtractThreadValues:
    def test_maps_interrupts_by_task_id_and_skips_tasks_without_interrupts(self) -> None:
        service = ThreadStateService()
        interrupt = make_interrupt(value="approve?", interrupt_id="int-1")
        snapshot = make_snapshot(
            {"messages": ["hi"]},
            {"configurable": {"checkpoint_id": "cp-1"}},
            tasks=(
                make_task(id="task-1", interrupts=(interrupt,)),
                make_task(id="task-2", interrupts=()),
            ),
        )

        result = service.extract_thread_values(snapshot)

        assert result.values == {"messages": ["hi"]}
        assert result.interrupts == {"task-1": [{"value": "approve?", "id": "int-1"}]}
        assert result.checkpoint_id == "cp-1"

    def test_serializes_message_objects_to_json_safe_dicts(self) -> None:
        service = ThreadStateService()
        snapshot = make_snapshot(
            {"messages": [AIMessage(content="sunny in Paris", id="msg-1")]},
            {"configurable": {"checkpoint_id": "cp-1"}},
            tasks=(),
        )

        result = service.extract_thread_values(snapshot)

        [message] = result.values["messages"]
        assert message["type"] == "ai"
        assert message["content"] == "sunny in Paris"
        assert result.interrupts == {}

    def test_returns_empty_values_when_snapshot_has_no_state(self) -> None:
        service = ThreadStateService()
        snapshot = make_snapshot({}, {"configurable": {}}, tasks=None)

        result = service.extract_thread_values(snapshot)

        assert result.values == {}
        assert result.interrupts == {}
        assert result.checkpoint_id is None

    def test_drops_non_mapping_values(self) -> None:
        service = ThreadStateService()
        snapshot = make_snapshot({}, {"configurable": {}}, tasks=())
        snapshot.values = ["not", "a", "mapping"]

        result = service.extract_thread_values(snapshot)

        assert result.values == {}
