"""Unit tests for ThreadStateService."""

from datetime import UTC, datetime
from unittest.mock import patch

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


def test_convert_snapshot_to_thread_state_moves_task_config_to_checkpoint() -> None:
    service = ThreadStateService()
    task = make_task(
        state={
            "configurable": {
                "thread_id": "thread-123",
                "checkpoint_ns": "gather-intent-agent:123",
                "checkpoint_id": "checkpoint-subgraph",
                "checkpoint_map": {"": "checkpoint-parent"},
            }
        },
        interrupts=(),
    )
    snapshot = make_snapshot(
        {},
        {"configurable": {"checkpoint_id": "checkpoint-1", "checkpoint_ns": ""}},
        tasks=(task,),
    )

    result = service.convert_snapshot_to_thread_state(snapshot, "thread-123", subgraphs=False)

    task_result = result.tasks[0]
    assert task_result["state"] is None
    assert task_result["checkpoint"] == {
        "thread_id": "thread-123",
        "checkpoint_ns": "gather-intent-agent:123",
        "checkpoint_id": "checkpoint-subgraph",
        "checkpoint_map": {"": "checkpoint-parent"},
    }


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


def test_convert_snapshot_to_thread_state_preserves_checkpoint_map() -> None:
    service = ThreadStateService()
    snapshot = make_snapshot(
        {},
        {
            "configurable": {
                "checkpoint_id": "checkpoint-1",
                "checkpoint_ns": "",
                "checkpoint_map": {"child": "checkpoint-child"},
            }
        },
    )

    result = service.convert_snapshot_to_thread_state(snapshot, "thread-123")

    assert result.checkpoint.checkpoint_map == {"child": "checkpoint-child"}


def test_convert_snapshots_to_thread_states_forwards_subgraphs() -> None:
    service = ThreadStateService()
    child_snapshot = make_snapshot(
        {"foo": "bar"},
        {"configurable": {"checkpoint_id": "checkpoint-child", "checkpoint_ns": "child"}},
    )
    snapshot = make_snapshot(
        {},
        {"configurable": {"checkpoint_id": "checkpoint-parent", "checkpoint_ns": ""}},
        tasks=(make_task(state=child_snapshot, interrupts=()),),
    )

    result = service.convert_snapshots_to_thread_states([snapshot], "thread-123", subgraphs=True)

    nested_state = result[0].tasks[0]["state"]
    assert nested_state.values == {"foo": "bar"}
    assert nested_state.checkpoint.checkpoint_id == "checkpoint-child"


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
