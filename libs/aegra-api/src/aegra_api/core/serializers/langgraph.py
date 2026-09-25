"""LangGraph-specific serialization"""

import json
from typing import Any

import structlog

from aegra_api.core.serializers.base import SerializationError, Serializer
from aegra_api.core.serializers.general import GeneralSerializer

logger = structlog.getLogger(__name__)


class LangGraphSerializer(Serializer):
    """Handles serialization of LangGraph objects (tasks, interrupts, snapshots)"""

    def __init__(self):
        self.general_serializer = GeneralSerializer()

    def serialize(self, obj: Any) -> Any:
        """Main serialization entry point"""
        return json.loads(json.dumps(obj, default=self.general_serializer.serialize))

    def serialize_task(self, task: Any) -> dict[str, Any]:
        """Serialize a LangGraph task to ThreadTask format"""
        try:
            if hasattr(task, "id") and hasattr(task, "name"):
                # Proper task object
                task_dict = {
                    "id": getattr(task, "id", ""),
                    "name": getattr(task, "name", ""),
                    "error": getattr(task, "error", None),
                    "interrupts": [],
                    "checkpoint": self._checkpoint_from_config(getattr(task, "state", None)),
                    "state": self._state_from_task_state(getattr(task, "state", None)),
                    "result": getattr(task, "result", None),
                }

                # Handle task interrupts
                if hasattr(task, "interrupts") and task.interrupts:
                    task_dict["interrupts"] = self.serialize(task.interrupts)

                return task_dict
            else:
                # Raw task data - serialize as-is but safely
                serialized_task = self.serialize(task)
                if isinstance(serialized_task, dict):
                    return serialized_task
                else:
                    raise SerializationError(
                        f"Task serialization resulted in non-dict: {type(serialized_task)}",
                        task.__class__.__name__,
                    )
        except Exception as e:
            if isinstance(e, SerializationError):
                raise
            raise SerializationError(f"Failed to serialize task: {str(e)}", task.__class__.__name__, e) from e

    def _checkpoint_from_config(self, state: Any) -> dict[str, Any] | None:
        if not self._is_checkpoint_config(state):
            return None

        configurable = state.get("configurable", {})
        return {
            "thread_id": configurable.get("thread_id"),
            "checkpoint_ns": configurable.get("checkpoint_ns", ""),
            "checkpoint_id": configurable.get("checkpoint_id"),
            "checkpoint_map": configurable.get("checkpoint_map") or {},
        }

    def _state_from_task_state(self, state: Any) -> Any:
        if self._is_checkpoint_config(state):
            return None

        return state

    def _is_checkpoint_config(self, state: Any) -> bool:
        if not isinstance(state, dict):
            return False

        configurable = state.get("configurable")
        if not isinstance(configurable, dict):
            return False

        return any(key in configurable for key in ("thread_id", "checkpoint_ns", "checkpoint_id", "checkpoint_map"))

    def serialize_interrupt(self, interrupt: Any) -> dict[str, Any]:
        """Serialize a LangGraph interrupt"""
        try:
            return self.serialize(interrupt)
        except Exception as e:
            raise SerializationError(
                f"Failed to serialize interrupt: {str(e)}",
                interrupt.__class__.__name__,
                e,
            ) from e

    def extract_tasks_from_snapshot(self, snapshot: Any) -> list[dict[str, Any]]:
        """Extract and serialize tasks from a snapshot"""
        tasks = []

        if not (hasattr(snapshot, "tasks") and snapshot.tasks):
            return tasks

        for task in snapshot.tasks:
            try:
                serialized_task = self.serialize_task(task)
                tasks.append(serialized_task)
            except SerializationError as e:
                logger.warning(
                    f"Task serialization failed, skipping task: {e} "
                    f"(task_type={type(task).__name__}, task_id={getattr(task, 'id', 'unknown')})"
                )
                continue

        return tasks

    def extract_interrupts_from_snapshot(self, snapshot: Any) -> list[dict[str, Any]]:
        """Extract and serialize interrupts from a snapshot"""
        interrupts = []
        if hasattr(snapshot, "interrupts") and snapshot.interrupts:
            try:
                interrupts = self.serialize(snapshot.interrupts)
                if interrupts:
                    return interrupts
            except Exception as e:
                logger.warning(
                    f"Snapshot interrupt serialization failed: {e} (snapshot_type={type(snapshot).__name__})"
                )
        return interrupts if isinstance(interrupts, list) else []
