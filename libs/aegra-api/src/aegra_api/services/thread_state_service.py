"""Thread state conversion service"""

from datetime import datetime
from typing import Any

import structlog

from aegra_api.core.serializers import LangGraphSerializer
from aegra_api.models.threads import ThreadCheckpoint, ThreadState

logger = structlog.getLogger(__name__)


class ThreadStateService:
    """Service for converting LangGraph snapshots to ThreadState objects"""

    def __init__(self) -> None:
        self.serializer = LangGraphSerializer()

    def convert_snapshot_to_thread_state(self, snapshot: Any, thread_id: str, subgraphs: bool = False) -> ThreadState:
        """Convert a LangGraph snapshot to ThreadState format"""
        try:
            # Extract basic values
            values = getattr(snapshot, "values", {})
            next_nodes = getattr(snapshot, "next", []) or []
            metadata = getattr(snapshot, "metadata", {}) or {}
            created_at = self._extract_created_at(snapshot)

            # Extract tasks and interrupts using serializer
            tasks = self.serializer.extract_tasks_from_snapshot(snapshot)

            # Recursively serialize tasks' state (which might be subgraphs)
            if subgraphs:
                for task in tasks:
                    if "state" in task and task["state"] is not None:
                        try:
                            task["state"] = self.convert_snapshot_to_thread_state(
                                task["state"], thread_id, subgraphs=True
                            )
                        except Exception as e:
                            logger.error(f"Failed to serialize subgraph state for task {task.get('id')}: {e}")
                            task["state"] = None

            interrupts = self.serializer.extract_interrupts_from_snapshot(snapshot)

            # Create checkpoint objects
            current_checkpoint = self._create_checkpoint(snapshot.config, thread_id)
            parent_checkpoint = (
                self._create_checkpoint(snapshot.parent_config, thread_id) if snapshot.parent_config else None
            )

            # Extract checkpoint IDs for backward compatibility
            checkpoint_id = self._extract_checkpoint_id(snapshot.config)
            parent_checkpoint_id = (
                self._extract_checkpoint_id(snapshot.parent_config) if snapshot.parent_config else None
            )

            return ThreadState(
                values=values,
                next=next_nodes,
                tasks=tasks,
                interrupts=interrupts,
                metadata=metadata,
                created_at=created_at,
                checkpoint=current_checkpoint,
                parent_checkpoint=parent_checkpoint,
                checkpoint_id=checkpoint_id,
                parent_checkpoint_id=parent_checkpoint_id,
            )

        except Exception as e:
            logger.error(
                f"Failed to convert snapshot to thread state: {e} "
                f"(thread_id={thread_id}, snapshot_type={type(snapshot).__name__})"
            )
            raise

    def convert_snapshots_to_thread_states(
        self,
        snapshots: list[Any],
        thread_id: str,
        *,
        subgraphs: bool = False,
    ) -> list[ThreadState]:
        """Convert multiple snapshots to ThreadState objects"""
        thread_states = []

        for i, snapshot in enumerate(snapshots):
            try:
                thread_state = self.convert_snapshot_to_thread_state(snapshot, thread_id, subgraphs=subgraphs)
                thread_states.append(thread_state)
            except Exception as e:
                logger.error(f"Failed to convert snapshot in batch: {e} (thread_id={thread_id}, snapshot_index={i})")
                # Continue with other snapshots rather than failing the entire batch
                continue

        return thread_states

    def _extract_created_at(self, snapshot: Any) -> datetime | None:
        """Extract created_at timestamp from snapshot"""
        created_at = getattr(snapshot, "created_at", None)
        if isinstance(created_at, str):
            try:
                return datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                logger.warning(f"Invalid created_at format: {created_at}")
                return None
        elif isinstance(created_at, datetime):
            return created_at
        return None

    def _create_checkpoint(self, config: Any, thread_id: str) -> ThreadCheckpoint:
        """Create ThreadCheckpoint from config"""
        if not config or not isinstance(config, dict):
            return ThreadCheckpoint(checkpoint_id=None, thread_id=thread_id, checkpoint_ns="")

        configurable = config.get("configurable", {})
        checkpoint_id = configurable.get("checkpoint_id")
        checkpoint_ns = configurable.get("checkpoint_ns", "")
        checkpoint_map = configurable.get("checkpoint_map") or {}

        return ThreadCheckpoint(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns,
            checkpoint_map=checkpoint_map,
        )

    def _extract_checkpoint_id(self, config: Any) -> str | None:
        """Extract checkpoint ID from config for backward compatibility"""
        if not config or not isinstance(config, dict):
            return None

        configurable = config.get("configurable", {})
        checkpoint_id = configurable.get("checkpoint_id")
        return str(checkpoint_id) if checkpoint_id is not None else None
