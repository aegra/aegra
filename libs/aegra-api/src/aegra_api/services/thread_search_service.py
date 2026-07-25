"""Enrich thread search results with select projection, values join, and extract."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any

import structlog

from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.models.auth import User
from aegra_api.models.threads import (
    CHECKPOINT_SELECT_FIELDS,
    Thread,
    ThreadSearchRequest,
)
from aegra_api.services.thread_state_service import ThreadStateService
from aegra_api.utils.json_path import resolve_json_path, sources_needed_by_extract

logger = structlog.getLogger(__name__)

_VALUES_SOFT_CAP_BYTES = 256 * 1024
_STATE_FETCH_CONCURRENCY = 8
# Bound per-thread checkpoint join so a stuck graph load cannot stall search.
_STATE_FETCH_TIMEOUT_SECS = 15.0

_BASE_THREAD_FIELD_ORDER: tuple[str, ...] = (
    "thread_id",
    "status",
    "metadata",
    "user_id",
    "created_at",
    "updated_at",
)
_BASE_THREAD_FIELDS: frozenset[str] = frozenset(_BASE_THREAD_FIELD_ORDER)


class ThreadSearchService:
    """Build projected / enriched thread search responses."""

    def __init__(self, *, state_service: ThreadStateService | None = None) -> None:
        self._state_service = state_service or ThreadStateService()

    async def build_response(
        self,
        rows: Sequence[ThreadORM],
        request: ThreadSearchRequest,
        *,
        user: User,
        serialize_thread: Callable[[ThreadORM], Thread],
    ) -> list[dict[str, Any]]:
        """Return search results as plain dicts (supports sparse ``select`` projection)."""
        if request.select is None and not request.extract:
            return [serialize_thread(row).model_dump(mode="json") for row in rows]

        select_fields = set(request.select) if request.select is not None else set(_BASE_THREAD_FIELDS)
        extract = request.extract or {}
        extract_sources = sources_needed_by_extract(extract) if extract else set()

        need_checkpoint = bool(select_fields & CHECKPOINT_SELECT_FIELDS) or bool(
            extract_sources & {"values", "config"}
        )

        state_by_id: dict[str, dict[str, Any]] = {}
        if need_checkpoint and rows:
            state_by_id = await self._fetch_states_for_threads(rows, user=user)

        # When select is omitted but extract is set, keep the default thin field set.
        project_fields: list[str] = (
            list(request.select) if request.select is not None else list(_BASE_THREAD_FIELD_ORDER)
        )

        results: list[dict[str, Any]] = []
        for row in rows:
            base = serialize_thread(row).model_dump(mode="json")
            thread_id = base["thread_id"]
            state_blob = state_by_id.get(
                thread_id,
                {"values": {}, "interrupts": [], "config": {}},
            )

            item: dict[str, Any] = {}
            for field in project_fields:
                if field in _BASE_THREAD_FIELDS:
                    item[field] = base[field]
                elif field == "values":
                    item["values"] = state_blob["values"]
                elif field == "interrupts":
                    item["interrupts"] = state_blob["interrupts"]
                elif field == "config":
                    item["config"] = state_blob["config"]

            if extract:
                sources: dict[str, Any] = {
                    "values": state_blob["values"],
                    "metadata": base.get("metadata") or {},
                    "config": state_blob["config"],
                }
                extracted: dict[str, Any] = {}
                for alias, path in extract.items():
                    root_name = path.split(".", 1)[0]
                    extracted[alias] = resolve_json_path(sources.get(root_name), path)
                item["extracted"] = extracted

            results.append(item)
        return results

    async def _fetch_states_for_threads(
        self,
        rows: Sequence[ThreadORM],
        *,
        user: User,
    ) -> dict[str, dict[str, Any]]:
        sem = asyncio.Semaphore(_STATE_FETCH_CONCURRENCY)

        async def one(row: ThreadORM) -> tuple[str, dict[str, Any]]:
            thread_id = str(row.thread_id)
            async with sem:
                return thread_id, await self._load_thread_state_fields(row, user=user)

        pairs = await asyncio.gather(*(one(row) for row in rows))
        return dict(pairs)

    async def _load_thread_state_fields(self, row: ThreadORM, *, user: User) -> dict[str, Any]:
        empty: dict[str, Any] = {"values": {}, "interrupts": [], "config": {}}
        metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
        graph_id = metadata.get("graph_id")
        if not graph_id:
            return empty

        from aegra_api.services.langgraph_service import (
            create_thread_config,
            get_langgraph_service,
        )

        thread_id = str(row.thread_id)
        langgraph_service = get_langgraph_service()
        config = create_thread_config(thread_id, user)

        async def _load_snapshot() -> Any:
            async with langgraph_service.get_graph(
                graph_id,
                config=config,
                access_context="threads.read",
                user=user,
            ) as agent:
                agent = agent.with_config(config)
                return await agent.aget_state(config, subgraphs=False)

        try:
            snapshot = await asyncio.wait_for(_load_snapshot(), timeout=_STATE_FETCH_TIMEOUT_SECS)
        except TimeoutError as exc:
            logger.warning(
                "thread search state join failed",
                thread_id=thread_id,
                graph_id=graph_id,
                error=f"timeout after {_STATE_FETCH_TIMEOUT_SECS}s: {exc}",
            )
            return empty
        except Exception as exc:
            logger.warning(
                "thread search state join failed",
                thread_id=thread_id,
                graph_id=graph_id,
                error=str(exc),
            )
            return empty

        if not snapshot:
            return empty

        try:
            thread_state = self._state_service.convert_snapshot_to_thread_state(snapshot, thread_id)
        except Exception as exc:
            logger.warning(
                "thread search state conversion failed",
                thread_id=thread_id,
                error=str(exc),
            )
            return empty

        values = _maybe_truncate_values(dict(thread_state.values or {}))
        config_out: dict[str, Any] = {}
        snap_config = getattr(snapshot, "config", None)
        if isinstance(snap_config, dict):
            config_out = snap_config

        return {
            "values": values,
            "interrupts": list(thread_state.interrupts or []),
            "config": config_out,
        }


def _maybe_truncate_values(values: dict[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(values, default=str)
    except (TypeError, ValueError):
        return {"__truncated__": True}
    if len(encoded.encode("utf-8")) <= _VALUES_SOFT_CAP_BYTES:
        return values
    return {"__truncated__": True}
