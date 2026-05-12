"""Pregel-compatible wrapper for LangGraph.js graphs.

Wraps a JS graph (loaded via :class:`~aegra_api.services.js_bridge.JSBridge`)
so the existing Aegra streaming, checkpointing, and run machinery works
without modification.

Checkpointing is handled natively on the JS side by
``@langchain/langgraph-checkpoint-postgres`` — the wrapper delegates all
execution to the bridge process and only adapts the interface.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from aegra_api.services.js_bridge import JSBridge

logger = structlog.get_logger(__name__)


def _serialize_input(input_data: Any) -> dict[str, Any]:
    """Serialize graph input for the JS bridge.

    If *input_data* is a LangGraph ``Command`` (resume, goto, update),
    it is converted to a wire-format dict that the JS bridge reconstructs
    into the corresponding JS ``Command`` object.

    Regular dict inputs pass through unchanged.
    """
    # Check for LangGraph Command objects (resume/goto/update)
    try:
        from langgraph.types import Command

        if isinstance(input_data, Command):
            cmd: dict[str, Any] = {"__command__": {}}
            if input_data.resume is not None:
                cmd["__command__"]["resume"] = input_data.resume
            if input_data.goto is not None:
                cmd["__command__"]["goto"] = input_data.goto
            if input_data.update is not None:
                cmd["__command__"]["update"] = input_data.update
            return cmd
    except ImportError:
        pass

    if isinstance(input_data, dict):
        return input_data
    return {"input": input_data}


class JSGraphWrapper:
    """Presents a Pregel-like interface backed by a JS bridge.

    This is *not* a real ``Pregel`` subclass — it implements just enough
    of the public API so that :mod:`aegra_api.services.graph_streaming`,
    :mod:`aegra_api.services.langgraph_service`, and the run endpoints
    can work with it transparently.

    Checkpointing, interrupts, resume, and time-travel are handled by the
    native ``@langchain/langgraph-checkpoint-postgres`` checkpointer on the
    JS side. The Python wrapper does **not** touch checkpoints.

    Parameters
    ----------
    bridge:
        The :class:`JSBridge` instance managing the Node.js subprocess.
    graph_id:
        Unique graph identifier (matches the ``aegra.json`` key).
    graph_info:
        Schema information returned by ``bridge.load_graph()``.
    """

    def __init__(
        self,
        bridge: JSBridge,
        graph_id: str,
        graph_info: dict[str, Any],
        *,
        file_path: str = "",
        export_name: str = "graph",
    ) -> None:
        self._bridge = bridge
        self._graph_id = graph_id
        self._graph_info = graph_info
        self._file_path = file_path
        self._export_name = export_name

        # Checkpointer/store are injected via copy() for interface parity
        # with Pregel graphs. The actual checkpointing is done natively on
        # the JS side — these attributes are only inspected by the run
        # machinery to decide whether persistence is available.
        self.checkpointer: Any = None
        self.store: Any = None
        self.config: dict[str, Any] = {}

        # Expose schema info
        self._input_schema = graph_info.get("inputSchema", {})
        self._output_schema = graph_info.get("outputSchema", {})
        self._config_schema = graph_info.get("configSchema", {})

        # Name for display purposes
        self.name = graph_info.get("graphId", graph_id)

    # ------------------------------------------------------------------
    # Schema methods (used by assistant schema endpoints)
    # ------------------------------------------------------------------

    def get_input_jsonschema(self) -> dict[str, Any]:
        """Return the JSON Schema for graph input."""
        return copy.deepcopy(self._input_schema)

    def get_output_jsonschema(self) -> dict[str, Any]:
        """Return the JSON Schema for graph output."""
        return copy.deepcopy(self._output_schema)

    def get_context_jsonschema(self) -> dict[str, Any]:
        """Return the JSON Schema for graph context (empty for JS graphs)."""
        return {}

    def get_graph_state_jsonschema(self) -> dict[str, Any]:
        """Return the JSON Schema for graph state (same as output)."""
        return copy.deepcopy(self._output_schema)

    # ------------------------------------------------------------------
    # Execution methods
    # ------------------------------------------------------------------

    async def _ensure_bridge(self) -> None:
        """Re-start the bridge and re-load the graph if the process died."""
        if self._bridge.is_running:
            return
        await self._bridge.start()
        if self._file_path:
            await self._bridge.load_graph(
                self._file_path,
                self._export_name,
                self._graph_id,
            )
            await logger.ainfo(
                "Re-loaded JS graph after bridge restart",
                graph_id=self._graph_id,
            )

    async def ainvoke(
        self,
        input_data: Any,
        config: RunnableConfig | dict[str, Any] | None = None,
        *,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Invoke the graph and return the final state.

        Checkpointing is handled by the JS-side PostgresSaver — the
        ``thread_id`` in config is forwarded to the bridge.
        """
        await self._ensure_bridge()
        merged_config = self._merge_config(config)
        serialized_input = _serialize_input(input_data)

        result = await self._bridge.invoke(
            self._graph_id,
            serialized_input,
            merged_config,
        )

        return result.get("state", result)

    async def astream(
        self,
        input_data: Any,
        config: RunnableConfig | dict[str, Any] | None = None,
        *,
        stream_mode: list[str] | str = "values",
        context: dict[str, Any] | None = None,
        subgraphs: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[tuple[str, Any]]:
        """Stream graph execution events.

        Yields ``(mode, data)`` tuples compatible with the streaming
        service's ``_process_stream_event`` function.
        """
        await self._ensure_bridge()
        merged_config = self._merge_config(config)
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)
        serialized_input = _serialize_input(input_data)

        async for event in self._bridge.stream(
            self._graph_id,
            serialized_input,
            merged_config,
            stream_mode=modes,
        ):
            mode = event.get("mode", "values")
            data = event.get("data")
            yield (mode, data)

    async def astream_events(
        self,
        input_data: Any,
        config: RunnableConfig | dict[str, Any] | None = None,
        *,
        version: str = "v2",
        stream_mode: list[str] | str = "values",
        context: dict[str, Any] | None = None,
        subgraphs: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream events in LangChain v2 event format.

        Wraps :meth:`astream` output into the ``on_chain_stream`` event
        format that :func:`graph_streaming.stream_graph_events` expects
        when ``use_astream_events`` is True.
        """
        merged_config = self._merge_config(config)
        modes = [stream_mode] if isinstance(stream_mode, str) else list(stream_mode)

        run_id = (merged_config.get("configurable") or {}).get("run_id", "")

        async for mode, data in self.astream(
            input_data,
            merged_config,
            stream_mode=modes,
            context=context,
            subgraphs=subgraphs,
        ):
            yield {
                "event": "on_chain_stream",
                "run_id": run_id,
                "data": {"chunk": (mode, data)},
                "tags": [],
            }

    # ------------------------------------------------------------------
    # Copy / clone (used by LangGraphService.get_graph)
    # ------------------------------------------------------------------

    def copy(self, update: dict[str, Any] | None = None) -> JSGraphWrapper:
        """Return a shallow copy with optional attribute updates.

        This mirrors Pregel's ``copy(update={checkpointer, store, ...})``
        pattern used in :meth:`LangGraphService.get_graph`.
        """
        clone = JSGraphWrapper(
            bridge=self._bridge,
            graph_id=self._graph_id,
            graph_info=self._graph_info,
            file_path=self._file_path,
            export_name=self._export_name,
        )
        clone.config = copy.deepcopy(self.config)
        clone.checkpointer = self.checkpointer
        clone.store = self.store

        if update:
            for key, value in update.items():
                if hasattr(clone, key):
                    setattr(clone, key, value)

        return clone

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _merge_config(self, config: RunnableConfig | dict[str, Any] | None) -> dict[str, Any]:
        """Merge the provided config with the instance config."""
        base = copy.deepcopy(self.config) if self.config else {}
        if config:
            config_dict = dict(config) if not isinstance(config, dict) else config
            base_cfg = base.get("configurable", {})
            new_cfg = config_dict.get("configurable", {})
            merged_cfg = {**base_cfg, **new_cfg}
            base = {**base, **config_dict, "configurable": merged_cfg}
        return base

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"JSGraphWrapper(graph_id={self._graph_id!r})"
