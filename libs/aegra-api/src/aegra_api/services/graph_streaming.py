"""Graph event streaming service.

This module provides streaming functionality for LangGraph graph executions,
handling message accumulation, event processing, and multiple stream modes.
"""

import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import aclosing
from typing import Any, cast

import structlog
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ToolMessageChunk,
    convert_to_messages,
    message_chunk_to_message,
)
from langchain_core.runnables import RunnableConfig
from langgraph.errors import (
    EmptyChannelError,
    EmptyInputError,
    GraphRecursionError,
    InvalidUpdateError,
)
from langgraph.pregel.debug import CheckpointPayload, TaskResultPayload
from pydantic import ValidationError
from pydantic.v1 import ValidationError as ValidationErrorLegacy

from aegra_api.services.js_graph_wrapper import JSGraphWrapper
from aegra_api.utils.run_utils import _filter_context_by_schema

logger = structlog.getLogger(__name__)

# Type alias for stream output
AnyStream = AsyncIterator[tuple[str, Any]]


def _normalize_checkpoint_task(task: dict[str, Any]) -> dict[str, Any]:
    """Normalize checkpoint task structure by extracting configurable state."""
    state_data = task.get("state")

    # Only process if state contains configurable data
    if not state_data or "configurable" not in state_data:
        return task

    configurable = state_data.get("configurable")
    if not configurable:
        return task

    # Restructure task with checkpoint reference
    task["checkpoint"] = configurable
    del task["state"]
    return task


def _normalize_checkpoint_payload(
    payload: CheckpointPayload | None,
) -> dict[str, Any] | None:
    """Normalize debug checkpoint payload structure.

    Ensures checkpoint payloads have consistent task formatting.
    """
    if not payload:
        return None

    # Process all tasks in the checkpoint
    normalized_tasks = [_normalize_checkpoint_task(t) for t in payload["tasks"]]

    return {
        **payload,
        "tasks": normalized_tasks,
    }


async def stream_graph_events(
    graph: Any,
    input_data: Any,
    config: RunnableConfig,
    *,
    stream_mode: list[str],
    context: dict[str, Any] | None = None,
    subgraphs: bool = False,
    output_keys: list[str] | None = None,
    on_checkpoint: Callable[[CheckpointPayload | None], None] = lambda _: None,
    on_task_result: Callable[[TaskResultPayload], None] = lambda _: None,
) -> AnyStream:
    """Stream events from a graph execution.

    Handles both standard streaming (astream) and event-based streaming (astream_events)
    depending on the graph type and requested stream modes. Automatically accumulates
    message chunks and yields appropriate partial/complete events.

    Args:
        graph: The graph instance to execute
        input_data: Input data for graph execution
        config: RunnableConfig for execution
        stream_mode: List of stream modes (e.g., ["messages", "values", "debug"])
        context: Optional context dictionary
        subgraphs: Whether to include subgraph namespaces in event types
        output_keys: Optional output channel keys for astream
        on_checkpoint: Callback invoked when checkpoint events are received
        on_task_result: Callback invoked when task result events are received

    Yields:
        Tuples of (mode, payload) where mode is the stream mode and payload is the event data
    """
    run_id = str(config.get("configurable", {}).get("run_id", uuid.uuid4()))

    # Prepare stream modes
    stream_modes_set: set[str] = set(stream_mode) - {"events"}
    if "debug" not in stream_modes_set:
        stream_modes_set.add("debug")

    # Check if graph is a remote (JavaScript) implementation
    is_js_graph = isinstance(graph, JSGraphWrapper)

    # Also check for external bridge (BaseRemotePregel)
    if not is_js_graph:
        try:
            from langgraph_api.js.base import BaseRemotePregel

            is_js_graph = isinstance(graph, BaseRemotePregel)
        except ImportError:
            pass

    # Python graphs need messages-tuple converted to standard messages mode
    if "messages-tuple" in stream_modes_set and not is_js_graph:
        stream_modes_set.discard("messages-tuple")
        stream_modes_set.add("messages")

    # Ensure updates mode is enabled for interrupt support
    updates_explicitly_requested = "updates" in stream_modes_set
    if not updates_explicitly_requested:
        stream_modes_set.add("updates")

    # Track whether to filter non-interrupt updates
    only_interrupt_updates = not updates_explicitly_requested

    # Apply context schema filtering if available
    if context and not is_js_graph:
        try:
            context_schema = graph.get_context_jsonschema()
            context = await _filter_context_by_schema(context, context_schema)
        except Exception as e:
            await logger.adebug(f"Failed to get context schema for filtering: {e}", exc_info=e)

    # Initialize streaming state
    messages: dict[str, BaseMessageChunk] = {}

    # Choose streaming method based on mode and graph type
    use_astream_events = "events" in stream_mode or is_js_graph

    # Yield metadata event
    yield (
        "metadata",
        {"run_id": run_id, "attempt": config.get("metadata", {}).get("run_attempt", 1)},
    )

    # Stream execution using appropriate method
    if use_astream_events:
        async with aclosing(
            graph.astream_events(
                input_data,
                config,
                context=context,
                version="v2",
                stream_mode=list(stream_modes_set),
                subgraphs=subgraphs,
            )
        ) as stream:
            async for event in stream:
                event = cast("dict", event)

                # Filter events marked as hidden
                if event.get("tags") and "langsmith:hidden" in event["tags"]:
                    continue

                # Extract message events from JavaScript graphs
                is_message_event = "messages" in stream_mode and is_js_graph and event.get("event") == "on_custom_event"

                if is_message_event:
                    event_name = event.get("name")
                    if event_name in (
                        "messages/complete",
                        "messages/partial",
                        "messages/metadata",
                    ):
                        yield event_name, event["data"]

                # Process on_chain_stream events
                if event.get("event") == "on_chain_stream" and event.get("run_id") == run_id:
                    chunk_data = event.get("data", {}).get("chunk")
                    if chunk_data is None:
                        continue

                    if subgraphs:
                        if isinstance(chunk_data, (tuple, list)) and len(chunk_data) == 3:
                            ns, mode, chunk = chunk_data
                        else:
                            # Fallback: assume 2-tuple
                            mode, chunk = chunk_data
                            ns = None
                    else:
                        if isinstance(chunk_data, (tuple, list)) and len(chunk_data) == 2:
                            mode, chunk = chunk_data
                        else:
                            # Single value
                            mode = "values"
                            chunk = chunk_data
                        ns = None

                    # Shared logic for processing events
                    processed = _process_stream_event(
                        mode=mode,
                        chunk=chunk,
                        namespace=ns,
                        subgraphs=subgraphs,
                        stream_mode=stream_mode,
                        messages=messages,
                        only_interrupt_updates=only_interrupt_updates,
                        on_checkpoint=on_checkpoint,
                        on_task_result=on_task_result,
                    )

                    if processed:
                        for event_tuple in processed:
                            yield event_tuple

                    # Update checkpoint state for debug tracking
                    if mode == "debug" and chunk.get("type") == "checkpoint":
                        _normalize_checkpoint_payload(chunk.get("payload"))

                    # Also yield as raw "events" event if "events" mode requested
                    # This ensures on_chain_stream events are available as raw events
                    if "events" in stream_mode:
                        yield "events", event

                # Pass through raw events if "events" mode requested
                elif "events" in stream_mode:
                    yield "events", event

    else:
        # Use astream for standard streaming
        if output_keys is None:
            output_keys = getattr(graph, "output_channels", None)

        async with aclosing(
            graph.astream(
                input_data,
                config,
                context=context,
                stream_mode=list(stream_modes_set),
                output_keys=output_keys,
                subgraphs=subgraphs,
            )
        ) as stream:
            async for event in stream:
                # Parse event tuple
                if subgraphs:
                    if isinstance(event, tuple) and len(event) == 3:
                        ns, mode, chunk = event
                    else:
                        # Fallback: assume 2-tuple format
                        mode, chunk = cast("tuple[str, dict[str, Any]]", event)
                        ns = None
                else:
                    mode, chunk = cast("tuple[str, dict[str, Any]]", event)
                    ns = None

                # Shared logic for processing events
                processed = _process_stream_event(
                    mode=mode,
                    chunk=chunk,
                    namespace=ns,
                    subgraphs=subgraphs,
                    stream_mode=stream_mode,
                    messages=messages,
                    only_interrupt_updates=only_interrupt_updates,
                    on_checkpoint=on_checkpoint,
                    on_task_result=on_task_result,
                )

                if processed:
                    for event_tuple in processed:
                        yield event_tuple

                # Update checkpoint state for debug tracking
                if mode == "debug" and chunk.get("type") == "checkpoint":
                    _normalize_checkpoint_payload(chunk.get("payload"))


def _process_stream_event(
    mode: str,
    chunk: Any,
    namespace: str | None,
    subgraphs: bool,
    stream_mode: list[str],
    messages: dict[str, BaseMessageChunk],
    only_interrupt_updates: bool,
    on_checkpoint: Callable[[CheckpointPayload | None], None],
    on_task_result: Callable[[TaskResultPayload], None],
) -> list[tuple[str, Any]] | None:
    """Process a single stream event and generate output events.

    Handles message accumulation, debug events, and stream mode routing.
    Used by both astream and astream_events execution paths.

    Args:
        mode: The stream mode (e.g., "messages", "values", "debug")
        chunk: The event chunk data
        namespace: Optional namespace for subgraph events
        subgraphs: Whether subgraph namespaces should be included
        stream_mode: List of requested stream modes
        messages: Dictionary for accumulating message chunks by ID
        only_interrupt_updates: Whether to filter non-interrupt updates
        on_checkpoint: Callback for checkpoint events
        on_task_result: Callback for task result events

    Returns:
        List of (mode, payload) tuples to yield, or None if nothing to yield
    """
    results: list[tuple[str, Any]] = []

    # Process debug mode events
    if mode == "debug":
        debug_type = chunk.get("type")

        if debug_type == "checkpoint":
            # Normalize checkpoint and invoke callback
            normalized = _normalize_checkpoint_payload(chunk.get("payload"))
            chunk["payload"] = normalized
            on_checkpoint(normalized)
        elif debug_type == "task_result":
            # Forward task results to callback
            on_task_result(chunk.get("payload"))

    # Handle messages mode
    if mode == "messages":
        if "messages-tuple" in stream_mode:
            # Pass through raw tuple format
            if subgraphs and namespace:
                ns_str = "|".join(namespace) if isinstance(namespace, (list, tuple)) else str(namespace)
                results.append((f"messages|{ns_str}", chunk))
            else:
                results.append(("messages", chunk))
        else:
            # Accumulate and yield messages/partial or messages/complete
            msg_, meta = cast("tuple[BaseMessage | dict, dict[str, Any]]", chunk)

            # Handle dict-to-message conversion
            is_chunk_type = False
            if isinstance(msg_, dict):
                msg_type = msg_.get("type", "").lower()
                msg_role = msg_.get("role", "").lower()

                # Detect if this is a streaming chunk based on type/role indicators
                has_chunk_indicator = "chunk" in msg_type or "chunk" in msg_role

                if has_chunk_indicator:
                    # Instantiate appropriate chunk class based on role
                    if "ai" in msg_role:
                        msg = AIMessageChunk(**msg_)  # type: ignore[arg-type]
                    elif "tool" in msg_role:
                        msg = ToolMessageChunk(**msg_)  # type: ignore[arg-type]
                    else:
                        msg = BaseMessageChunk(**msg_)  # type: ignore[arg-type]
                    is_chunk_type = True
                else:
                    # Complete message - convert to proper message instance
                    msg = convert_to_messages([msg_])[0]
            else:
                msg = msg_

            # Track and accumulate messages by ID
            msg_id = msg.id
            is_new_message = msg_id not in messages

            if is_new_message:
                messages[msg_id] = msg
                # First time seeing this message - send metadata
                results.append(("messages/metadata", {msg_id: {"metadata": meta}}))
            else:
                # Accumulate additional chunks
                messages[msg_id] += msg

            # Determine event type based on message instance type
            is_partial_message = isinstance(msg, BaseMessageChunk)
            event_name = "messages/partial" if is_partial_message else "messages/complete"

            # Format accumulated message for output
            if is_chunk_type:
                # Keep raw chunks for streaming messages
                formatted_msg = messages[msg_id]
            else:
                # Convert accumulated chunks to complete message
                formatted_msg = message_chunk_to_message(messages[msg_id])

            results.append((event_name, [formatted_msg]))

    # Handle other stream modes
    elif mode in stream_mode:
        if subgraphs and namespace:
            ns_str = "|".join(namespace) if isinstance(namespace, (list, tuple)) else str(namespace)
            results.append((f"{mode}|{ns_str}", chunk))
        else:
            results.append((mode, chunk))

    # Special handling for interrupt events when updates mode not explicitly requested
    elif mode == "updates" and only_interrupt_updates:
        # Check if this update contains interrupt data
        has_interrupt_data = (
            isinstance(chunk, dict) and "__interrupt__" in chunk and len(chunk.get("__interrupt__", [])) > 0
        )

        if has_interrupt_data:
            # Remap interrupt updates to values events for backward compatibility
            if subgraphs and namespace:
                ns_str = "|".join(namespace) if isinstance(namespace, (list, tuple)) else str(namespace)
                results.append((f"values|{ns_str}", chunk))
            else:
                results.append(("values", chunk))

    return results if results else None


# Expected error types for error handling
EXPECTED_ERRORS = (
    ValueError,
    InvalidUpdateError,
    GraphRecursionError,
    EmptyInputError,
    EmptyChannelError,
    ValidationError,
    ValidationErrorLegacy,
)
