import copy
from typing import Any

import structlog
from langgraph.types import Command, Send

logger = structlog.getLogger(__name__)

# A body-supplied thread_id overrides the route-verified one (the checkpointer
# keys on thread_id alone), so the server pins these instead of trusting them.
SERVER_PINNED_CONFIG_KEYS: frozenset[str] = frozenset({"thread_id", "run_id"})


def strip_pinned_config_keys(client_config: dict[str, Any]) -> dict[str, Any]:
    """Drop server-authoritative identity keys from a client-supplied config dict."""
    return {k: v for k, v in client_config.items() if k not in SERVER_PINNED_CONFIG_KEYS}


def map_command_to_langgraph(cmd: dict[str, Any]) -> Command:
    """Convert an API command dict to a LangGraph Command object."""
    goto = cmd.get("goto")
    if goto is not None and not isinstance(goto, list):
        goto = [goto]

    cmd_update = cmd.get("update")
    if isinstance(cmd_update, (tuple, list)) and all(
        isinstance(t, (tuple, list)) and len(t) == 2 and isinstance(t[0], str) for t in cmd_update
    ):
        cmd_update = [tuple(t) for t in cmd_update]

    return Command(
        update=cmd_update,
        goto=([it if isinstance(it, str) else Send(it["node"], it["input"]) for it in goto] if goto else None),
        resume=cmd.get("resume"),
    )


def _should_skip_event(raw_event: Any) -> bool:
    """Check if an event should be skipped based on langsmith:nostream tag"""
    try:
        # Check if the event has metadata with tags containing 'langsmith:nostream'
        if isinstance(raw_event, tuple) and len(raw_event) >= 2:
            # For tuple events, check the third element (metadata tuple)
            metadata_tuple = raw_event[len(raw_event) - 1]
            if isinstance(metadata_tuple, tuple) and len(metadata_tuple) >= 2:
                # Get the second item in the metadata tuple
                metadata = metadata_tuple[1]
                if isinstance(metadata, dict) and "tags" in metadata:
                    tags = metadata["tags"]
                    if isinstance(tags, list) and "langsmith:nostream" in tags:
                        return True
        return False
    except Exception:
        # If we can't parse the event structure, don't skip it
        return False


def _merge_jsonb(*objects: dict) -> dict:
    """Mimics PostgreSQL's JSONB merge behavior"""
    result = {}
    for obj in objects:
        if obj is not None:
            result.update(copy.deepcopy(obj))
    return result


async def _filter_context_by_schema(context: dict[str, Any], context_schema: dict | None) -> dict[str, Any]:
    """Filter context parameters based on the context schema."""
    if not context_schema or not context:
        return context

    # Extract valid properties from the schema
    properties = context_schema.get("properties", {})
    if not properties:
        return context

    # Filter context to only include parameters defined in the schema
    filtered_context = {}
    for key, value in context.items():
        if key in properties:
            filtered_context[key] = value
        else:
            await logger.adebug(
                f"Filtering out context parameter '{key}' not found in context schema",
                context_key=key,
                available_keys=list(properties.keys()),
            )

    return filtered_context
