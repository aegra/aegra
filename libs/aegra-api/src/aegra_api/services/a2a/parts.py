"""Conversion of A2A message parts into LangGraph graph input. The wire contract is issue #625."""

from dataclasses import dataclass
from typing import Any, Final

from aegra_api.services.a2a.jsonrpc import JsonRpcError, JsonRpcErrorCode

# ``file`` is the legacy nested shape; ``raw`` and ``url`` are the modern flat one.
FILE_KEYS: Final[tuple[str, ...]] = ("file", "raw", "url")


@dataclass(frozen=True)
class ConvertedParts:
    graph_input: dict[str, Any]
    has_conversational_content: bool


def convert_parts_to_graph_input(parts: list[Any], message_id: str, role: str) -> ConvertedParts:
    messages: list[dict[str, Any]] = []
    merged_data: dict[str, Any] = {}
    message_role = "user" if role in ("user", "ROLE_USER") else "assistant"

    for part in parts:
        if not isinstance(part, dict):
            raise JsonRpcError(
                JsonRpcErrorCode.CONTENT_TYPE_NOT_SUPPORTED,
                "Each message part must be an object. A2A agents support 'text', 'data', and 'file' parts only.",
            )

        present: list[str] = []
        if "text" in part:
            present.append("text")
        if any(key in part for key in FILE_KEYS):
            present.append("file")
        if "data" in part:
            present.append("data")

        if len(present) > 1:
            raise JsonRpcError(
                JsonRpcErrorCode.CONTENT_TYPE_NOT_SUPPORTED,
                "Each message part must contain exactly one of 'text', 'file', or 'data'. "
                f"Got multiple: {', '.join(present)}.",
            )

        if not present:
            raise JsonRpcError(
                JsonRpcErrorCode.CONTENT_TYPE_NOT_SUPPORTED,
                "Unsupported part type. A2A agents support 'text', 'data', and 'file' parts only.",
            )

        kind = present[0]

        if kind == "text":
            messages.append({"role": message_role, "content": part["text"], "id": message_id})
        elif kind == "data":
            if not isinstance(part["data"], dict):
                raise JsonRpcError(
                    JsonRpcErrorCode.CONTENT_TYPE_NOT_SUPPORTED,
                    "DataPart must contain a JSON object in the 'data' field",
                )
            merged_data.update(part["data"])
        elif kind == "file":
            raise JsonRpcError(
                JsonRpcErrorCode.CONTENT_TYPE_NOT_SUPPORTED,
                "File parts are not supported yet. Send 'text' or 'data' parts.",
            )

    graph_input: dict[str, Any] = {"messages": messages} if messages else {}
    # Data is applied last so a data key named ``messages`` overrides the converted ones.
    graph_input.update(merged_data)
    return ConvertedParts(graph_input=graph_input, has_conversational_content=bool(messages))
