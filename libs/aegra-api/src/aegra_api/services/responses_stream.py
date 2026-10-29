"""Translate broker events into Open Responses SSE events."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from aegra_api.core.sse import format_sse_message
from aegra_api.models import Run
from aegra_api.services.responses_mapping import (
    _as_mapping,
    _message_role,
    _text_from_content,
    interrupt_items,
    project_response_output,
)
from aegra_api.services.streaming_service import streaming_service
from aegra_api.utils.sse_utils import extract_event_sequence

_TERMINAL_RUN_STATUSES = frozenset({"success", "error", "timeout", "interrupted"})


class ResponsesTranslator:
    """Fold ``messages`` and ``values`` events into Responses SSE payloads."""

    def __init__(self, response: dict[str, Any], *, message_offset: int = 0) -> None:
        self._message_offset = message_offset
        self.response = dict(response)
        initial = response.get("output")
        self._initial_output: list[dict[str, Any]] = list(initial) if isinstance(initial, list) else []
        self.response["output"] = []
        self._text: dict[str, str] = {}
        self._opened: set[str] = set()
        self._indexes: dict[str, int] = {}
        self._output_items: list[dict[str, Any]] = []
        self._interrupt_ids: set[str] = set()
        self._last_values: dict[str, Any] | None = None
        self._native_item_id: str | None = None
        self._closed = False
        self.saw_interrupt = False

    def opening_events(self) -> list[dict[str, Any]]:
        """``response.created`` and ``response.in_progress`` before broker events."""
        created = dict(self.response)
        created["status"] = "in_progress"
        created["output"] = []
        return [
            {"type": "response.created", "response": created},
            {"type": "response.in_progress", "response": created},
        ]

    def feed(self, event_type: str, payload: Any) -> list[dict[str, Any]]:
        """Translate one broker event. Later events after a terminal event are ignored."""
        if self._closed:
            return []
        if event_type in ("messages/partial", "messages/complete"):
            return self._on_messages(event_type, payload)
        if event_type == "messages":
            return self._on_native_message(payload)
        if event_type == "values" or event_type.startswith("values"):
            if isinstance(payload, dict):
                self._last_values = payload
            return self._on_interrupts(payload)
        if event_type == "error":
            return self._fail(_error_message(payload))
        if event_type == "end":
            return self._on_end(payload if isinstance(payload, dict) else {})
        return []

    def snapshot_events(self) -> list[dict[str, Any]]:
        """A finished response as a short SSE sequence, with no token deltas."""
        events = self.opening_events()
        status = self.response.get("status")
        if status == "failed":
            events.extend(self._fail(self._stored_error()))
            return events
        self._copy_output(self._initial_output)
        if status == "incomplete":
            events.append(self._terminal("response.incomplete"))
            return events
        events.append(self._terminal("response.completed"))
        return events

    def _on_messages(self, event_type: str, payload: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for message in _message_payloads(payload):
            mapped = _as_mapping(message)
            if mapped is None or _message_role(mapped) != "assistant":
                continue
            raw_id = mapped.get("id")
            item_id = raw_id if isinstance(raw_id, str) else "msg"
            text = _text_from_content(mapped.get("content"))
            events.extend(self._ensure_open(item_id))
            previous = self._text.get(item_id, "")
            if text.startswith(previous):
                delta = text[len(previous) :]
            else:
                delta = text
            self._text[item_id] = text
            if delta:
                events.append(
                    {
                        "type": "response.output_text.delta",
                        "item_id": item_id,
                        "output_index": self._indexes[item_id],
                        "content_index": 0,
                        "delta": delta,
                    }
                )
            if event_type == "messages/complete":
                events.extend(self._close_message(item_id, text))
        return events

    def _ensure_open(self, item_id: str) -> list[dict[str, Any]]:
        if item_id in self._opened:
            return []
        index = len(self._output_items)
        self._indexes[item_id] = index
        self._opened.add(item_id)
        item = {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [{"type": "output_text", "text": "", "annotations": []}],
        }
        self._output_items.append(item)
        return [
            {"type": "response.output_item.added", "output_index": index, "item": item},
            {
                "type": "response.content_part.added",
                "item_id": item_id,
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            },
        ]

    def _close_message(self, item_id: str, text: str) -> list[dict[str, Any]]:
        index = self._indexes[item_id]
        item = {
            "id": item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        self._output_items[index] = item
        return [
            {
                "type": "response.output_text.done",
                "item_id": item_id,
                "output_index": index,
                "content_index": 0,
                "text": text,
            },
            {
                "type": "response.content_part.done",
                "item_id": item_id,
                "output_index": index,
                "content_index": 0,
                "part": {"type": "output_text", "text": text, "annotations": []},
            },
            {"type": "response.output_item.done", "output_index": index, "item": item},
        ]

    def _on_interrupts(self, payload: Any) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for item in interrupt_items(payload if isinstance(payload, dict) else None):
            interrupt_id = item["id"]
            if interrupt_id in self._interrupt_ids:
                continue
            self._interrupt_ids.add(interrupt_id)
            self.saw_interrupt = True
            index = len(self._output_items)
            self._output_items.append(item)
            events.append({"type": "response.output_item.added", "output_index": index, "item": item})
            events.append({"type": "response.output_item.done", "output_index": index, "item": item})
        return events

    def _on_end(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        status = payload.get("status")
        if status == "error":
            return self._fail("execution failed")
        if status == "interrupted" and self.saw_interrupt:
            self.response["status"] = "incomplete"
            self.response["incomplete_details"] = {"reason": "approval_required"}
            self._reconcile_final_values()
            return [self._terminal("response.incomplete")]
        if status == "interrupted":
            self.response["status"] = "cancelled"
            self._reconcile_final_values()
            return [self._terminal("response.completed")]
        self.response["status"] = "completed"
        self._reconcile_final_values()
        return [self._terminal("response.completed")]

    def _fail(self, message: str) -> list[dict[str, Any]]:
        self.response["status"] = "failed"
        self.response["error"] = {"code": "server_error", "message": message}
        return [self._terminal("response.failed")]

    def _reconcile_final_values(self) -> None:
        """Close unfinished messages and merge tool calls from the final state."""
        self._close_open_messages()
        if not isinstance(self._last_values, dict):
            self.response["output"] = list(self._output_items)
            return
        projected = project_response_output(self._last_values, offset=self._message_offset)
        if not self._output_items:
            self._copy_output(projected)
            return
        message_index = {
            item["id"]: index
            for index, item in enumerate(self._output_items)
            if item.get("type") == "message" and isinstance(item.get("id"), str)
        }
        seen = {(item.get("type"), item.get("id"), item.get("call_id")) for item in self._output_items}
        for item in projected:
            if item.get("type") == "message" and item.get("id") in message_index:
                index = message_index[item["id"]]
                if len(_item_text(item)) >= len(_item_text(self._output_items[index])):
                    self._output_items[index] = item
                continue
            key = (item.get("type"), item.get("id"), item.get("call_id"))
            if key not in seen:
                self._output_items.append(item)
                seen.add(key)
        self.response["output"] = list(self._output_items)

    def _close_open_messages(self) -> None:
        for index, item in enumerate(self._output_items):
            if item.get("type") != "message" or item.get("status") != "in_progress":
                continue
            raw_id = item.get("id")
            item_id = raw_id if isinstance(raw_id, str) else "msg"
            text = self._text.get(item_id, "")
            self._output_items[index] = {
                **item,
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }

    def _on_native_message(self, payload: Any) -> list[dict[str, Any]]:
        """Translate an Agent Protocol v2 ``messages`` event into Responses deltas."""
        data = _native_message_data(payload)
        if data is None:
            return []
        kind = data.get("event")
        if kind == "message-start":
            role = data.get("role")
            if isinstance(role, str) and role.lower() not in ("ai", "assistant"):
                self._native_item_id = None
                return []
            raw_id = data.get("id")
            item_id = raw_id if isinstance(raw_id, str) else "msg"
            self._native_item_id = item_id
            return self._ensure_open(item_id)
        if kind == "content-block-delta":
            delta = data.get("delta")
            text = delta.get("text") if isinstance(delta, dict) else None
            item_id = self._native_item_id
            if not isinstance(text, str) or not text or item_id is None:
                return []
            events = self._ensure_open(item_id)
            self._text[item_id] = self._text.get(item_id, "") + text
            events.append(
                {
                    "type": "response.output_text.delta",
                    "item_id": item_id,
                    "output_index": self._indexes[item_id],
                    "content_index": 0,
                    "delta": text,
                }
            )
            return events
        if kind == "message-finish":
            item_id = self._native_item_id
            self._native_item_id = None
            if not isinstance(item_id, str) or item_id not in self._opened:
                return []
            return self._close_message(item_id, self._text.get(item_id, ""))
        return []

    def _copy_output(self, items: list[dict[str, Any]]) -> None:
        self._output_items = items
        self.response["output"] = items

    def _terminal(self, event_type: str) -> dict[str, Any]:
        self._closed = True
        self.response["output"] = list(self._output_items)
        return {"type": event_type, "response": dict(self.response)}

    def _stored_error(self) -> str:
        error = self.response.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
        return "Run failed"


def encode_responses_event(event: dict[str, Any], event_id: str | None) -> str:
    """Serialize one Responses event as an SSE frame."""
    event_type = event.get("type")
    name = event_type if isinstance(event_type, str) else "response.completed"
    return format_sse_message(name, event, event_id)


async def iter_responses_sse(
    run: Run,
    response: dict[str, Any],
    *,
    starting_after: str | None,
    message_offset: int = 0,
) -> AsyncGenerator[str, None]:
    """Stream a run. Replay stored events first; snapshot only when that history is gone."""
    translator = ResponsesTranslator(response, message_offset=message_offset)
    if starting_after is None:
        for event in translator.opening_events():
            yield encode_responses_event(event, None)

    cursor_seq = _cursor_sequence(starting_after)
    cursor_found = starting_after is None
    saw_event = False
    async for event_id, event_type, payload in streaming_service.iter_broker_events(run, None):
        saw_event = True
        produced = translator.feed(event_type, payload)
        if starting_after is None:
            for event in produced:
                yield encode_responses_event(event, event_id)
            continue
        if event_id == starting_after:
            cursor_found = True
            continue
        if cursor_found or (cursor_seq is not None and extract_event_sequence(event_id) > cursor_seq):
            cursor_found = True
            for event in produced:
                yield encode_responses_event(event, event_id)

    if starting_after is not None and not cursor_found:
        if not saw_event and run.status in _TERMINAL_RUN_STATUSES:
            for event in translator.snapshot_events():
                yield encode_responses_event(event, None)
            return
        yield encode_responses_event(
            {
                "type": "error",
                "error": {
                    "message": "starting_after is no longer available",
                    "type": "invalid_request_error",
                    "code": "invalid_cursor",
                },
            },
            None,
        )
        return

    if not saw_event and run.status in _TERMINAL_RUN_STATUSES:
        for event in translator.snapshot_events():
            yield encode_responses_event(event, None)


def _message_payloads(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if payload is None:
        return []
    return [payload]


def _cursor_sequence(event_id: str | None) -> int | None:
    """Numeric suffix of a broker event id, or None when the id is not one of ours."""
    if event_id is None or "_event_" not in event_id:
        return None
    sequence = extract_event_sequence(event_id)
    return sequence if sequence > 0 else None


def _native_message_data(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict) and isinstance(payload.get("event"), str):
        return payload
    if isinstance(payload, dict):
        data = payload.get("params", {}).get("data") if isinstance(payload.get("params"), dict) else None
        if isinstance(data, dict) and isinstance(data.get("event"), str):
            return data
    return None


def _item_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if not isinstance(content, list) or not content or not isinstance(content[0], dict):
        return ""
    text = content[0].get("text")
    return text if isinstance(text, str) else ""


def _error_message(payload: Any) -> str:
    if isinstance(payload, dict) and isinstance(payload.get("message"), str):
        return payload["message"]
    return "execution failed"
