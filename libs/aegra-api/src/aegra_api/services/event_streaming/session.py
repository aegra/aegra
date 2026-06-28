"""Thread-scoped session that forwards a thread's native v3 events as v2 events.

A v2 stream is scoped to a *thread*, not a run (matching the LangGraph SDK):
the client opens the stream, issues ``run.start``, and the events of whatever
run(s) execute on the thread flow through. The run's broker already holds
native protocol events (the v3 producer wrote them) as ``(method, event)``
pairs, plus a terminal ``("end"|"error", payload)``. The session re-envelopes
each under one thread-monotonic ``seq``: it restamps seq/event_id, filters by
channel, splits interrupts onto the ``input`` channel, normalizes state
message payloads, and derives the terminal lifecycle event.

``seq`` is the reconnect cursor: a client resends the last ``seq`` it saw as
``since`` and the session skips anything at or below it. ``event_id`` is a
distinct unique-per-event string used by the client for dedup.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from aegra_api.services.broker import broker_manager
from aegra_api.services.event_streaming.channels import is_supported_channel
from aegra_api.services.event_streaming.normalizers import (
    lifecycle_status,
    normalize_input_requested,
    normalize_state_payload,
    normalize_updates,
    strip_interrupts,
)
from aegra_api.services.event_streaming.protocol import build_event

__all__ = ["RunLister", "ThreadEventSession", "validate_channels"]

# Grace window covering the SDK gap between stream open and run.start landing.
_IDLE_GRACE_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 0.25

# Async callable returning the thread's run ids, oldest first.
RunLister = Callable[[], Awaitable[list[str]]]

# A projected channel event: (wire method, params.data, namespace).
_ChannelEvent = tuple[str, dict[str, Any], list[str]]


class ThreadEventSession:
    """Forwards native v3 events for all runs on a thread, filtered to channels."""

    def __init__(
        self,
        thread_id: str,
        *,
        channels: set[str],
        list_run_ids: RunLister,
        since: int | None = None,
        idle_grace_seconds: float = _IDLE_GRACE_SECONDS,
    ) -> None:
        self._thread_id = thread_id
        self._channels = channels
        self._list_run_ids = list_run_ids
        self._since = since
        self._idle_grace = idle_grace_seconds
        self._seq = 0
        self._drained: set[str] = set()

    @property
    def applied_through_seq(self) -> int:
        """The highest seq assigned so far (the SDK's initial cursor)."""
        return self._seq

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield v2 envelopes for the thread's runs until they finish + idle.

        The stream stays open across the gap between runs so a HITL resume
        (``input.respond`` starts a fresh run on the same thread) is delivered
        on the same SSE connection the SDK holds open. An interrupted run keeps
        the full grace window; a terminal run still waits one grace window so a
        same-thread follow-up run is not dropped, but never blocks forever.
        """
        idle_deadline: float | None = None
        loop = asyncio.get_running_loop()

        while True:
            progressed = False
            for run_id in await self._fresh_run_ids():
                async for envelope in self._drain_run(run_id):
                    progressed = True
                    yield envelope
                self._drained.add(run_id)

            if progressed:
                idle_deadline = None
                continue

            now = loop.time()
            if idle_deadline is None:
                idle_deadline = now + self._idle_grace
            elif now >= idle_deadline:
                return
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    async def _fresh_run_ids(self) -> list[str]:
        return [run_id for run_id in await self._list_run_ids() if run_id not in self._drained]

    async def _drain_run(self, run_id: str) -> AsyncIterator[dict[str, Any]]:
        """Replay then tail one run's broker, re-enveloping each native event."""
        broker = broker_manager.get_or_create_broker(run_id)
        seen: set[str] = set()

        for event_id, raw_event in await broker.replay(None):
            seen.add(event_id)
            for envelope in self._project(event_id, raw_event):
                yield envelope
            if _is_terminal(raw_event):
                return

        async for event_id, raw_event in broker.aiter():
            if event_id in seen:
                continue
            for envelope in self._project(event_id, raw_event):
                yield envelope
            if _is_terminal(raw_event):
                return

    def _project(self, event_id: str, raw_event: Any) -> list[dict[str, Any]]:
        """Re-envelope one raw broker event into filtered, seq'd envelopes."""
        method, payload = _unwrap(raw_event)
        if method is None:
            return []

        channel_events = self._channel_events(method, payload)

        envelopes: list[dict[str, Any]] = []
        for index, (channel, data, namespace) in enumerate(channel_events):
            # seq counts before channel filtering — absolute cursor, so a
            # reconnect with a different channel set still resumes correctly.
            self._seq += 1
            if not self._wants(channel):
                continue
            if self._since is not None and self._seq <= self._since:
                continue
            envelopes.append(
                build_event(
                    channel,
                    data,
                    namespace=namespace,
                    seq=self._seq,
                    event_id=f"{event_id}:{index}",
                )
            )
        return envelopes

    def _channel_events(self, method: str, payload: Any) -> list[_ChannelEvent]:
        """Map one raw broker event to zero or more (channel, data, namespace)."""
        if method in ("end", "error"):
            return self._lifecycle(method, payload)

        # Native producer events: payload is a ProtocolEvent dict.
        params = payload.get("params", {}) if isinstance(payload, dict) else {}
        data = params.get("data")
        namespace = params.get("namespace") or []

        if method == "values":
            return self._values_events(data, params.get("interrupts"), namespace)
        if method == "updates":
            return self._updates_events(data, namespace)
        return [(method, data if isinstance(data, dict) else {"value": data}, namespace)]

    def _values_events(self, data: Any, interrupts: Any, namespace: list[str]) -> list[_ChannelEvent]:
        """Split interrupts onto the input channel; forward cleaned, normalized values."""
        events: list[_ChannelEvent] = []
        requests, cleaned = strip_interrupts(data)
        if isinstance(interrupts, (list, tuple)) and interrupts:
            requests = _dedupe_requests(requests + normalize_input_requested(_coerce_interrupts(interrupts)))
        for request in requests:
            events.append(("input.requested", request, namespace))
        if _has_state(cleaned):
            events.append(("values", normalize_state_payload(cleaned), namespace))
        return events

    def _updates_events(self, data: Any, namespace: list[str]) -> list[_ChannelEvent]:
        """Forward updates, splitting any embedded interrupt onto the input channel.

        v3 emits updates as raw ``{node: values}``; an interrupt arrives as the
        ``__interrupt__`` node whose values are the interrupt array.
        """
        if isinstance(data, dict) and "__interrupt__" in data:
            return [("input.requested", req, namespace) for req in normalize_input_requested(data["__interrupt__"])]
        normalized = normalize_updates(data)
        normalized["values"] = normalize_state_payload(normalized["values"])
        return [("updates", normalized, namespace)]

    def _lifecycle(self, method: str, payload: Any) -> list[_ChannelEvent]:
        """Build a lifecycle event from a terminal broker payload.

        The ``error`` broker event carries ``{error, message}`` with no status —
        it is terminal and drains the run before the trailing ``end`` event, so
        the failed status comes from the method, not a status key.
        """
        status = payload.get("status") if isinstance(payload, dict) else None
        if method == "error":
            status = "error"
        data: dict[str, Any] = {"event": lifecycle_status(status or "")}
        if isinstance(payload, dict) and (message := payload.get("message")):
            data["error"] = message
        return [("lifecycle", data, [])]

    def _wants(self, channel: str) -> bool:
        """True if the client subscribed to this channel.

        The ``input.requested`` wire method maps to the ``input`` channel. A
        custom subscription — plain ``custom`` or namespaced ``custom:<name>``
        — matches the base ``custom`` channel. Named filtering is a follow-up.
        """
        if channel == "input.requested":
            return "input" in self._channels
        if channel == "custom":
            return any(c == "custom" or c.startswith("custom:") for c in self._channels)
        return channel in self._channels


def _is_terminal(raw_event: Any) -> bool:
    """True for a run's final ``end`` / ``error`` broker event."""
    method, _ = _unwrap(raw_event)
    return method in ("end", "error")


def _unwrap(raw_event: Any) -> tuple[str | None, Any]:
    """Pull ``(method, payload)`` out of a broker event; ``(None, None)`` if unknown."""
    if isinstance(raw_event, (tuple, list)) and len(raw_event) == 2:
        return raw_event[0], raw_event[1]
    return None, None


def _has_state(value: Any) -> bool:
    """True if a values/updates payload carries forwardable state."""
    if isinstance(value, dict):
        return bool(value)
    return value is not None


def _coerce_interrupts(interrupts: Any) -> list[dict[str, Any]]:
    """Coerce native Interrupt objects/dicts into ``{id, value}`` entries."""
    out: list[dict[str, Any]] = []
    for item in interrupts:
        if isinstance(item, dict):
            out.append(item)
            continue
        interrupt_id = getattr(item, "id", None)
        if isinstance(interrupt_id, str):
            out.append({"id": interrupt_id, "value": getattr(item, "value", None)})
    return out


def _dedupe_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop duplicate input requests by interrupt_id, preserving order."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for request in requests:
        interrupt_id = request.get("interrupt_id")
        if interrupt_id in seen:
            continue
        if isinstance(interrupt_id, str):
            seen.add(interrupt_id)
        out.append(request)
    return out


def validate_channels(channels: Any) -> tuple[set[str], list[str]]:
    """Split a requested channel list into (valid set, invalid names)."""
    if not isinstance(channels, list) or not channels:
        return set(), ["channels must be a non-empty array"]
    valid: set[str] = set()
    invalid: list[str] = []
    for channel in channels:
        if isinstance(channel, str) and is_supported_channel(channel):
            valid.add(channel)
        else:
            invalid.append(str(channel))
    return valid, invalid
