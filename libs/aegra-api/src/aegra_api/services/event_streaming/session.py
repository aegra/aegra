"""Thread-scoped session that turns one run's broker events into v2 events.

A session subscribes to a run's broker (the same broker the legacy SSE
path uses), translates each raw event into protocol channel events,
assigns a session-local monotonic ``seq``, applies the client's channel
filter, and yields wire envelopes. Lifecycle events are derived from the
run's terminal ``end`` / ``error`` broker signals.

Resume: a client passes the last ``seq`` it saw as ``since``; the session
replays the broker's buffer and skips anything at or below that cursor
before going live.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import structlog

from aegra_api.services.broker import broker_manager
from aegra_api.services.event_streaming.channels import is_supported_channel
from aegra_api.services.event_streaming.protocol import build_event
from aegra_api.services.event_streaming.translator import EventTranslator

logger = structlog.getLogger(__name__)

# Map a run's terminal broker status to a lifecycle AgentStatus.
_STATUS_TO_LIFECYCLE: dict[str, str] = {
    "success": "completed",
    "completed": "completed",
    "interrupted": "interrupted",
    "error": "failed",
}


class ThreadEventSession:
    """Streams v2 events for a single run, filtered to requested channels."""

    def __init__(self, run_id: str, *, channels: set[str], since: int | None = None) -> None:
        self._run_id = run_id
        self._channels = channels
        self._since = since
        self._seq = 0
        self._translator = EventTranslator()

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield v2 event envelopes for the run: replayed first, then live.

        The broker stores each event in both a replay buffer and a live
        queue, so an id seen during replay is skipped when it reappears
        live — dedup is on the durable broker ``event_id``.
        """
        broker = broker_manager.get_or_create_broker(self._run_id)
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
        """Translate one raw broker event into filtered, seq'd envelopes."""
        mode, payload = _unwrap(raw_event)
        if mode is None:
            return []

        channel_events = (
            self._lifecycle(payload) if mode in ("end", "error") else self._translator.translate(mode, payload)
        )

        envelopes: list[dict[str, Any]] = []
        for channel, params in channel_events:
            if not self._wants(channel):
                continue
            self._seq += 1
            if self._since is not None and self._seq <= self._since:
                continue
            envelopes.append(build_event(channel, params, seq=self._seq, event_id=_with_seq(event_id, self._seq)))
        return envelopes

    def _wants(self, channel: str) -> bool:
        """True if the client subscribed to this channel.

        The translator emits the base ``custom`` channel, so any custom
        subscription — plain ``custom`` or namespaced ``custom:<name>`` —
        matches it. Named filtering is a follow-up.
        """
        if channel == "custom":
            return any(c == "custom" or c.startswith("custom:") for c in self._channels)
        return channel in self._channels

    def _lifecycle(self, payload: Any) -> list[tuple[str, dict[str, Any]]]:
        """Build a lifecycle event from a terminal broker payload."""
        status = payload.get("status") if isinstance(payload, dict) else None
        event = _STATUS_TO_LIFECYCLE.get(status or "", "completed")
        data: dict[str, Any] = {"event": event}
        if isinstance(payload, dict) and (message := payload.get("message")):
            data["error"] = message
        return [("lifecycle", data)]


def _is_terminal(raw_event: Any) -> bool:
    """True for a run's final ``end`` / ``error`` broker event."""
    mode, _ = _unwrap(raw_event)
    return mode in ("end", "error")


def _unwrap(raw_event: Any) -> tuple[str | None, Any]:
    """Pull ``(mode, payload)`` out of a broker event; ``(None, None)`` if unknown."""
    if isinstance(raw_event, (tuple, list)) and len(raw_event) == 2:
        return raw_event[0], raw_event[1]
    return None, None


def _with_seq(event_id: str, seq: int) -> str:
    """Wire event_id: broker id plus the session seq, unique per emitted event.

    One raw broker event can fan out to several protocol events, so the
    broker id alone isn't unique; the seq suffix keeps each distinct.
    """
    return f"{event_id}:{seq}"


def validate_channels(channels: Any) -> tuple[set[str], list[str]]:
    """Split a requested channel list into (valid set, invalid names).

    Used by the route to reject unknown channels up front rather than
    opening a stream that never emits.
    """
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
