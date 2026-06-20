"""Channel constants for Agent Protocol v2 event streaming.

A channel is a top-level event stream a client subscribes to (the
``method`` of a server event). These mirror the protocol's channel set;
``custom:<name>`` is also accepted for user-defined custom events.
"""

from __future__ import annotations

# Channels a client may request on the SSE filter / subscribe.
SUPPORTED_CHANNELS: frozenset[str] = frozenset(
    {
        "values",
        "updates",
        "messages",
        "tools",
        "lifecycle",
        "input",
        "checkpoints",
        "tasks",
        "custom",
    }
)

_CUSTOM_CHANNEL_PREFIX = "custom:"


def is_supported_channel(channel: str) -> bool:
    """True for a known channel or a non-empty ``custom:<name>`` channel."""
    if channel in SUPPORTED_CHANNELS:
        return True
    if channel.startswith(_CUSTOM_CHANNEL_PREFIX):
        return len(channel) > len(_CUSTOM_CHANNEL_PREFIX)
    return False


# stream_mode set requested from langgraph on every v2 run. Drives which
# raw modes the translator can turn into channel events.
DEFAULT_RUN_STREAM_MODES: list[str] = [
    "values",
    "updates",
    "messages",
    "custom",
    "tasks",
]
