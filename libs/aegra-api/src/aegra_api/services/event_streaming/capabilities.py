"""Runtime probe for Agent Protocol v2 streaming support.

The installed ``langgraph`` / ``langchain-core`` must be new enough to emit
native content-block message events. Rather than let a too-old runtime
surface as a mid-stream ImportError or a silently-empty message stream, we
probe the required symbols once and reject up front with a clear message.

Capability (can the runtime serve v2?) is separate from the feature flag
(is v2 turned on?) — they produce different remediation: "upgrade deps" vs
"flip FF_V2_EVENT_STREAMING".
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache

from aegra_api.settings import settings

# (module, attribute) pairs that a v2 run depends on. Submodule paths are
# intentional — these are not re-exported from the package top level.
_REQUIRED_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("langgraph.pregel._messages", "StreamMessagesHandlerV2"),
    ("langgraph.pregel._tools", "StreamToolCallHandler"),
    ("langgraph.stream._mux", "StreamMux"),
    ("langgraph.stream.transformers", "LifecycleTransformer"),
    ("langchain_core.language_models.chat_model_stream", "ChatModelStream"),
    ("langchain_core.language_models.chat_model_stream", "AsyncChatModelStream"),
    ("langchain_core.language_models._compat_bridge", "message_to_events"),
)


@dataclass(frozen=True)
class V2Capabilities:
    """Outcome of the runtime + flag probe for v2 event streaming."""

    ok: bool
    missing: tuple[str, ...] = ()
    disabled_by_flag: bool = False

    @property
    def error_message(self) -> str:
        """Human-readable reason, suitable for an error envelope or 400 body."""
        if self.ok:
            return ""
        if self.disabled_by_flag:
            return (
                "Agent Protocol v2 event streaming is disabled on this server "
                "(FF_V2_EVENT_STREAMING=false). Set FF_V2_EVENT_STREAMING=true to enable."
            )
        return (
            "Agent Protocol v2 event streaming is not supported by the installed "
            "langgraph/langchain-core version. Missing symbol(s): "
            + ", ".join(self.missing)
            + ". Upgrade langgraph and langchain-core to releases that ship the "
            "new streaming framework."
        )


@lru_cache(maxsize=1)
def _probe_runtime_symbols() -> tuple[str, ...]:
    """Return the missing required symbols (empty tuple means supported).

    Memoised: the installed packages can't change without a process restart.
    """
    missing: list[str] = []
    for module_path, attr in _REQUIRED_SYMBOLS:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            missing.append(f"{module_path}.{attr}")
            continue
        if not hasattr(module, attr):
            missing.append(f"{module_path}.{attr}")
    return tuple(missing)


def get_v2_capabilities() -> V2Capabilities:
    """Resolve whether v2 event streaming can be served right now.

    Checks the feature flag first, then the runtime symbols. The flag is
    read live (not cached) so it stays honest under settings overrides in
    tests; the symbol probe is memoised.
    """
    if not settings.event_streaming.FF_V2_EVENT_STREAMING:
        return V2Capabilities(ok=False, disabled_by_flag=True)

    missing = _probe_runtime_symbols()
    return V2Capabilities(ok=not missing, missing=missing)
