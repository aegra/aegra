"""Thread TTL: config resolution and expiry sweep (issue #288 phase 2).

Threads opt into a TTL via a ``thread_ttl`` row (server default on creation or
per-thread override). A background sweep claims expired rows with
``FOR UPDATE SKIP LOCKED`` and applies the row's strategy:

- ``delete``  — remove checkpoints then the thread row (cascades runs/crons).
- ``keep_latest`` — prune checkpoint history, keep the latest state, re-arm.
"""

import json
from functools import cache
from typing import Literal

import structlog
from pydantic import BaseModel, Field

from aegra_api.config import load_checkpointer_config
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)


class ThreadTTLConfig(BaseModel):
    """Validated TTL configuration merged from env and aegra.json."""

    strategy: Literal["delete", "keep_latest"] = "delete"
    default_ttl: float = Field(43200, gt=0)  # minutes; 30 days
    sweep_interval_minutes: float = Field(5, gt=0)
    sweep_limit: int = Field(1000, gt=0)


@cache
def get_thread_ttl_config() -> ThreadTTLConfig | None:
    """Resolve TTL config: AEGRA_THREAD_TTL env var wins over aegra.json.

    The env var is either a bare number (default_ttl in minutes) or a JSON
    object; when set it replaces the checkpointer.ttl block entirely (same
    whole-source precedence as DATABASE_URL over POSTGRES_*). Returns None
    when neither source is configured — the feature is off. Invalid config
    raises so a misconfigured retention policy fails at startup instead of
    silently deleting (or retaining) the wrong data.
    """
    raw = settings.thread_ttl.AEGRA_THREAD_TTL
    if raw is not None and raw.strip():
        try:
            data: dict[str, object] = {"default_ttl": float(raw)}
        except ValueError:
            data = json.loads(raw)
        return ThreadTTLConfig.model_validate(data)

    checkpointer_config = load_checkpointer_config()
    ttl_config = checkpointer_config.get("ttl") if checkpointer_config else None
    if ttl_config is not None:
        return ThreadTTLConfig.model_validate(ttl_config)

    return None
