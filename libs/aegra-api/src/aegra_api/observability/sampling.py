"""Disable run_id→trace_id seeding when a ratio sampler would see biased UUID bits."""

from __future__ import annotations

import logging
import os
from functools import cache

from opentelemetry.sdk.trace.sampling import OTEL_TRACES_SAMPLER

from aegra_api.settings import settings

logger = logging.getLogger(__name__)

_RATIO_SAMPLERS = frozenset({"traceidratio", "parentbased_traceidratio"})


def is_ratio_sampler_configured() -> bool:
    name = os.getenv(OTEL_TRACES_SAMPLER, "parentbased_always_on").lower()
    return name in _RATIO_SAMPLERS


@cache
def is_trace_id_from_run_id_enabled() -> bool:
    """Whether run_id→trace_id seeding is active. Cached; call once at OTEL setup."""
    if not settings.observability.OTEL_TRACE_ID_FROM_RUN_ID:
        return False
    if is_ratio_sampler_configured():
        logger.warning(
            "OTEL_TRACE_ID_FROM_RUN_ID disabled: ratio sampler "
            "(traceidratio / parentbased_traceidratio) is incompatible with "
            "UUID(run_id)-seeded trace ids (biased low-64 bits). Using random "
            "trace ids so sampling rates apply correctly."
        )
        return False
    return True
