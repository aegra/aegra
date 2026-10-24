"""Disable run_id→trace_id seeding when a ratio sampler would see biased UUID bits."""

from __future__ import annotations

import logging
import os

from opentelemetry.sdk.trace.sampling import OTEL_TRACES_SAMPLER

logger = logging.getLogger(__name__)

_RATIO_SAMPLERS = frozenset({"traceidratio", "parentbased_traceidratio"})
_seeding_enabled: bool | None = None


def is_ratio_sampler_configured() -> bool:
    name = os.getenv(OTEL_TRACES_SAMPLER, "parentbased_always_on").lower()
    return name in _RATIO_SAMPLERS


def configure_trace_id_from_run_id(*, enabled_by_settings: bool) -> bool:
    """Resolve seeding; warn once if a ratio sampler forces random trace ids."""
    global _seeding_enabled
    if not enabled_by_settings:
        _seeding_enabled = False
        return False
    if is_ratio_sampler_configured():
        _seeding_enabled = False
        logger.warning(
            "OTEL_TRACE_ID_FROM_RUN_ID disabled: ratio sampler "
            "(traceidratio / parentbased_traceidratio) is incompatible with "
            "UUID(run_id)-seeded trace ids (biased low-64 bits). Using random "
            "trace ids so sampling rates apply correctly."
        )
        return False
    _seeding_enabled = True
    return True


def is_trace_id_from_run_id_enabled() -> bool:
    global _seeding_enabled
    if _seeding_enabled is None:
        from aegra_api.settings import settings

        configure_trace_id_from_run_id(enabled_by_settings=settings.observability.OTEL_TRACE_ID_FROM_RUN_ID)
    return bool(_seeding_enabled)


def reset_trace_id_seeding_for_tests() -> None:
    global _seeding_enabled
    _seeding_enabled = None
