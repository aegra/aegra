"""Shared E2E test helpers that can be imported across packages."""

from __future__ import annotations

import json
from typing import Any

import pytest


def elog(title: str, payload: Any) -> None:
    """Emit pretty JSON logs for E2E visibility."""
    try:
        formatted = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        formatted = str(payload)
    print(f"\n=== {title} ===\n{formatted}\n")


def check_and_skip_if_geo_blocked(run_data: dict[str, Any]) -> None:
    """Skip tests when failures match the known geo-blocking signature."""
    if run_data.get("status") != "error":
        return

    message = str(run_data.get("error_message", "")).lower()
    if "unsupported_country_region_territory" in message or "generator didn't stop" in message:
        pytest.skip(f"⛔️ Skipped: OpenAI Geo-block detected. ({message[:60]}...)")
