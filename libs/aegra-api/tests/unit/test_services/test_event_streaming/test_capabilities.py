"""Tests for the v2 event streaming capability probe and feature flag."""

from collections.abc import Iterator

import pytest

from aegra_api.services.event_streaming import capabilities as caps


@pytest.fixture(autouse=True)
def _clear_probe_cache() -> Iterator[None]:
    """Reset the memoised symbol probe around each test."""
    caps._probe_runtime_symbols.cache_clear()
    yield
    caps._probe_runtime_symbols.cache_clear()


def test_disabled_by_flag_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the flag off, capabilities report disabled regardless of runtime."""
    monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", False)
    result = caps.get_v2_capabilities()
    assert result.ok is False
    assert result.disabled_by_flag is True
    assert "FF_V2_EVENT_STREAMING" in result.error_message


def test_supported_when_flag_on_and_symbols_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """On our pinned stack the required symbols exist, so v2 is supported."""
    monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", True)
    result = caps.get_v2_capabilities()
    assert result.ok is True
    assert result.missing == ()
    assert result.error_message == ""


def test_unsupported_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing runtime symbol yields an unsupported result with an upgrade hint."""
    monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", True)
    monkeypatch.setattr(
        caps,
        "_REQUIRED_SYMBOLS",
        (("langgraph.pregel._messages", "ThisSymbolDoesNotExist"),),
    )
    caps._probe_runtime_symbols.cache_clear()
    result = caps.get_v2_capabilities()
    assert result.ok is False
    assert result.disabled_by_flag is False
    assert "ThisSymbolDoesNotExist" in result.error_message
    assert "Upgrade" in result.error_message


def test_unsupported_when_module_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing module is reported the same as a missing attribute."""
    monkeypatch.setattr(caps.settings.event_streaming, "FF_V2_EVENT_STREAMING", True)
    monkeypatch.setattr(
        caps,
        "_REQUIRED_SYMBOLS",
        (("langgraph.this_module_is_not_real", "X"),),
    )
    caps._probe_runtime_symbols.cache_clear()
    result = caps.get_v2_capabilities()
    assert result.ok is False
    assert "langgraph.this_module_is_not_real.X" in result.missing
