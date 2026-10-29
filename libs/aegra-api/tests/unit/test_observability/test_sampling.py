"""Tests for ratio-sampler bias and run_id→trace_id seeding policy."""

from __future__ import annotations

import logging
import uuid

import pytest
from opentelemetry.sdk.trace.sampling import Decision, TraceIdRatioBased

from aegra_api.observability.sampling import (
    configure_trace_id_from_run_id,
    is_ratio_sampler_configured,
    is_trace_id_from_run_id_enabled,
    reset_trace_id_seeding_for_tests,
)
from aegra_api.observability.span_enrichment import _run_trace_id, prepare_run_otel_trace


class TestUuidTraceIdRatioBias:
    """Stock TraceIdRatioBased is unusable with UUID(run_id).int trace ids."""

    def test_stock_ratio_01_drops_every_uuid4_trace_id(self) -> None:
        sampler = TraceIdRatioBased(0.1)
        for _ in range(200):
            result = sampler.should_sample(None, uuid.uuid4().int, "span")
            assert result.decision is Decision.DROP

    def test_stock_ratio_05_drops_every_uuid4_trace_id(self) -> None:
        sampler = TraceIdRatioBased(0.5)
        for _ in range(200):
            result = sampler.should_sample(None, uuid.uuid4().int, "span")
            assert result.decision is Decision.DROP

    def test_stock_ratio_075_samples_every_uuid4_trace_id(self) -> None:
        sampler = TraceIdRatioBased(0.75)
        for _ in range(200):
            result = sampler.should_sample(None, uuid.uuid4().int, "span")
            assert result.decision is Decision.RECORD_AND_SAMPLE


class TestTraceIdFromRunIdPolicy:
    def setup_method(self) -> None:
        reset_trace_id_seeding_for_tests()
        _run_trace_id.set(None)

    def teardown_method(self) -> None:
        reset_trace_id_seeding_for_tests()
        _run_trace_id.set(None)

    def test_enabled_when_settings_true_and_no_ratio_sampler(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_TRACES_SAMPLER", raising=False)
        assert configure_trace_id_from_run_id(enabled_by_settings=True) is True
        assert is_trace_id_from_run_id_enabled() is True

    def test_disabled_when_settings_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_TRACES_SAMPLER", raising=False)
        assert configure_trace_id_from_run_id(enabled_by_settings=False) is False
        assert is_trace_id_from_run_id_enabled() is False

    def test_ratio_sampler_disables_seeding_and_warns(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("OTEL_TRACES_SAMPLER", "parentbased_traceidratio")
        monkeypatch.setenv("OTEL_TRACES_SAMPLER_ARG", "0.1")
        assert is_ratio_sampler_configured() is True
        with caplog.at_level(logging.WARNING, logger="aegra_api.observability.sampling"):
            assert configure_trace_id_from_run_id(enabled_by_settings=True) is False
        assert is_trace_id_from_run_id_enabled() is False
        assert any("ratio sampler" in r.message for r in caplog.records)

    def test_prepare_skips_seed_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_TRACES_SAMPLER", raising=False)
        configure_trace_id_from_run_id(enabled_by_settings=False)
        run_id = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"
        prepare_run_otel_trace(run_id)
        assert _run_trace_id.get() is None

    def test_prepare_seeds_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_TRACES_SAMPLER", raising=False)
        configure_trace_id_from_run_id(enabled_by_settings=True)
        run_id = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"
        prepare_run_otel_trace(run_id)
        assert _run_trace_id.get() == uuid.UUID(run_id).int
        _run_trace_id.set(None)
