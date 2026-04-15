"""Unit tests for the OpenTelemetry provider."""

from unittest.mock import MagicMock, patch

import pytest

from aegra_api.observability.otel import OpenTelemetryProvider
from aegra_api.observability.targets import (
    BaseOtelTarget,
    GenericOtelTarget,
    LangfuseTarget,
    PhoenixTarget,
)


class TestOpenTelemetryProviderInit:
    """Tests for initialization and target resolution logic."""

    def test_init_disabled_by_default(self):
        """Test that provider is disabled when no targets are configured."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            assert provider.is_enabled() is False
            assert len(provider._active_targets) == 0

    def test_init_parses_targets_correctly(self):
        """Test that known targets are parsed and instantiated."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = "LANGFUSE, PHOENIX, GENERIC"
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            assert provider.is_enabled() is True
            assert len(provider._active_targets) == 3

            target_types = {type(t) for t in provider._active_targets}
            assert LangfuseTarget in target_types
            assert PhoenixTarget in target_types
            assert GenericOtelTarget in target_types

    def test_init_handles_whitespace_and_casing(self):
        """Test that parsing is robust to whitespace and case sensitivity."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = "  langfuse ,  phoenix "
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            assert len(provider._active_targets) == 2
            target_types = {type(t) for t in provider._active_targets}
            assert LangfuseTarget in target_types
            assert PhoenixTarget in target_types

    def test_init_ignores_unknown_targets(self):
        """Test that unknown targets are logged and ignored."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = "LANGFUSE, UNKNOWN_VENDOR"
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            with patch("aegra_api.observability.otel.logger") as mock_logger:
                provider = OpenTelemetryProvider()

                assert len(provider._active_targets) == 1
                assert isinstance(provider._active_targets[0], LangfuseTarget)

                # Verify warning was logged
                mock_logger.warning.assert_called_with("Unknown OTEL target in settings: UNKNOWN_VENDOR")

    def test_init_enables_console_export(self):
        """Test that console export enables the provider even without targets."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            mock_settings.observability.OTEL_CONSOLE_EXPORT = True

            provider = OpenTelemetryProvider()

            assert provider.is_enabled() is True

    def test_add_custom_target(self):
        """Test dynamically adding a custom target."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            # FIX: Explicitly set False so truthy Mock doesn't trigger it
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()
            assert provider.is_enabled() is False

            mock_target = MagicMock(spec=BaseOtelTarget)
            provider.add_custom_target(mock_target)

            assert provider.is_enabled() is True
            assert mock_target in provider._active_targets

    def test_add_custom_target_attaches_exporter_after_setup(self) -> None:
        """Test that adding a target after setup() wires its exporter immediately."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            mock_settings.observability.OTEL_CONSOLE_EXPORT = True

            provider = OpenTelemetryProvider()
            provider.setup()

            mock_target = MagicMock(spec=BaseOtelTarget)
            mock_target.name = "Late"
            mock_exporter = MagicMock()
            mock_target.get_exporter.return_value = mock_exporter

            provider.add_custom_target(mock_target)

            mock_target.get_exporter.assert_called_once()
            assert provider._tracer_provider is not None


class TestOpenTelemetryProviderSetup:
    """Tests for the setup() method and tracer configuration."""

    @pytest.fixture
    def mock_deps(self):
        """Patch all external OTEL dependencies."""
        with (
            patch("aegra_api.observability.otel.TracerProvider") as mock_tp,
            patch("aegra_api.observability.otel.BatchSpanProcessor") as mock_bsp,
            patch("aegra_api.observability.otel.ConsoleSpanExporter") as mock_cse,
            patch("aegra_api.observability.otel.LangChainInstrumentor") as mock_lci,
            patch("aegra_api.observability.otel.trace") as mock_trace,
            patch("aegra_api.observability.otel.Resource") as mock_resource,
            patch("aegra_api.observability.otel.settings") as mock_settings,
        ):
            # Setup defaults
            mock_settings.observability.OTEL_SERVICE_NAME = "test-service"
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False  # Default to False to prevent noise
            mock_settings.app.VERSION = "1.0.0"
            mock_settings.app.ENV_MODE = "TEST"

            yield {
                "tp": mock_tp,
                "bsp": mock_bsp,
                "cse": mock_cse,
                "lci": mock_lci,
                "trace": mock_trace,
                "resource": mock_resource,
                "settings": mock_settings,
            }

    def test_setup_is_idempotent(self, mock_deps):
        """Test that setup runs only once."""
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = True

        provider = OpenTelemetryProvider()

        # First call
        provider.setup()
        assert mock_deps["tp"].called

        # Reset mocks
        mock_deps["tp"].reset_mock()

        # Second call
        provider.setup()
        assert not mock_deps["tp"].called  # Should not be called again

    def test_setup_creates_correct_resource(self, mock_deps):
        """Test that Resource is created with correct attributes."""
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = True

        provider = OpenTelemetryProvider()
        provider.setup()

        mock_deps["resource"].create.assert_called_with(
            attributes={
                "service.name": "test-service",
                "service.version": "1.0.0",
                "deployment.environment": "test",
            }
        )
        mock_deps["tp"].assert_called_with(resource=mock_deps["resource"].create.return_value)

    def test_setup_attaches_configured_targets(self, mock_deps):
        """Test that exporters from targets are attached to the tracer."""
        # Setup a mock target that returns an exporter
        mock_exporter = MagicMock()
        mock_target = MagicMock(spec=BaseOtelTarget)
        mock_target.get_exporter.return_value = mock_exporter
        mock_target.name = "MockTarget"

        provider = OpenTelemetryProvider()
        # Manually inject target
        provider._active_targets = [mock_target]
        provider._enabled = True

        provider.setup()

        mock_target.get_exporter.assert_called_once()
        mock_deps["bsp"].assert_any_call(mock_exporter)
        tracer_provider_instance = mock_deps["tp"].return_value
        tracer_provider_instance.add_span_processor.assert_called()

    def test_setup_handles_target_errors_gracefully(self, mock_deps):
        """Test that setup continues even if one target fails."""
        # FIX: Ensure console export is disabled for this test to avoid confusion
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = False

        # Target 1 throws exception
        bad_target = MagicMock(spec=BaseOtelTarget)
        bad_target.get_exporter.side_effect = Exception("Config Error")
        bad_target.name = "BadTarget"

        # Target 2 works
        good_exporter = MagicMock()
        good_target = MagicMock(spec=BaseOtelTarget)
        good_target.get_exporter.return_value = good_exporter
        good_target.name = "GoodTarget"

        provider = OpenTelemetryProvider()
        provider._active_targets = [bad_target, good_target]
        provider._enabled = True

        with patch("aegra_api.observability.otel.logger") as mock_logger:
            provider.setup()

            # Should log error for bad target
            mock_logger.error.assert_called()

            # Should still add processor for good target
            tracer_provider_instance = mock_deps["tp"].return_value
            mock_deps["bsp"].assert_called_with(good_exporter)
            # SpanEnrichmentProcessor is added unconditionally + one BatchSpanProcessor
            # for the good target → two calls total
            assert tracer_provider_instance.add_span_processor.call_count == 2

    def test_setup_instruments_globally(self, mock_deps):
        """Test that global tracer and instrumentation are set."""
        mock_deps["settings"].observability.OTEL_CONSOLE_EXPORT = True

        provider = OpenTelemetryProvider()
        provider.setup()

        mock_deps["trace"].set_tracer_provider.assert_called_with(mock_deps["tp"].return_value)

        mock_deps["lci"].return_value.instrument.assert_called_with(tracer_provider=mock_deps["tp"].return_value)


class TestOpenTelemetryProviderRuntime:
    """Tests for runtime methods (get_callbacks, get_metadata)."""

    def test_get_callbacks_triggers_setup(self):
        """Test that get_callbacks calls setup if enabled."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_CONSOLE_EXPORT = True

            provider = OpenTelemetryProvider()

            # Mock setup to verify it's called
            provider.setup = MagicMock()

            callbacks = provider.get_callbacks()

            assert callbacks == []
            provider.setup.assert_called_once()

    def test_get_metadata_returns_correct_structure(self):
        """Test metadata generation when enabled."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_CONSOLE_EXPORT = True

            provider = OpenTelemetryProvider()

            meta = provider.get_metadata(run_id="run-123", thread_id="thread-456", user_identity="user-789")

            assert meta == {
                "run_id": "run-123",
                "thread_id": "thread-456",
                "session_id": "thread-456",
                "user_id": "user-789",
            }

    def test_get_metadata_includes_langfuse_keys_when_langfuse_active(self) -> None:
        """Test that langfuse_* keys are emitted only when Langfuse target is active."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = "LANGFUSE"
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            meta = provider.get_metadata(run_id="run-123", thread_id="thread-456", user_identity="user-789")

            assert meta["langfuse_session_id"] == "thread-456"
            assert meta["langfuse_user_id"] == "user-789"

    def test_get_metadata_langfuse_without_user_identity(self) -> None:
        """Test that langfuse_session_id is present but langfuse_user_id absent when user_identity is None."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = "LANGFUSE"
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            meta = provider.get_metadata(run_id="run-123", thread_id="thread-456")

            assert meta["langfuse_session_id"] == "thread-456"
            assert "langfuse_user_id" not in meta

    def test_get_metadata_excludes_langfuse_keys_when_langfuse_inactive(self) -> None:
        """Test that langfuse_* keys are NOT emitted when only non-Langfuse targets are active."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            mock_settings.observability.OTEL_CONSOLE_EXPORT = True

            provider = OpenTelemetryProvider()

            meta = provider.get_metadata(run_id="run-123", thread_id="thread-456", user_identity="user-789")

            assert "langfuse_session_id" not in meta
            assert "langfuse_user_id" not in meta

    def test_get_metadata_empty_when_disabled(self):
        """Test metadata returns empty dict when disabled."""
        with patch("aegra_api.observability.otel.settings") as mock_settings:
            mock_settings.observability.OTEL_TARGETS = ""
            mock_settings.observability.OTEL_CONSOLE_EXPORT = False

            provider = OpenTelemetryProvider()

            meta = provider.get_metadata("run-1", "thread-1")
            assert meta == {}
