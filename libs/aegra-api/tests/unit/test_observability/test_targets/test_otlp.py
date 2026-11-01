"""Unit tests for the Generic (OTLP) observability target."""

from unittest.mock import patch

from aegra_api.observability.targets.otlp import GenericOtelTarget


class TestGenericOtelTarget:
    """Tests for GenericOtelTarget configuration and header parsing."""

    def test_target_name(self):
        """Test that the target has the correct friendly name."""
        target = GenericOtelTarget()
        assert target.name == "GenericOTLP"

    def test_get_exporter_missing_endpoint(self):
        """Test that exporter is None if the OTLP endpoint is not configured."""
        with patch("aegra_api.observability.targets.otlp.settings") as mock_settings:
            mock_settings.observability.OTEL_EXPORTER_OTLP_ENDPOINT = None

            target = GenericOtelTarget()
            assert target.get_exporter() is None

    def test_get_exporter_with_endpoint_only(self) -> None:
        """Test exporter creation with only an endpoint (no headers)."""
        endpoint = "http://jaeger:4318/v1/traces"

        with (
            patch("aegra_api.observability.targets.otlp.settings") as mock_settings,
            patch("aegra_api.observability.targets.otlp.OTLPSpanExporter") as mock_exporter_cls,
        ):
            mock_settings.observability.OTEL_EXPORTER_OTLP_ENDPOINT = endpoint
            mock_settings.observability.OTEL_EXPORTER_OTLP_HEADERS = None

            target = GenericOtelTarget()
            exporter = target.get_exporter()

            assert exporter is mock_exporter_cls.return_value
            mock_exporter_cls.assert_called_once_with(endpoint=endpoint, headers={})

    def test_header_parsing_valid(self) -> None:
        """Test parsing of a valid header string."""
        endpoint = "http://localhost:4318"
        headers_str = "Authorization=Bearer 123, X-Custom=Value"

        with (
            patch("aegra_api.observability.targets.otlp.settings") as mock_settings,
            patch("aegra_api.observability.targets.otlp.OTLPSpanExporter") as mock_exporter_cls,
        ):
            mock_settings.observability.OTEL_EXPORTER_OTLP_ENDPOINT = endpoint
            mock_settings.observability.OTEL_EXPORTER_OTLP_HEADERS = headers_str

            target = GenericOtelTarget()
            target.get_exporter()

            expected_headers = {"Authorization": "Bearer 123", "X-Custom": "Value"}
            assert mock_exporter_cls.call_args.kwargs["headers"] == expected_headers

    def test_header_parsing_malformed_items(self) -> None:
        """Test that malformed header items are skipped gracefully."""
        endpoint = "http://localhost:4318"
        # 'InvalidItem' lacks '=', so it should be skipped
        headers_str = "Key1=Value1, InvalidItem, Key2=Value2"

        with (
            patch("aegra_api.observability.targets.otlp.settings") as mock_settings,
            patch("aegra_api.observability.targets.otlp.OTLPSpanExporter") as mock_exporter_cls,
        ):
            mock_settings.observability.OTEL_EXPORTER_OTLP_ENDPOINT = endpoint
            mock_settings.observability.OTEL_EXPORTER_OTLP_HEADERS = headers_str

            target = GenericOtelTarget()
            target.get_exporter()

            # Should contain valid pairs, invalid one ignored
            expected_headers = {"Key1": "Value1", "Key2": "Value2"}
            assert mock_exporter_cls.call_args.kwargs["headers"] == expected_headers

    def test_header_parsing_empty_string(self) -> None:
        """Test that empty string results in empty headers."""
        endpoint = "http://localhost"

        with (
            patch("aegra_api.observability.targets.otlp.settings") as mock_settings,
            patch("aegra_api.observability.targets.otlp.OTLPSpanExporter") as mock_exporter_cls,
        ):
            mock_settings.observability.OTEL_EXPORTER_OTLP_ENDPOINT = endpoint
            mock_settings.observability.OTEL_EXPORTER_OTLP_HEADERS = ""

            target = GenericOtelTarget()
            target.get_exporter()

            assert mock_exporter_cls.call_args.kwargs["headers"] == {}
