"""Unit tests for the Langfuse observability target."""

import base64
from unittest.mock import patch

from aegra_api.observability.targets.langfuse import LangfuseTarget


class TestLangfuseTarget:
    """Tests for LangfuseTarget configuration and exporter creation."""

    def test_target_name(self):
        """Test that the target has the correct friendly name."""
        target = LangfuseTarget()
        assert target.name == "Langfuse"

    def test_get_exporter_missing_credentials(self):
        """Test that exporter is None if public key or secret key is missing."""
        with patch("aegra_api.observability.targets.langfuse.settings") as mock_settings:
            # Case 1: No Public Key
            mock_settings.observability.LANGFUSE_PUBLIC_KEY = None
            mock_settings.observability.LANGFUSE_SECRET_KEY = "sk-123"

            target = LangfuseTarget()
            assert target.get_exporter() is None

            # Case 2: No Secret Key
            mock_settings.observability.LANGFUSE_PUBLIC_KEY = "pk-123"
            mock_settings.observability.LANGFUSE_SECRET_KEY = ""

            assert target.get_exporter() is None

    def test_get_exporter_success_configuration(self) -> None:
        """Test that exporter is configured correctly with valid credentials."""
        pk = "pk-test-key"
        sk = "sk-test-secret"
        host = "https://cloud.langfuse.com"

        with (
            patch("aegra_api.observability.targets.langfuse.settings") as mock_settings,
            patch("aegra_api.observability.targets.langfuse.OTLPSpanExporter") as mock_exporter_cls,
        ):
            mock_settings.observability.LANGFUSE_PUBLIC_KEY = pk
            mock_settings.observability.LANGFUSE_SECRET_KEY = sk
            mock_settings.observability.LANGFUSE_BASE_URL = host

            target = LangfuseTarget()
            exporter = target.get_exporter()

            assert exporter is mock_exporter_cls.return_value
            call_kwargs = mock_exporter_cls.call_args.kwargs
            assert call_kwargs["endpoint"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"

            # Verify Authorization Header (Basic Auth Base64)
            expected_auth = base64.b64encode(f"{pk}:{sk}".encode()).decode()
            assert call_kwargs["headers"]["Authorization"] == f"Basic {expected_auth}"

    def test_endpoint_slash_handling(self) -> None:
        """Test that trailing slashes in host are handled correctly."""
        host_with_slash = "https://custom.langfuse.host/"

        with (
            patch("aegra_api.observability.targets.langfuse.settings") as mock_settings,
            patch("aegra_api.observability.targets.langfuse.OTLPSpanExporter") as mock_exporter_cls,
        ):
            mock_settings.observability.LANGFUSE_PUBLIC_KEY = "pk"
            mock_settings.observability.LANGFUSE_SECRET_KEY = "sk"
            mock_settings.observability.LANGFUSE_BASE_URL = host_with_slash

            target = LangfuseTarget()
            target.get_exporter()

            # Should not have double slash //api...
            endpoint = mock_exporter_cls.call_args.kwargs["endpoint"]
            assert endpoint == "https://custom.langfuse.host/api/public/otel/v1/traces"

    def test_ingestion_version_header_set(self):
        """Test that x-langfuse-ingestion-version: 4 header is passed to OTLPSpanExporter."""
        with (
            patch("aegra_api.observability.targets.langfuse.settings") as mock_settings,
            patch("aegra_api.observability.targets.langfuse.OTLPSpanExporter") as mock_exporter_cls,
        ):
            mock_settings.observability.LANGFUSE_PUBLIC_KEY = "pk-test"
            mock_settings.observability.LANGFUSE_SECRET_KEY = "sk-test"
            mock_settings.observability.LANGFUSE_BASE_URL = "https://us.cloud.langfuse.com"

            target = LangfuseTarget()
            target.get_exporter()

            call_kwargs = mock_exporter_cls.call_args[1]
            assert call_kwargs["headers"]["x-langfuse-ingestion-version"] == "4"
