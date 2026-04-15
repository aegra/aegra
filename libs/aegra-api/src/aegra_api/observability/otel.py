"""
Unified OpenTelemetry Provider.
Orchestrates trace generation and fan-out export to multiple targets.
"""

import logging
from typing import Any

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from aegra_api.observability.base import ObservabilityProvider
from aegra_api.observability.span_enrichment import SpanEnrichmentProcessor
from aegra_api.observability.targets import (
    BaseOtelTarget,
    GenericOtelTarget,
    LangfuseTarget,
    PhoenixTarget,
)
from aegra_api.settings import settings

logger = logging.getLogger(__name__)


class OpenTelemetryProvider(ObservabilityProvider):
    """
    Main provider that configures the global OpenTelemetry Tracer.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._tracer_provider: TracerProvider | None = None

        # Defining the list of active targets
        self._active_targets: list[BaseOtelTarget] = self._resolve_targets()
        self._has_langfuse = any(isinstance(target, LangfuseTarget) for target in self._active_targets)

        if self._active_targets or settings.observability.OTEL_CONSOLE_EXPORT:
            self._enabled = True

    def is_enabled(self) -> bool:
        return self._enabled

    def _resolve_targets(self) -> list[BaseOtelTarget]:
        targets: list[BaseOtelTarget] = []
        raw_targets = settings.observability.OTEL_TARGETS

        if not raw_targets:
            return targets

        for name in raw_targets.split(","):
            name_clean = name.strip().upper()
            if not name_clean:
                continue

            if name_clean == "LANGFUSE":
                targets.append(LangfuseTarget())
            elif name_clean == "PHOENIX":
                targets.append(PhoenixTarget())
            elif name_clean in ("GENERIC", "DEFAULT", "OTLP"):
                targets.append(GenericOtelTarget())
            else:
                logger.warning(f"Unknown OTEL target in settings: {name_clean}")

        return targets

    def add_custom_target(self, target: BaseOtelTarget) -> None:
        """Allow registering custom targets dynamically."""
        self._active_targets.append(target)
        if isinstance(target, LangfuseTarget):
            self._has_langfuse = True
        self._enabled = True

        if self._tracer_provider is not None:
            try:
                exporter = target.get_exporter()
                if exporter:
                    self._tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                    logger.info(f"Observability: Attached target '{target.name}'")
            except Exception as e:
                logger.error(f"Observability: Failed to attach target '{target.name}': {e}")

    def setup(self) -> None:
        """Initializes the Global Tracer Provider. Runs once."""
        if self._tracer_provider:
            return

        # 1. Resource
        resource = Resource.create(
            attributes={
                "service.name": settings.observability.OTEL_SERVICE_NAME,
                "service.version": settings.app.VERSION,
                "deployment.environment": settings.app.ENV_MODE.lower(),
            }
        )

        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(SpanEnrichmentProcessor())
        processors_count = 0

        # 2. Attach Exporters
        for target in self._active_targets:
            try:
                exporter = target.get_exporter()
                if exporter:
                    processor = BatchSpanProcessor(exporter)
                    self._tracer_provider.add_span_processor(processor)
                    processors_count += 1
                    logger.info(f"Observability: Attached target '{target.name}'")
            except Exception as e:
                logger.error(f"Observability: Failed to attach target '{target.name}': {e}")

        # 3. Console Exporter
        if settings.observability.OTEL_CONSOLE_EXPORT:
            self._tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
            processors_count += 1
            logger.info("Observability: Console export enabled")

        # 4. Set Global Tracer & Instrument
        if processors_count > 0:
            trace.set_tracer_provider(self._tracer_provider)
            LangChainInstrumentor().instrument(tracer_provider=self._tracer_provider)
            logger.info("Observability: Auto-instrumentation enabled")

    def get_callbacks(self) -> list[Any]:
        if self.is_enabled():
            self.setup()
        return []

    def get_metadata(self, run_id: str, thread_id: str, user_identity: str | None = None) -> dict[str, Any]:
        if not self.is_enabled():
            return {}

        meta: dict[str, Any] = {
            "run_id": run_id,
            "thread_id": thread_id,
            "session_id": thread_id,
        }
        if user_identity:
            meta["user_id"] = user_identity

        if self._has_langfuse:
            # Langfuse CallbackHandler only promotes langfuse_* prefixed keys
            # to trace-level fields; plain session_id stays in generic metadata.
            meta["langfuse_session_id"] = thread_id
            if user_identity:
                meta["langfuse_user_id"] = user_identity

        return meta


otel_provider = OpenTelemetryProvider()
