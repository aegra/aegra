"""Unit tests for SpanEnrichmentProcessor and set_trace_context."""

import asyncio
import logging
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from aegra_api.observability.span_enrichment import (
    SpanEnrichmentProcessor,
    _trace_attrs,
    make_run_trace_context,
    merge_run_metadata,
    set_trace_context,
)


class TestSetTraceContext:
    """Tests for the set_trace_context() helper."""

    def setup_method(self) -> None:
        """Reset context var before each test."""
        _trace_attrs.set(None)

    def test_sets_all_attributes_when_all_provided(self) -> None:
        """All provided values appear in the context var under both naming schemes."""
        set_trace_context(user_id="user-1", session_id="thread-1", trace_name="my_graph")

        attrs = _trace_attrs.get()
        assert attrs["langfuse.user.id"] == "user-1"
        assert attrs["user.id"] == "user-1"
        assert attrs["langfuse.session.id"] == "thread-1"
        assert attrs["session.id"] == "thread-1"
        assert attrs["langfuse.trace.name"] == "my_graph"

    def test_skips_none_values(self) -> None:
        """None arguments are not stored; only provided values appear."""
        set_trace_context(user_id="user-1")

        attrs = _trace_attrs.get()
        assert "langfuse.user.id" in attrs
        assert "user.id" in attrs
        assert "langfuse.session.id" not in attrs
        assert "langfuse.trace.name" not in attrs

    def test_empty_call_stores_none(self) -> None:
        """Calling with no arguments resets the context var to None (no-op state)."""
        set_trace_context(user_id="previous")
        set_trace_context()

        assert _trace_attrs.get() is None

    def test_only_trace_name_set(self) -> None:
        """Only trace_name provided — only that key is stored."""
        set_trace_context(trace_name="matter_agent")

        attrs = _trace_attrs.get()
        assert attrs == {"langfuse.trace.name": "matter_agent"}

    def test_metadata_stored_with_langfuse_prefix(self) -> None:
        """metadata dict keys are stored as langfuse.trace.metadata.<key>."""
        set_trace_context(
            user_id="u1",
            metadata={"run_id": "run-abc", "graph_id": "matter_agent"},
        )

        attrs = _trace_attrs.get()
        assert attrs["langfuse.trace.metadata.run_id"] == "run-abc"
        assert attrs["langfuse.trace.metadata.graph_id"] == "matter_agent"
        # first-class attrs still present
        assert attrs["langfuse.user.id"] == "u1"

    def test_empty_metadata_dict_ignored(self) -> None:
        """Passing metadata={} produces no extra keys."""
        set_trace_context(trace_name="g", metadata={})

        attrs = _trace_attrs.get()
        assert attrs == {"langfuse.trace.name": "g"}

    def test_metadata_supports_non_string_values(self) -> None:
        """metadata values may be int, float, or bool — all valid OTEL attribute types."""
        set_trace_context(
            metadata={"retry_count": 3, "latency_ms": 1.5, "cached": True},
        )

        attrs = _trace_attrs.get()
        assert attrs["langfuse.trace.metadata.retry_count"] == 3
        assert attrs["langfuse.trace.metadata.latency_ms"] == 1.5
        assert attrs["langfuse.trace.metadata.cached"] is True

    @pytest.mark.asyncio
    async def test_context_var_isolation_between_tasks(self) -> None:
        """Context var changes in one asyncio Task are not visible in another."""

        async def task_a() -> dict[str, str]:
            set_trace_context(user_id="user-a")
            await asyncio.sleep(0)
            return _trace_attrs.get()

        async def task_b() -> dict[str, str]:
            set_trace_context(user_id="user-b")
            await asyncio.sleep(0)
            return _trace_attrs.get()

        t_a = asyncio.create_task(task_a())
        t_b = asyncio.create_task(task_b())
        attrs_a, attrs_b = await asyncio.gather(t_a, t_b)
        assert attrs_a.get("user.id") == "user-a"
        assert attrs_b.get("user.id") == "user-b"


class TestSpanEnrichmentProcessor:
    """Tests for SpanEnrichmentProcessor."""

    def setup_method(self) -> None:
        """Reset context var before each test."""
        _trace_attrs.set(None)

    def test_on_start_sets_span_attributes_on_root_span(self) -> None:
        """on_start() enriches a root span (parent=None) with all context var attrs."""
        set_trace_context(user_id="u1", session_id="s1", trace_name="graph_x")
        processor = SpanEnrichmentProcessor()
        mock_span = MagicMock()
        mock_span.parent = None  # root span

        processor.on_start(mock_span)

        calls = {call.args[0]: call.args[1] for call in mock_span.set_attribute.call_args_list}
        assert calls["langfuse.user.id"] == "u1"
        assert calls["user.id"] == "u1"
        assert calls["langfuse.session.id"] == "s1"
        assert calls["session.id"] == "s1"
        assert calls["langfuse.trace.name"] == "graph_x"

    def test_on_start_skips_local_child_spans(self) -> None:
        """on_start() does NOT enrich local child spans (valid, non-remote parent)."""
        set_trace_context(user_id="u1", session_id="s1", trace_name="graph_x")
        processor = SpanEnrichmentProcessor()
        mock_span = MagicMock()
        mock_span.parent = MagicMock()
        mock_span.parent.is_valid = True
        mock_span.parent.is_remote = False  # local child span

        processor.on_start(mock_span)

        mock_span.set_attribute.assert_not_called()

    def test_on_start_enriches_span_with_remote_parent(self) -> None:
        """on_start() enriches spans whose parent arrived via W3C traceparent.

        A span with a remote parent is the local root of a distributed trace
        and must be enriched so that Langfuse receives user/session metadata.
        """
        set_trace_context(user_id="u1", trace_name="graph_x")
        processor = SpanEnrichmentProcessor()
        mock_span = MagicMock()
        mock_span.parent = MagicMock()
        mock_span.parent.is_valid = True
        mock_span.parent.is_remote = True  # arrived via traceparent header

        processor.on_start(mock_span)

        mock_span.set_attribute.assert_called()

    def test_on_start_no_op_when_context_var_empty(self) -> None:
        """on_start() sets no attributes when the context var holds an empty dict."""
        processor = SpanEnrichmentProcessor()
        mock_span = MagicMock()
        mock_span.parent = None

        processor.on_start(mock_span)

        mock_span.set_attribute.assert_not_called()

    def test_on_start_accepts_parent_context_argument(self) -> None:
        """on_start() can be called with an explicit parent_context without error."""
        set_trace_context(user_id="u2")
        processor = SpanEnrichmentProcessor()
        mock_span = MagicMock()
        mock_span.parent = None
        mock_ctx = MagicMock()

        processor.on_start(mock_span, parent_context=mock_ctx)

        mock_span.set_attribute.assert_called()

    def test_on_end_is_no_op(self) -> None:
        """on_end() completes without raising."""
        processor = SpanEnrichmentProcessor()
        processor.on_end(MagicMock())  # Should not raise

    def test_force_flush_returns_true(self) -> None:
        """force_flush() returns True unconditionally."""
        assert SpanEnrichmentProcessor().force_flush() is True
        assert SpanEnrichmentProcessor().force_flush(timeout_millis=100) is True

    def test_shutdown_is_no_op(self) -> None:
        """shutdown() completes without raising."""
        SpanEnrichmentProcessor().shutdown()  # Should not raise


class TestMakeRunTraceContext:
    """Tests for make_run_trace_context()."""

    def setup_method(self) -> None:
        """Reset context var before each test."""
        _trace_attrs.set(None)

    _RUN_ID = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"

    def test_returned_context_contains_expected_attributes(self) -> None:
        """Returned context has all trace attributes pre-set."""
        ctx = make_run_trace_context(self._RUN_ID, "thread-1", "my_graph", "user-1")

        attrs = ctx.run(_trace_attrs.get)
        assert attrs["langfuse.user.id"] == "user-1"
        assert attrs["langfuse.session.id"] == "thread-1"
        assert attrs["langfuse.trace.name"] == "my_graph"
        assert attrs["langfuse.trace.metadata.run_id"] == self._RUN_ID
        assert attrs["langfuse.trace.metadata.thread_id"] == "thread-1"
        assert attrs["langfuse.trace.metadata.graph_id"] == "my_graph"

    def test_does_not_pollute_caller_context(self) -> None:
        """Calling make_run_trace_context() does not mutate the caller's context."""
        make_run_trace_context(self._RUN_ID, "thread-1", "my_graph", "user-1")

        assert _trace_attrs.get() is None

    def test_anonymous_user_omits_user_attributes(self) -> None:
        """Passing user_identity=None omits user.id keys from the context."""
        ctx = make_run_trace_context(self._RUN_ID, "thread-1", "my_graph", None)

        attrs = ctx.run(_trace_attrs.get)
        assert "langfuse.user.id" not in attrs
        assert "user.id" not in attrs
        assert attrs["langfuse.trace.name"] == "my_graph"

    def test_seeds_otel_trace_id_from_run_id(self) -> None:
        """The returned context has an OTEL trace_id derived from run_id."""
        import uuid

        from opentelemetry import trace

        ctx = make_run_trace_context(self._RUN_ID, "thread-1", "my_graph", "user-1")

        span = ctx.run(trace.get_current_span)
        trace_id = span.get_span_context().trace_id
        assert trace_id == uuid.UUID(self._RUN_ID).int

    def test_extra_metadata_merged_into_trace_context(self) -> None:
        """User-supplied extra_metadata appears alongside system runtime keys."""
        ctx = make_run_trace_context(
            self._RUN_ID,
            "thread-1",
            "my_graph",
            "user-1",
            extra_metadata={"tenant": "acme", "feature_flag": True},
        )

        attrs = ctx.run(_trace_attrs.get)
        assert attrs["langfuse.trace.metadata.tenant"] == "acme"
        assert attrs["langfuse.trace.metadata.feature_flag"] is True
        # System keys preserved
        assert attrs["langfuse.trace.metadata.run_id"] == self._RUN_ID
        assert attrs["langfuse.trace.metadata.thread_id"] == "thread-1"
        assert attrs["langfuse.trace.metadata.graph_id"] == "my_graph"

    def test_extra_metadata_cannot_override_system_keys(self) -> None:
        """Reserved system keys win on collision; user value is dropped."""
        ctx = make_run_trace_context(
            self._RUN_ID,
            "thread-1",
            "my_graph",
            "user-1",
            extra_metadata={"run_id": "run-spoofed", "tenant": "acme"},
        )

        attrs = ctx.run(_trace_attrs.get)
        assert attrs["langfuse.trace.metadata.run_id"] == self._RUN_ID
        assert attrs["langfuse.trace.metadata.tenant"] == "acme"


class TestMergeRunMetadata:
    """Tests for merge_run_metadata()."""

    def test_returns_system_only_when_extra_is_none(self) -> None:
        system = {"run_id": "r1", "thread_id": "t1"}
        assert merge_run_metadata(None, system) == system

    def test_returns_system_only_when_extra_is_empty(self) -> None:
        system = {"run_id": "r1", "thread_id": "t1"}
        assert merge_run_metadata({}, system) == system

    def test_user_keys_added_when_no_collision(self) -> None:
        system = {"run_id": "r1", "thread_id": "t1"}
        result = merge_run_metadata({"tenant": "acme", "retries": 3}, system)
        assert result == {"run_id": "r1", "thread_id": "t1", "tenant": "acme", "retries": 3}

    def test_system_wins_on_collision(self) -> None:
        system = {"run_id": "actual", "thread_id": "t1"}
        result = merge_run_metadata({"run_id": "spoofed", "tenant": "acme"}, system)
        assert result["run_id"] == "actual"
        assert result["tenant"] == "acme"

    def test_collision_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A user-supplied reserved key triggers a warning log."""
        with caplog.at_level(logging.WARNING, logger="aegra_api.observability.span_enrichment"):
            merge_run_metadata({"thread_id": "spoofed"}, {"thread_id": "actual"})

        assert any("thread_id" in record.message for record in caplog.records)

    def test_no_warning_when_keys_do_not_collide(self, caplog: pytest.LogCaptureFixture) -> None:
        """No warning of any kind is emitted when user keys do not hit reserved names."""
        with caplog.at_level(logging.WARNING, logger="aegra_api.observability.span_enrichment"):
            merge_run_metadata({"tenant": "acme"}, {"run_id": "r1"})

        # Stricter than a substring check: assert no WARNING-level record was
        # emitted from the merge_run_metadata logger at all.
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_original_request_id_passes_through_when_system_lacks_it(self) -> None:
        """``original_request_id`` is set only on the worker path; the local-executor
        path omits it. Reserving it would silently drop user values when the
        system value is missing — confirm it flows through as a regular key."""
        system = {"run_id": "r1", "thread_id": "t1", "graph_id": "g1"}
        result = merge_run_metadata({"original_request_id": "user-corr-id"}, system)
        assert result["original_request_id"] == "user-corr-id"

    def test_non_reserved_system_collision_drops_user_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A user value that collides with a non-reserved system key is dropped
        with a warning. Without this, the worker path silently overwrote
        user-supplied ``original_request_id`` (and any other future system
        injection) via ``merged.update(system_metadata)``."""
        system = {
            "run_id": "r1",
            "thread_id": "t1",
            "graph_id": "g1",
            "original_request_id": "system-corr-id",
        }
        with caplog.at_level(logging.WARNING, logger="aegra_api.observability.span_enrichment"):
            result = merge_run_metadata({"original_request_id": "user-corr-id"}, system)

        assert result["original_request_id"] == "system-corr-id"
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("original_request_id" in w.message for w in warnings)

    def test_non_primitive_value_is_dropped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """OTEL span attributes accept only primitives; nested values are
        dropped at this layer with an aegra-level warning so the loss is
        visible (rather than swallowed inside the OTEL SDK)."""
        system = {"run_id": "r1"}
        with caplog.at_level(logging.WARNING, logger="aegra_api.observability.span_enrichment"):
            result = merge_run_metadata(
                {"tenant": "acme", "nested": {"x": 1}, "items": [1, 2, 3]},
                system,
            )

        assert result["tenant"] == "acme"
        assert "nested" not in result
        assert "items" not in result
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("nested" in w.message for w in warnings)
        assert any("items" in w.message for w in warnings)

    def test_primitive_types_pass_through(self) -> None:
        """str, int, float, bool all flow through unchanged."""
        system: dict[str, str | int | float | bool] = {"run_id": "r1"}
        result = merge_run_metadata(
            {"s": "v", "i": 42, "f": 3.14, "b": True},
            system,
        )
        assert result["s"] == "v"
        assert result["i"] == 42
        assert result["f"] == 3.14
        assert result["b"] is True


class TestSpanEnrichmentEndToEnd:
    """End-to-end verification with the real OpenTelemetry SDK.

    The other tests in this file mock the SDK to isolate behavior.  This
    class wires up an actual ``TracerProvider`` with an
    ``InMemorySpanExporter`` so the full chain
    ``make_run_trace_context`` → context var →
    ``SpanEnrichmentProcessor.on_start`` → root-span attributes is
    exercised without mocks.

    Without this coverage, a refactor that quietly broke any link in the
    chain (e.g. a dropped ``extra_metadata`` kwarg, a wrongly-passed
    ``contextvars.Context`` to ``asyncio.create_task``, an
    ``on_start``-vs-``on_end`` ordering regression) would still pass
    every other test in the file.
    """

    _RUN_ID = "aaaaaaaa-1111-2222-3333-bbbbbbbbbbbb"
    _RUN_ID_2 = "bbbbbbbb-2222-3333-4444-cccccccccccc"

    def setup_method(self) -> None:
        _trace_attrs.set(None)
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        # Order matters: enrichment must run BEFORE export so the
        # exporter sees attributes set by ``on_start``.
        self.provider.add_span_processor(SpanEnrichmentProcessor())
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.tracer = self.provider.get_tracer(__name__)

    def teardown_method(self) -> None:
        self.provider.shutdown()
        _trace_attrs.set(None)

    def _emit_root_span(self, ctx) -> None:
        """Open and close a root span inside the supplied context."""

        def _create() -> None:
            with self.tracer.start_as_current_span("test_root_span"):
                pass

        ctx.run(_create)

    def test_user_metadata_reaches_root_span_attributes(self) -> None:
        """Happy path: every layer of the propagation pipeline cooperates."""
        ctx = make_run_trace_context(
            run_id=self._RUN_ID,
            thread_id="thread-1",
            graph_id="my_graph",
            user_identity="user-1",
            extra_metadata={"tenant": "acme", "retries": 3, "ratio": 0.5, "flag": True},
        )

        self._emit_root_span(ctx)

        spans = self.exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})

        # Langfuse-native + Phoenix/OpenInference aliases for first-class fields.
        assert attrs["langfuse.user.id"] == "user-1"
        assert attrs["user.id"] == "user-1"
        assert attrs["langfuse.session.id"] == "thread-1"
        assert attrs["session.id"] == "thread-1"
        assert attrs["langfuse.trace.name"] == "my_graph"

        # System metadata is always present.
        assert attrs["langfuse.trace.metadata.run_id"] == self._RUN_ID
        assert attrs["langfuse.trace.metadata.thread_id"] == "thread-1"
        assert attrs["langfuse.trace.metadata.graph_id"] == "my_graph"

        # User-supplied metadata flows through with primitive types preserved.
        assert attrs["langfuse.trace.metadata.tenant"] == "acme"
        assert attrs["langfuse.trace.metadata.retries"] == 3
        assert attrs["langfuse.trace.metadata.ratio"] == 0.5
        assert attrs["langfuse.trace.metadata.flag"] is True

    def test_reserved_collision_system_value_wins_on_root_span(self) -> None:
        """User attempt to overwrite a reserved key is dropped; non-collision
        keys still flow through.  This is the contract surfaced via the
        warning in ``merge_run_metadata`` — verified end-to-end here."""
        ctx = make_run_trace_context(
            run_id=self._RUN_ID_2,
            thread_id="thread-1",
            graph_id="my_graph",
            user_identity="user-1",
            extra_metadata={"run_id": "spoofed", "tenant": "acme"},
        )

        self._emit_root_span(ctx)

        attrs = dict(self.exporter.get_finished_spans()[0].attributes or {})
        assert attrs["langfuse.trace.metadata.run_id"] == self._RUN_ID_2
        assert attrs["langfuse.trace.metadata.tenant"] == "acme"

    def test_anonymous_user_with_no_extra_metadata(self) -> None:
        """``user_identity=None`` + ``extra_metadata=None`` produces a span
        with system metadata only and no user.id keys at all."""
        ctx = make_run_trace_context(
            run_id=self._RUN_ID,
            thread_id="t1",
            graph_id="g1",
            user_identity=None,
            extra_metadata=None,
        )

        self._emit_root_span(ctx)

        attrs = dict(self.exporter.get_finished_spans()[0].attributes or {})
        assert "langfuse.user.id" not in attrs
        assert "user.id" not in attrs
        assert attrs["langfuse.session.id"] == "t1"
        assert attrs["langfuse.trace.name"] == "g1"
        assert attrs["langfuse.trace.metadata.run_id"] == self._RUN_ID
        # No leftover metadata keys from a previous run/test.
        user_metadata_keys = [
            k
            for k in attrs
            if k.startswith("langfuse.trace.metadata.") and k.split(".")[-1] not in {"run_id", "thread_id", "graph_id"}
        ]
        assert user_metadata_keys == []

    def test_non_primitive_user_metadata_is_dropped_before_span(self) -> None:
        """A nested value submitted in ``extra_metadata`` is filtered by
        ``merge_run_metadata`` and never reaches the OTEL SDK — so the
        span carries the surviving primitive keys but not the nested
        value (which would otherwise be silently no-op'd inside
        ``span.set_attribute``)."""
        ctx = make_run_trace_context(
            run_id=self._RUN_ID,
            thread_id="t1",
            graph_id="g1",
            user_identity="u1",
            extra_metadata={"tenant": "acme", "nested": {"x": 1}},
        )

        self._emit_root_span(ctx)

        attrs = dict(self.exporter.get_finished_spans()[0].attributes or {})
        assert attrs["langfuse.trace.metadata.tenant"] == "acme"
        assert "langfuse.trace.metadata.nested" not in attrs
