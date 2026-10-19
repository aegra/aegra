"""Unit tests for run-completion webhook delivery."""

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunBehavior, RunExecution, RunIdentity, RunJob
from aegra_api.services import webhook_delivery, webhook_policy
from aegra_api.services.webhook_delivery import (
    RunOutcome,
    WebhookDispatcher,
    _PinnedAddressTransport,
    build_run_payload,
    notify_run_finished,
    redact_webhook_url,
)
from aegra_api.services.webhook_policy import WebhookUrlError, get_webhook_settings, webhooks_enabled

_WEBHOOK = "https://hooks.example.com/aegra?token=secret"
_CREATED_AT = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
_STARTED_AT = datetime(2026, 10, 1, 12, 0, 1, tzinfo=UTC)
_ENDED_AT = datetime(2026, 10, 1, 12, 0, 5, tzinfo=UTC)


def _job(webhook: str | None = _WEBHOOK) -> RunJob:
    return RunJob(
        identity=RunIdentity(
            run_id="run-1",
            thread_id="thread-1",
            graph_id="graph-1",
            assistant_id="assistant-1",
            created_at=_CREATED_AT,
        ),
        user=User(identity="user-1"),
        execution=RunExecution(
            input_data={"messages": [{"role": "user", "content": "hi"}]},
            config={"configurable": {"model": "m"}},
            context={"tenant": "acme"},
            stream_mode=["values"],
            durability="exit",
        ),
        behavior=RunBehavior(interrupt_before=["review"], multitask_strategy="enqueue", webhook=webhook),
        run_metadata={"source": "cron"},
    )


def _outcome(**overrides: Any) -> RunOutcome:
    fields: dict[str, Any] = {
        "status": "success",
        "values": {"step_count": 2},
        "run_started_at": _STARTED_AT,
        "run_ended_at": _ENDED_AT,
    }
    fields.update(overrides)
    return RunOutcome(**fields)


def _use_webhooks_config(monkeypatch: pytest.MonkeyPatch, webhooks: dict[str, Any]) -> None:
    monkeypatch.setattr(webhook_policy, "load_webhooks_config", lambda: webhooks)
    get_webhook_settings.cache_clear()


def _dispatcher_with(handler: Callable[[httpx.Request], httpx.Response]) -> WebhookDispatcher:
    """A dispatcher whose HTTP client answers from *handler* instead of the network."""
    dispatcher = WebhookDispatcher()
    dispatcher._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return dispatcher


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webhook_delivery, "_backoff_seconds", lambda _attempt: 0)


class TestBuildRunPayload:
    def test_matches_the_langgraph_webhook_shape(self) -> None:
        payload = build_run_payload(_job(), _outcome())

        assert payload["run_id"] == "run-1"
        assert payload["thread_id"] == "thread-1"
        assert payload["assistant_id"] == "assistant-1"
        assert payload["created_at"] == _CREATED_AT.isoformat()
        assert payload["updated_at"] == _ENDED_AT.isoformat()
        assert payload["status"] == "success"
        assert payload["metadata"] == {"source": "cron"}
        assert payload["multitask_strategy"] == "enqueue"
        assert payload["run_started_at"] == _STARTED_AT.isoformat()
        assert payload["run_ended_at"] == _ENDED_AT.isoformat()
        assert datetime.fromisoformat(payload["webhook_sent_at"]) >= _ENDED_AT
        assert payload["values"] == {"step_count": 2}
        assert "error" not in payload

    def test_kwargs_carry_the_run_request(self) -> None:
        kwargs = build_run_payload(_job(), _outcome())["kwargs"]

        assert kwargs == {
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "command": None,
            "config": {"configurable": {"model": "m"}},
            "context": {"tenant": "acme"},
            "stream_mode": ["values"],
            "interrupt_before": ["review"],
            "interrupt_after": None,
            "webhook": _WEBHOOK,
            "subgraphs": False,
            "durability": "exit",
        }

    def test_error_is_reported_when_the_run_failed(self) -> None:
        error = {"error": "RuntimeError", "message": "boom"}

        payload = build_run_payload(_job(), _outcome(status="error", error=error))

        assert payload["status"] == "error"
        assert payload["error"] == error

    def test_values_are_null_when_the_run_produced_none(self) -> None:
        assert build_run_payload(_job(), _outcome(values={}))["values"] is None
        assert build_run_payload(_job(), _outcome(values=None))["values"] is None

    def test_values_are_json_serialized(self) -> None:
        payload = build_run_payload(_job(), _outcome(values={"messages": [AIMessage(content="done")]}))

        message = payload["values"]["messages"][0]
        assert message["content"] == "done"
        assert message["type"] == "ai"
        json.dumps(payload)

    def test_unserializable_values_become_null(self) -> None:
        with patch.object(webhook_delivery._serializer, "serialize", side_effect=TypeError("nope")):
            payload = build_run_payload(_job(), _outcome())

        assert payload["values"] is None

    def test_unknown_start_time_is_null(self) -> None:
        assert build_run_payload(_job(), _outcome(run_started_at=None))["run_started_at"] is None

    def test_allowed_fields_trims_the_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_webhooks_config(monkeypatch, {"allowed_fields": ["run_id", "status", "error"]})

        payload = build_run_payload(_job(), _outcome(status="error", error={"error": "E", "message": "m"}))

        assert payload == {"run_id": "run-1", "status": "error", "error": {"error": "E", "message": "m"}}


class TestRedactWebhookUrl:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://hooks.example.com/aegra?token=secret#frag", "https://hooks.example.com/aegra"),
            ("https://user:pass@hooks.example.com:8443/a/b", "https://hooks.example.com:8443/a/b"),
            ("http://[::1]:9000/x", "http://::1:9000/x"),
        ],
    )
    def test_drops_query_and_credentials(self, url: str, expected: str) -> None:
        assert redact_webhook_url(url) == expected

    def test_invalid_url(self) -> None:
        assert redact_webhook_url("http://hooks.example.com:abc/") == "<invalid url>"


class TestDelivery:
    async def test_posts_the_payload_as_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LG_WEBHOOK_TOKEN", "t0k3n")
        _use_webhooks_config(monkeypatch, {"headers": {"Authorization": "Bearer ${{ env.LG_WEBHOOK_TOKEN }}"}})
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200)

        await _dispatcher_with(handler)._deliver(_job(), _outcome())

        assert len(requests) == 1
        request = requests[0]
        assert request.method == "POST"
        assert str(request.url) == _WEBHOOK
        assert request.headers["content-type"] == "application/json"
        assert request.headers["authorization"] == "Bearer t0k3n"
        body = json.loads(request.content)
        assert body["run_id"] == "run-1"
        assert body["values"] == {"step_count": 2}

    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    async def test_retries_transient_statuses(self, status: int) -> None:
        statuses = iter([status, 200])
        calls: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(next(statuses))

        with patch.object(webhook_delivery.logger, "info") as info:
            await _dispatcher_with(handler)._deliver(_job(), _outcome())

        assert len(calls) == 2
        assert info.call_args.kwargs["attempts"] == 2

    async def test_retries_network_errors(self) -> None:
        outcomes: list[Exception | int] = [httpx.ConnectError("refused"), httpx.ReadTimeout("slow"), 204]

        def handler(request: httpx.Request) -> httpx.Response:
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return httpx.Response(outcome)

        with patch.object(webhook_delivery.logger, "info") as info:
            await _dispatcher_with(handler)._deliver(_job(), _outcome())

        assert info.call_args.kwargs["attempts"] == 3

    async def test_gives_up_after_three_attempts(self) -> None:
        calls: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(503)

        with patch.object(webhook_delivery.logger, "warning") as warning:
            await _dispatcher_with(handler)._deliver(_job(), _outcome())

        assert len(calls) == 3
        assert warning.call_args.args[0] == "Webhook delivery failed"
        assert "503" in warning.call_args.kwargs["error"]

    @pytest.mark.parametrize("status", [301, 302, 400, 401, 404, 410])
    async def test_does_not_retry_or_follow_other_statuses(self, status: int) -> None:
        calls: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(status, headers={"Location": "http://169.254.169.254/"})

        with patch.object(webhook_delivery.logger, "warning") as warning:
            await _dispatcher_with(handler)._deliver(_job(), _outcome())

        assert len(calls) == 1
        assert warning.call_args.args[0] == "Webhook delivery failed"

    async def test_url_rejected_by_the_current_policy_is_not_called(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _use_webhooks_config(monkeypatch, {"url": {"allowed_domains": ["other.example.com"]}})
        calls: list[int] = []

        def handler(_request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200)

        with patch.object(webhook_delivery.logger, "warning") as warning:
            await _dispatcher_with(handler)._deliver(_job(), _outcome())

        assert calls == []
        assert warning.call_args.args[0] == "Webhook blocked by the URL policy"
        assert warning.call_args.kwargs["reason"] == "Webhook domain not allowed"

    async def test_httpx_request_logs_hide_the_webhook_url_only_during_delivery(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200)

        dispatcher = _dispatcher_with(handler)
        with caplog.at_level(logging.INFO, logger="httpx"):
            dispatcher.submit(_job(), _outcome())
            await dispatcher.stop()
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as other:
                await other.get("https://api.example.com/other")

        messages = [record.getMessage() for record in caplog.records if record.name == "httpx"]
        assert not any("secret" in message for message in messages)
        assert any("api.example.com/other" in message for message in messages)

    async def test_logs_never_carry_the_webhook_secret(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500)

        with patch.object(webhook_delivery.logger, "warning") as warning:
            await _dispatcher_with(handler)._deliver(_job(), _outcome())

        assert warning.call_args.kwargs["webhook"] == "https://hooks.example.com/aegra"
        assert "secret" not in repr(warning.call_args)

    async def test_unexpected_errors_are_logged_not_raised(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise RuntimeError("receiver exploded")

        with patch.object(webhook_delivery.logger, "exception") as exception:
            await _dispatcher_with(handler)._deliver(_job(), _outcome())

        exception.assert_called_once()


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await request.aread()
        self.requests.append(request)
        return httpx.Response(200)


class TestPinnedAddressTransport:
    @staticmethod
    def _transport(monkeypatch: pytest.MonkeyPatch, address: str | None) -> tuple[_PinnedAddressTransport, Any]:
        async def _resolve(host: str, port: int, policy: webhook_policy.WebhookUrlPolicy) -> str | None:
            return address

        monkeypatch.setattr(webhook_delivery, "resolve_webhook_address", _resolve)
        transport = _PinnedAddressTransport()
        recorder = _RecordingTransport()
        transport._inner = recorder  # type: ignore[assignment]
        return transport, recorder

    async def test_https_connects_to_the_checked_address_with_original_host_and_sni(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        transport, recorder = self._transport(monkeypatch, "93.184.215.14")

        async with httpx.AsyncClient(transport=transport) as client:
            await client.post("https://hooks.example.com/aegra?token=t", content=b"{}")

        sent = recorder.requests[0]
        assert sent.url.host == "93.184.215.14"
        assert sent.url.path == "/aegra"
        assert sent.url.query == b"token=t"
        assert sent.headers["host"] == "hooks.example.com"
        assert sent.extensions["sni_hostname"] == "hooks.example.com"
        assert sent.content == b"{}"

    async def test_http_pins_without_sni(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport, recorder = self._transport(monkeypatch, "2606:2800::1")

        async with httpx.AsyncClient(transport=transport) as client:
            await client.post("http://hooks.example.com:8080/x", content=b"{}")

        sent = recorder.requests[0]
        assert sent.url.host == "2606:2800::1"
        assert sent.url.port == 8080
        assert sent.headers["host"] == "hooks.example.com:8080"
        assert "sni_hostname" not in sent.extensions

    async def test_unpinned_hosts_pass_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport, recorder = self._transport(monkeypatch, None)

        async with httpx.AsyncClient(transport=transport) as client:
            await client.post("http://10.0.0.5/x", content=b"{}")

        assert recorder.requests[0].url.host == "10.0.0.5"

    async def test_blocked_hosts_raise_before_connecting(self) -> None:
        transport = _PinnedAddressTransport()
        recorder = _RecordingTransport()
        transport._inner = recorder  # type: ignore[assignment]

        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(WebhookUrlError, match="cloud metadata endpoint"):
                await client.post("http://169.254.169.254/latest", content=b"{}")

        assert recorder.requests == []


class TestDispatcher:
    async def test_submit_delivers_in_the_background(self) -> None:
        delivered = asyncio.Event()

        def handler(_request: httpx.Request) -> httpx.Response:
            delivered.set()
            return httpx.Response(200)

        dispatcher = _dispatcher_with(handler)
        dispatcher.submit(_job(), _outcome())

        await asyncio.wait_for(delivered.wait(), timeout=1)
        await dispatcher.stop()

    async def test_stop_waits_for_in_flight_deliveries(self) -> None:
        dispatcher = WebhookDispatcher()
        finished: list[str] = []

        async def _deliver(job: RunJob, _outcome: RunOutcome) -> None:
            await asyncio.sleep(0.01)
            finished.append(job.identity.run_id)

        dispatcher._deliver = _deliver  # type: ignore[method-assign]
        dispatcher.submit(_job(), _outcome())
        await dispatcher.stop()

        assert finished == ["run-1"]

    async def test_stop_cancels_deliveries_that_outlive_the_grace_period(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webhook_delivery, "_DRAIN_TIMEOUT_SECONDS", 0.01)
        dispatcher = WebhookDispatcher()
        cancelled = asyncio.Event()

        async def _deliver(_job: RunJob, _outcome: RunOutcome) -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        dispatcher._deliver = _deliver  # type: ignore[method-assign]
        dispatcher.submit(_job(), _outcome())
        await dispatcher.stop()

        assert cancelled.is_set()
        assert dispatcher._tasks == set()

    async def test_stop_closes_the_client(self) -> None:
        dispatcher = WebhookDispatcher()
        client = dispatcher._get_client()

        await dispatcher.stop()

        assert client.is_closed
        assert dispatcher._client is None

    async def test_client_skips_proxies_and_redirects(self) -> None:
        dispatcher = WebhookDispatcher()
        client = dispatcher._get_client()

        assert client.trust_env is False
        assert client.follow_redirects is False
        assert isinstance(client._transport, _PinnedAddressTransport)
        await dispatcher.stop()


class TestNotifyRunFinished:
    def test_skips_runs_without_a_webhook(self) -> None:
        with patch.object(webhook_delivery, "webhook_dispatcher") as dispatcher:
            notify_run_finished(_job(webhook=None), _outcome())

        dispatcher.submit.assert_not_called()

    def test_skips_when_webhooks_are_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webhook_policy, "load_http_config", lambda: {"disable_webhooks": True})
        webhooks_enabled.cache_clear()

        with patch.object(webhook_delivery, "webhook_dispatcher") as dispatcher:
            notify_run_finished(_job(), _outcome())

        dispatcher.submit.assert_not_called()

    def test_submits_runs_with_a_webhook(self) -> None:
        job = _job()
        outcome = _outcome()

        with patch.object(webhook_delivery, "webhook_dispatcher") as dispatcher:
            notify_run_finished(job, outcome)

        dispatcher.submit.assert_called_once_with(job, outcome)

    def test_never_raises_without_an_event_loop(self) -> None:
        dispatcher = MagicMock()
        dispatcher.submit.side_effect = RuntimeError("no running event loop")

        with patch.object(webhook_delivery, "webhook_dispatcher", dispatcher):
            notify_run_finished(_job(), _outcome())

        dispatcher.submit.assert_called_once()
