"""Unit tests for webhook delivery: retries, signing, body merge, dispatch guards."""

import functools

import httpx
import pytest

from aegra_api.models.auth import User
from aegra_api.models.run_job import RunExecution, RunIdentity, RunJob
from aegra_api.models.webhooks import WebhookConfig
from aegra_api.services import webhook_service


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero backoff so retry tests don't sleep; deterministic attempt cap."""
    monkeypatch.setattr(webhook_service.settings.webhook, "WEBHOOK_BACKOFF_BASE_SECONDS", 0.0)
    monkeypatch.setattr(webhook_service.settings.webhook, "WEBHOOK_MAX_ATTEMPTS", 3)


def _install_handler(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Route the module's AsyncClient through an httpx.MockTransport."""
    factory = functools.partial(httpx.AsyncClient, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(webhook_service.httpx, "AsyncClient", factory)


class TestDeliver:
    @pytest.mark.asyncio
    async def test_success_single_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200)

        _install_handler(monkeypatch, handler)
        await webhook_service.deliver(WebhookConfig(url="https://x.io/h"), {"run_id": "r1"}, msg_id="r1")
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        statuses = iter([503, 200])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(next(statuses))

        _install_handler(monkeypatch, handler)
        await webhook_service.deliver(WebhookConfig(url="https://x.io/h"), {}, msg_id="r1")
        # exhausts nothing: second attempt returns 200

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(503)

        _install_handler(monkeypatch, handler)
        await webhook_service.deliver(WebhookConfig(url="https://x.io/h"), {}, msg_id="r1")
        assert len(calls) == 3  # WEBHOOK_MAX_ATTEMPTS

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(404)

        _install_handler(monkeypatch, handler)
        await webhook_service.deliver(WebhookConfig(url="https://x.io/h"), {}, msg_id="r1")
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_transport_error_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            raise httpx.ConnectError("boom")

        _install_handler(monkeypatch, handler)
        # Must not raise despite every attempt failing.
        await webhook_service.deliver(WebhookConfig(url="https://x.io/h"), {}, msg_id="r1")
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_signature_headers_only_with_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, httpx.Headers] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["headers"] = request.headers
            return httpx.Response(200)

        _install_handler(monkeypatch, handler)
        await webhook_service.deliver(WebhookConfig(url="https://x.io/h"), {}, msg_id="r1")
        assert "webhook-signature" not in seen["headers"]

        await webhook_service.deliver(WebhookConfig(url="https://x.io/h", secret="whsec_YWJj"), {}, msg_id="r1")
        assert "webhook-signature" in seen["headers"]
        assert seen["headers"]["webhook-id"] == "r1"

    @pytest.mark.asyncio
    async def test_body_merge_params_and_headers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["body"] = json.loads(request.content)
            captured["query"] = request.url.params.get("source")
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200)

        _install_handler(monkeypatch, handler)
        webhook = WebhookConfig(
            url="https://x.io/h",
            headers={"Authorization": "Bearer tok"},
            params={"source": "cron"},
            body={"status": "override"},
        )
        await webhook_service.deliver(webhook, {"run_id": "r1", "status": "success"}, msg_id="r1")
        assert captured["body"]["run_id"] == "r1"
        assert captured["body"]["status"] == "override"  # webhook.body overrides run payload
        assert captured["body"]["webhook_sent_at"]  # LangGraph-aligned send timestamp
        assert captured["query"] == "cron"
        assert captured["auth"] == "Bearer tok"


def _job_with_webhook(webhook: WebhookConfig | None) -> RunJob:
    return RunJob(
        identity=RunIdentity(run_id="r1", thread_id="t1", graph_id="g1", assistant_id="a1"),
        user=User(identity="u1"),
        execution=RunExecution(webhook=webhook),
    )


class TestDispatch:
    def test_noop_when_no_webhook(self) -> None:
        before = len(webhook_service._delivery_tasks)
        webhook_service.dispatch(_job_with_webhook(None), "success", {}, None)
        assert len(webhook_service._delivery_tasks) == before

    def test_noop_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webhook_service.settings.webhook, "WEBHOOK_ENABLED", False)
        before = len(webhook_service._delivery_tasks)
        webhook_service.dispatch(_job_with_webhook(WebhookConfig(url="https://x.io/h")), "success", {}, None)
        assert len(webhook_service._delivery_tasks) == before


class TestBlockedSSRF:
    """_blocked: link-local/metadata always refused; private only when opted in (B1)."""

    @staticmethod
    def _patch_resolve(monkeypatch: pytest.MonkeyPatch, ip: str) -> None:
        import asyncio
        from unittest.mock import AsyncMock

        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "getaddrinfo", AsyncMock(return_value=[(2, 1, 6, "", (ip, 0))]))

    @pytest.mark.asyncio
    async def test_link_local_metadata_always_blocked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webhook_service.settings.webhook, "WEBHOOK_BLOCK_PRIVATE_IPS", False)
        self._patch_resolve(monkeypatch, "169.254.169.254")
        assert await webhook_service._blocked("http://metadata.example/x") is True

    @pytest.mark.asyncio
    async def test_private_allowed_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webhook_service.settings.webhook, "WEBHOOK_BLOCK_PRIVATE_IPS", False)
        self._patch_resolve(monkeypatch, "192.168.1.5")
        assert await webhook_service._blocked("http://internal.example/x") is False

    @pytest.mark.asyncio
    async def test_private_blocked_when_opted_in(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webhook_service.settings.webhook, "WEBHOOK_BLOCK_PRIVATE_IPS", True)
        self._patch_resolve(monkeypatch, "192.168.1.5")
        assert await webhook_service._blocked("http://internal.example/x") is True

    @pytest.mark.asyncio
    async def test_public_host_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(webhook_service.settings.webhook, "WEBHOOK_BLOCK_PRIVATE_IPS", False)
        self._patch_resolve(monkeypatch, "93.184.216.34")
        assert await webhook_service._blocked("https://example.com/x") is False


class TestBuildPayload:
    """Delivery body is shaped like a LangGraph Platform Run webhook payload."""

    @pytest.mark.asyncio
    async def test_langgraph_aligned_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import AsyncMock

        monkeypatch.setattr(
            webhook_service,
            "_read_run",
            AsyncMock(
                return_value={
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:05+00:00",
                    "input": {"messages": []},
                    "config": {"configurable": {"model": "openai/gpt-4o"}},
                    "context": {"model": "openai/gpt-4o"},
                    "error_message": None,
                }
            ),
        )
        meta = {
            "run_id": "r1",
            "thread_id": "t1",
            "assistant_id": "a1",
            "status": "success",
            "values": {"messages": ["hi"]},
            "error": None,
            "metadata": {"env": "prod"},
            "multitask_strategy": "reject",
        }
        payload = await webhook_service._build_payload(meta)
        for field in (
            "run_id",
            "thread_id",
            "assistant_id",
            "status",
            "metadata",
            "multitask_strategy",
            "values",
            "error",
            "kwargs",
            "created_at",
            "updated_at",
            "run_ended_at",
        ):
            assert field in payload, f"missing LangGraph field {field}"
        assert payload["values"] == {"messages": ["hi"]}
        assert payload["kwargs"]["input"] == {"messages": []}
        assert payload["run_ended_at"] == "2026-01-01T00:00:05+00:00"
        assert payload["error"] is None

    @pytest.mark.asyncio
    async def test_error_object_and_db_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import AsyncMock

        monkeypatch.setattr(webhook_service, "_read_run", AsyncMock(return_value=None))
        meta = {
            "run_id": "r1",
            "thread_id": "t1",
            "assistant_id": "a1",
            "status": "error",
            "values": None,
            "error": "boom",
            "metadata": {},
            "multitask_strategy": None,
        }
        payload = await webhook_service._build_payload(meta)
        assert payload["error"] == {"message": "boom"}  # LangGraph error object shape
        assert payload["kwargs"] == {}  # DB unreadable → captured-only fallback
