"""Run-completion webhooks.

When a run that carries a ``webhook`` reaches a terminal status, POST it to
that URL in LangGraph's webhook format: the Run plus ``values``, ``error`` and
timing fields. Delivery is best effort, as on LangGraph Platform: a few retries
on transient failures, in a background task that never delays or fails the run.
"""

import asyncio
import json
import logging
import random
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx
import structlog
from pydantic import ValidationError

from aegra_api.core.serializers import GeneralSerializer
from aegra_api.models.run_job import RunJob
from aegra_api.services.webhook_policy import (
    WebhookDNSError,
    WebhookUrlError,
    check_webhook_url,
    get_webhook_settings,
    resolve_webhook_addresses,
    webhook_host,
    webhook_port,
    webhooks_enabled,
)

logger = structlog.getLogger(__name__)

_MAX_ATTEMPTS = 3
_CONNECT_TIMEOUT_SECONDS = 5.0
_REQUEST_TIMEOUT_SECONDS = 30.0
_DRAIN_TIMEOUT_SECONDS = 5.0
_MAX_PENDING_DELIVERIES = 1000

_serializer = GeneralSerializer()

_in_delivery: ContextVar[bool] = ContextVar("webhook_delivery", default=False)


class _HideDeliveryRequestLogs(logging.Filter):
    """httpx logs every request URL at INFO, and webhook URLs often carry the receiver's token."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not _in_delivery.get()


logging.getLogger("httpx").addFilter(_HideDeliveryRequestLogs())


@dataclass(frozen=True, slots=True)
class RunOutcome:
    """How a run ended, as reported in its webhook."""

    status: str
    values: Any = None
    error: dict[str, str] | None = None
    run_started_at: datetime | None = None
    run_ended_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class WebhookResponseError(Exception):
    """The receiver answered with a status that is not 2xx."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"Webhook receiver returned HTTP {status_code}")
        self.status_code = status_code


def _isoformat(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _serialize_values(values: Any, run_id: str) -> Any:
    try:
        return _serializer.serialize(values)
    except Exception as exc:
        logger.warning("Webhook values serialization failed", run_id=run_id, error=str(exc))
        return None


def build_run_payload(job: RunJob, outcome: RunOutcome) -> dict[str, Any]:
    """LangGraph's webhook body: the Run, its kwargs, final values and timings."""
    run_id = job.identity.run_id
    payload: dict[str, Any] = {
        "run_id": run_id,
        "thread_id": job.identity.thread_id,
        "assistant_id": job.identity.assistant_id,
        "created_at": _isoformat(job.identity.created_at),
        "updated_at": _isoformat(outcome.run_ended_at),
        "status": outcome.status,
        "metadata": job.run_metadata,
        "kwargs": {
            "input": job.execution.input_data,
            "command": job.execution.command,
            "config": job.execution.config,
            "context": job.execution.context,
            "stream_mode": job.execution.stream_mode,
            "interrupt_before": job.behavior.interrupt_before,
            "interrupt_after": job.behavior.interrupt_after,
            "webhook": job.behavior.webhook,
            "subgraphs": job.behavior.subgraphs,
            "durability": job.execution.durability,
        },
        "multitask_strategy": job.behavior.multitask_strategy,
        "run_started_at": _isoformat(outcome.run_started_at),
        "run_ended_at": _isoformat(outcome.run_ended_at),
        "webhook_sent_at": datetime.now(UTC).isoformat(),
        "values": _serialize_values(outcome.values, run_id) if outcome.values is not None else None,
    }
    if outcome.error is not None:
        payload["error"] = outcome.error
    allowed_fields = get_webhook_settings().allowed_fields
    if allowed_fields:
        payload = {key: value for key, value in payload.items() if key in allowed_fields}
    return payload


def redact_webhook_url(url: str) -> str:
    """Scheme, host and path only: query strings and userinfo often carry the receiver's token."""
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        return "<invalid url>"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{parsed.host}{port}{parsed.path}"


def _is_retryable(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def _backoff_seconds(attempt: int) -> float:
    return 2 ** (attempt - 1) + random.uniform(0, 1)  # noqa: S311  # nosec B311


def _pinned(request: httpx.Request, address: str, extensions: dict[str, Any]) -> httpx.Request:
    """*request* sent to *address*, keeping the original Host header and TLS name."""
    return httpx.Request(
        request.method,
        request.url.copy_with(host=address),
        headers=request.headers,
        stream=request.stream,
        extensions=extensions,
    )


class _PinnedAddressTransport(httpx.AsyncBaseTransport):
    """Connect to the address the URL policy approved, so DNS cannot change between check and connect."""

    def __init__(self) -> None:
        # No keep-alive: pooled connections are keyed by IP and could carry another host's TLS session.
        self._inner = httpx.AsyncHTTPTransport(limits=httpx.Limits(max_keepalive_connections=0))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        host = webhook_host(request.url)
        addresses = await resolve_webhook_addresses(host, webhook_port(request.url), get_webhook_settings().url)
        if addresses is None:
            return await self._inner.handle_async_request(request)
        extensions = dict(request.extensions)
        if request.url.scheme == "https":
            extensions["sni_hostname"] = host
        # Every address passed the policy; move on when one is unreachable, e.g. IPv6 first on an IPv4-only host.
        for address in addresses[:-1]:
            try:
                return await self._inner.handle_async_request(_pinned(request, address, extensions))
            except (httpx.ConnectError, httpx.ConnectTimeout):
                continue
        return await self._inner.handle_async_request(_pinned(request, addresses[-1], extensions))

    async def aclose(self) -> None:
        await self._inner.aclose()


class WebhookDispatcher:
    """Sends run webhooks from background tasks and drains them at shutdown."""

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None
        self._tasks: set[asyncio.Task[None]] = set()

    def submit(self, job: RunJob, outcome: RunOutcome) -> None:
        """Deliver in the background; the caller never waits on the receiver."""
        # Each pending delivery holds the run's values, so slow receivers must not grow memory without bound.
        if len(self._tasks) >= _MAX_PENDING_DELIVERIES:
            logger.warning("Too many webhook deliveries pending, dropping this one", run_id=job.identity.run_id)
            return
        task = asyncio.create_task(self._deliver(job, outcome), name=f"webhook-{job.identity.run_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def stop(self) -> None:
        """Give in-flight deliveries a short grace period, then cancel the rest."""
        if self._tasks:
            _done, pending = await asyncio.wait(set(self._tasks), timeout=_DRAIN_TIMEOUT_SECONDS)
            if pending:
                logger.warning("Cancelling undelivered webhooks at shutdown", count=len(pending))
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                transport=_PinnedAddressTransport(),
                timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS, connect=_CONNECT_TIMEOUT_SECONDS),
                follow_redirects=False,
                # Proxy env vars would mount transports that skip the address pinning.
                trust_env=False,
            )
        return self._client

    async def _deliver(self, job: RunJob, outcome: RunOutcome) -> None:
        # Each delivery runs in its own task, so this only hides this task's httpx logs.
        _in_delivery.set(True)
        run_id = job.identity.run_id
        url = job.behavior.webhook or ""
        target = redact_webhook_url(url)
        try:
            settings = get_webhook_settings()
            check_webhook_url(url, settings.url)
            body = json.dumps(build_run_payload(job, outcome), default=str).encode()
            attempts = await self._post(url, body, {"Content-Type": "application/json", **settings.headers})
        except (httpx.HTTPError, WebhookResponseError, WebhookDNSError, TimeoutError) as exc:
            error = str(exc) or type(exc).__name__
            logger.warning("Webhook delivery failed", run_id=run_id, webhook=target, error=error)
        except WebhookUrlError as exc:
            logger.warning("Webhook blocked by the URL policy", run_id=run_id, webhook=target, reason=str(exc))
        except Exception:
            logger.exception("Webhook delivery crashed", run_id=run_id, webhook=target)
        else:
            logger.info("Webhook delivered", run_id=run_id, webhook=target, status=outcome.status, attempts=attempts)

    async def _post(self, url: str, body: bytes, headers: dict[str, str]) -> int:
        """POST with retries on network and DNS errors, timeouts, 429 and 5xx; return the attempt that succeeded."""
        attempt = 1
        while True:
            try:
                # httpx timeouts restart on every byte, so a trickling receiver needs a wall-clock cap.
                async with asyncio.timeout(_REQUEST_TIMEOUT_SECONDS):
                    # Streamed so a large response body is never read into memory.
                    async with self._get_client().stream("POST", url, content=body, headers=headers) as response:
                        status_code = response.status_code
            except (httpx.TimeoutException, httpx.NetworkError, WebhookDNSError, TimeoutError):
                if attempt >= _MAX_ATTEMPTS:
                    raise
            else:
                if response.is_success:
                    return attempt
                if attempt >= _MAX_ATTEMPTS or not _is_retryable(status_code):
                    raise WebhookResponseError(status_code)
            await asyncio.sleep(_backoff_seconds(attempt))
            attempt += 1


webhook_dispatcher = WebhookDispatcher()


def job_for_webhook(
    params: dict[str, Any] | None,
    *,
    run_id: str,
    thread_id: str,
    assistant_id: str | None,
    created_at: datetime | None,
) -> RunJob | None:
    """Rebuild a stored run's job when it asked for a webhook, else None.

    For terminal transitions made outside the executor, from the row's ``execution_params``.
    """
    if not params:
        return None
    behavior = params.get("behavior")
    if not isinstance(behavior, dict) or not behavior.get("webhook"):
        return None
    try:
        return RunJob.from_execution_params(
            params, run_id=run_id, thread_id=thread_id, assistant_id=assistant_id, created_at=created_at
        )
    except (KeyError, ValidationError):
        logger.warning("Cannot rebuild the run for its webhook", run_id=run_id)
        return None


def notify_run_finished(job: RunJob, outcome: RunOutcome) -> None:
    """Queue the run's webhook when it has one. Synchronous and never raises, so it is safe in ``finally``."""
    if not job.behavior.webhook:
        return
    if not webhooks_enabled():
        logger.info("Webhooks are disabled, skipping run webhook", run_id=job.identity.run_id)
        return
    try:
        webhook_dispatcher.submit(job, outcome)
    except RuntimeError:
        logger.exception("Could not schedule run webhook", run_id=job.identity.run_id)
