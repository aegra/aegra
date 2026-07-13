"""Best-effort outbound webhook delivery on run completion.

``dispatch`` is fire-and-forget: it schedules a detached task and returns
immediately so the run pipeline is never blocked or failed by a webhook.
Delivery retries transient failures with exponential backoff and signs the
request per the Standard Webhooks spec when a secret is configured.

The body is a LangGraph Platform-aligned Run object: run_id / thread_id /
assistant_id / status / metadata / multitask_strategy / kwargs / values / error /
created_at / updated_at / run_ended_at / webhook_sent_at.
"""

import asyncio
import ipaddress
import json
import socket
import time
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from sqlalchemy.exc import SQLAlchemyError

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import _get_session_maker
from aegra_api.models.run_job import RunJob
from aegra_api.models.webhooks import WebhookConfig, sign
from aegra_api.settings import settings

logger = structlog.getLogger(__name__)

# Strong refs so fire-and-forget delivery tasks survive GC until done.
_delivery_tasks: set[asyncio.Task[None]] = set()

# Transient statuses worth retrying; other 4xx are client-contract errors.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def dispatch(job: RunJob, status: str, output: dict[str, Any] | None, error: str | None) -> None:
    """Schedule webhook delivery for a completed run. No-op when unset or disabled.

    Captures only small fields + the final state (never the RunJob) so a slow or
    retrying delivery can't pin the run's full input/context in memory.
    """
    webhook = job.execution.webhook
    if webhook is None or not settings.webhook.WEBHOOK_ENABLED:
        return
    meta: dict[str, Any] = {
        "run_id": job.identity.run_id,
        "thread_id": job.identity.thread_id,
        "assistant_id": job.identity.assistant_id,
        "status": status,
        "values": output,
        "error": error,
        "metadata": dict(job.run_metadata),
        "multitask_strategy": job.behavior.multitask_strategy,
    }
    task = asyncio.create_task(_run_delivery(webhook, meta))
    _delivery_tasks.add(task)
    task.add_done_callback(_delivery_tasks.discard)


async def drain(timeout: float = 5.0) -> None:
    """Await outstanding delivery tasks on shutdown, up to *timeout* seconds."""
    if not _delivery_tasks:
        return
    await asyncio.wait(set(_delivery_tasks), timeout=timeout)


async def _run_delivery(webhook: WebhookConfig, meta: dict[str, Any]) -> None:
    """Task boundary: build the payload and deliver. Swallows all errors (best-effort)."""
    run_id = meta["run_id"]
    try:
        await deliver(webhook, await _build_payload(meta), msg_id=run_id)
    except Exception:
        logger.exception("Webhook delivery task crashed", run_id=run_id)


async def _build_payload(meta: dict[str, Any]) -> dict[str, Any]:
    """Shape the LangGraph Platform-aligned Run body from captured fields + the run row.

    Enriches with the persisted run (created_at/updated_at, kwargs); falls back to
    the captured fields when the row is unreadable/gone.
    """
    err = meta["error"]
    payload: dict[str, Any] = {
        "run_id": meta["run_id"],
        "thread_id": meta["thread_id"],
        "assistant_id": meta["assistant_id"],
        "status": meta["status"],
        "metadata": meta["metadata"],
        "multitask_strategy": meta["multitask_strategy"],
        "values": meta["values"],  # final thread state (latest checkpoint values)
        "error": {"message": err} if err else None,
        "kwargs": {},
        "created_at": None,
        "updated_at": None,
        "run_ended_at": None,
    }
    row = await _read_run(meta["run_id"])
    if row is not None:
        payload["created_at"] = row["created_at"]
        payload["updated_at"] = row["updated_at"]
        payload["run_ended_at"] = row["updated_at"]  # run just finalized; no separate end column
        payload["kwargs"] = {"input": row["input"], "config": row["config"], "context": row["context"]}
        if row["error_message"] and payload["error"] is None:
            payload["error"] = {"message": row["error_message"]}
    return payload


async def _read_run(run_id: str) -> dict[str, Any] | None:
    """Read the finalized run row's payload fields (None if unreadable/gone)."""
    maker = _get_session_maker()
    try:
        async with maker() as session:
            row = await session.get(RunORM, run_id)
            if row is None:
                return None
            return {
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
                "input": row.input,
                "config": row.config,
                "context": row.context,
                "error_message": row.error_message,
            }
    except SQLAlchemyError:
        logger.warning("Webhook payload DB read failed; sending captured fields only", run_id=run_id)
        return None


async def deliver(webhook: WebhookConfig, payload: dict[str, Any], *, msg_id: str) -> None:
    """POST *payload* to the webhook with retries. Best-effort — never raises."""
    if await _blocked(webhook.url):
        logger.warning("Webhook blocked by SSRF guard", host=_host(webhook.url))
        return
    body = json.dumps(
        {**payload, "webhook_sent_at": datetime.now(UTC).isoformat(), **(webhook.body or {})},
        default=str,
    ).encode()
    attempts = settings.webhook.WEBHOOK_MAX_ATTEMPTS
    async with httpx.AsyncClient(
        timeout=settings.webhook.WEBHOOK_TIMEOUT_SECONDS,
        follow_redirects=False,  # SSRF: no redirect-based bypass of the url check
    ) as client:
        for attempt in range(1, attempts + 1):
            if await _attempt(client, webhook, body, msg_id=msg_id):
                return
            if attempt < attempts:
                await asyncio.sleep(settings.webhook.WEBHOOK_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
    logger.warning("Webhook delivery exhausted", host=_host(webhook.url), attempts=attempts)


async def _attempt(client: httpx.AsyncClient, webhook: WebhookConfig, body: bytes, *, msg_id: str) -> bool:
    """Perform one delivery attempt. Return True to stop (done/non-retryable), False to retry."""
    headers = {"content-type": "application/json", **(webhook.headers or {})}
    if webhook.secret:
        timestamp = int(time.time())
        headers["webhook-id"] = msg_id
        headers["webhook-timestamp"] = str(timestamp)
        headers["webhook-signature"] = sign(webhook.secret, msg_id, timestamp, body)
    try:
        response = await client.request(
            webhook.method, webhook.url, content=body, headers=headers, params=webhook.params
        )
    except httpx.RequestError as exc:
        logger.warning("Webhook attempt failed", host=_host(webhook.url), error=type(exc).__name__)
        return False
    if response.status_code < 400:
        return True
    if response.status_code in _RETRYABLE_STATUS:
        logger.warning("Webhook retryable status", status=response.status_code, host=_host(webhook.url))
        return False
    logger.warning("Webhook rejected", status=response.status_code, host=_host(webhook.url))
    return True


def _host(url: str) -> str:
    """Hostname for logging — never log the full URL (may carry path secrets)."""
    return urlparse(url).hostname or "?"


async def _blocked(url: str) -> bool:
    """SSRF guard. Link-local (incl. cloud metadata 169.254.169.254) is always
    refused; loopback/private ranges only when WEBHOOK_BLOCK_PRIVATE_IPS is set."""
    host = urlparse(url).hostname
    if not host:
        return True
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None)
    except socket.gaierror:
        return True  # unresolvable → refuse
    block_private = settings.webhook.WEBHOOK_BLOCK_PRIVATE_IPS
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_link_local or ip.is_multicast or ip.is_reserved:
            return True
        if block_private and (ip.is_private or ip.is_loopback):
            return True
    return False
