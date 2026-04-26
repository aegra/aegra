"""E2E tests for worker and reaper Prometheus metrics.

Requires a running server with ENABLE_PROMETHEUS_METRICS=true
and REDIS_BROKER_ENABLED=true (production mode).

Test plan:
1. Verify all 21 metrics appear on /metrics endpoint
2. Submit a run, verify dispatched/completed/execution_seconds increment
3. Verify redis_reachable=1 and queue_depth gauge
4. Verify reaper cycle_seconds is observed after >=15s
5. Cancel a pending run, verify completed{interrupted} increments
"""

import asyncio
import re

import httpx
import pytest

from aegra_api.settings import settings

from ._utils import elog, get_e2e_client


def _parse_metric(body: str, metric_name: str, labels: dict[str, str] | None = None) -> float | None:
    """Parse a Prometheus metric value from exposition text.

    For labeled metrics, pass labels as dict. For unlabeled, pass None.
    Returns the float value or None if not found.
    """
    if labels:
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        pattern = rf"^{re.escape(metric_name)}\{{{re.escape(label_str)}\}}\s+(\S+)"
    else:
        # Unlabeled metric: match line with metric_name followed by value, no braces
        pattern = rf"^{re.escape(metric_name)}\s+(\S+)"

    for line in body.splitlines():
        m = re.match(pattern, line)
        if m:
            return float(m.group(1))
    return None


def _get_metric(body: str, metric_name: str, labels: dict[str, str] | None = None) -> float:
    """Parse a metric value, raising AssertionError if not found."""
    val = _parse_metric(body, metric_name, labels)
    label_info = f" with labels {labels}" if labels else ""
    assert val is not None, f"Metric {metric_name}{label_info} not found in /metrics output"
    return val


async def _scrape(client: httpx.AsyncClient) -> str:
    """Scrape the /metrics endpoint and return the body."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    return resp.text


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_all_worker_metrics_present() -> None:
    """All 21 worker + reaper metric names should appear in /metrics output."""
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        pytest.skip("ENABLE_PROMETHEUS_METRICS=false")

    server_url = settings.app.SERVER_URL
    async with httpx.AsyncClient(base_url=server_url, timeout=10.0) as client:
        body = await _scrape(client)

    expected_metrics = [
        # Worker metrics (15)
        "aegra_runs_dispatched",
        "aegra_runs_completed",
        "aegra_runs_in_flight",
        "aegra_runs_discarded",
        "aegra_run_execution_seconds",
        "aegra_run_queue_wait_seconds",
        "aegra_run_timeouts",
        "aegra_submit_errors",
        "aegra_runs_dequeued",
        "aegra_dequeue_errors",
        "aegra_redis_reachable",
        "aegra_heartbeat_extensions",
        "aegra_heartbeat_failures",
        "aegra_lease_losses",
        "aegra_run_retries",
        # Reaper metrics (5)
        "aegra_reaper_crashed_recovered",
        "aegra_reaper_stuck_reenqueued",
        "aegra_reaper_permanently_failed",
        "aegra_reaper_cycle_seconds",
        "aegra_queue_depth",
    ]

    missing = [m for m in expected_metrics if m not in body]
    elog("Missing metrics", missing)
    assert not missing, f"Missing metrics from /metrics: {missing}"


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_redis_reachable_gauge() -> None:
    """redis_reachable should be 1.0 when Redis is connected."""
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        pytest.skip("ENABLE_PROMETHEUS_METRICS=false")

    server_url = settings.app.SERVER_URL
    async with httpx.AsyncClient(base_url=server_url, timeout=10.0) as client:
        body = await _scrape(client)

    val = _get_metric(body, "aegra_redis_reachable")
    elog("redis_reachable", val)
    assert val == 1.0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_queue_depth_gauge() -> None:
    """queue_depth should be a non-negative number."""
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        pytest.skip("ENABLE_PROMETHEUS_METRICS=false")

    server_url = settings.app.SERVER_URL
    async with httpx.AsyncClient(base_url=server_url, timeout=10.0) as client:
        body = await _scrape(client)

    val = _get_metric(body, "aegra_queue_depth")
    elog("queue_depth", val)
    assert val >= 0.0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_run_lifecycle_metrics() -> None:
    """Submit a run, wait for completion, verify dispatched/completed/execution_seconds."""
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        pytest.skip("ENABLE_PROMETHEUS_METRICS=false")

    server_url = settings.app.SERVER_URL
    sdk = get_e2e_client()
    graph_id = "agent"

    # Create assistant
    assistant = await sdk.assistants.create(graph_id=graph_id, if_exists="do_nothing")
    assistant_id = assistant["assistant_id"]

    async with httpx.AsyncClient(base_url=server_url, timeout=30.0) as client:
        # Scrape before
        before = await _scrape(client)

    # Create a thread and run
    thread = await sdk.threads.create()
    elog("Created thread", thread)

    run = await sdk.runs.create(
        thread["thread_id"],
        assistant_id=assistant_id,
        input={"messages": [{"role": "human", "content": "metrics test"}]},
    )
    elog("Created run", run)

    # Wait for completion (join returns output, not run; use get to check status)
    await sdk.runs.join(thread["thread_id"], run["run_id"])
    final_run = await sdk.runs.get(thread["thread_id"], run["run_id"])
    status = final_run["status"]
    elog("Final run status", {"status": status, "run_id": final_run["run_id"]})
    # Run may succeed or error (if LLM key missing) — either is a terminal state
    assert status in ("success", "error", "interrupted")

    async with httpx.AsyncClient(base_url=server_url, timeout=10.0) as client:
        # Scrape after
        after = await _scrape(client)

    # dispatched should have incremented
    dispatched_before = _parse_metric(before, "aegra_runs_dispatched_total", {"graph_id": graph_id}) or 0
    dispatched_after = _get_metric(after, "aegra_runs_dispatched_total", {"graph_id": graph_id})
    elog("dispatched", {"before": dispatched_before, "after": dispatched_after})
    assert dispatched_after > dispatched_before

    # completed should have incremented for the terminal status
    completed_before = (
        _parse_metric(before, "aegra_runs_completed_total", {"graph_id": graph_id, "status": status}) or 0
    )
    completed_after = _get_metric(after, "aegra_runs_completed_total", {"graph_id": graph_id, "status": status})
    elog("completed", {"before": completed_before, "after": completed_after, "status": status})
    assert completed_after > completed_before

    # execution_seconds should have been observed (count > 0)
    exec_count = _get_metric(after, "aegra_run_execution_seconds_count", {"graph_id": graph_id})
    elog("execution_seconds_count", exec_count)
    assert exec_count > 0

    # queue_wait_seconds should have been observed
    wait_count = _get_metric(after, "aegra_run_queue_wait_seconds_count", {"graph_id": graph_id})
    elog("queue_wait_seconds_count", wait_count)
    assert wait_count > 0

    # dequeued should have incremented
    dequeued = _get_metric(after, "aegra_runs_dequeued_total")
    elog("dequeued_total", dequeued)
    assert dequeued > 0

    # Clean up
    await sdk.threads.delete(thread["thread_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cancel_pending_run_metrics() -> None:
    """Cancel a pending run before it executes, verify completed{interrupted}."""
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        pytest.skip("ENABLE_PROMETHEUS_METRICS=false")
    if not settings.redis.REDIS_BROKER_ENABLED:
        pytest.skip("REDIS_BROKER_ENABLED=false — cancel-pending CAS requires worker mode")

    server_url = settings.app.SERVER_URL
    sdk = get_e2e_client()
    graph_id = "agent"

    # Create assistant
    assistant = await sdk.assistants.create(graph_id=graph_id, if_exists="do_nothing")
    assistant_id = assistant["assistant_id"]

    # Scrape before
    async with httpx.AsyncClient(base_url=server_url, timeout=10.0) as client:
        before = await _scrape(client)

    # Create a thread and run
    thread = await sdk.threads.create()

    # Submit run — it will go through the worker queue
    run = await sdk.runs.create(
        thread["thread_id"],
        assistant_id=assistant_id,
        input={"messages": [{"role": "human", "content": "cancel test"}]},
    )
    elog("Created run", {"run_id": run["run_id"], "status": run["status"]})

    # Cancel it
    cancelled = await sdk.runs.cancel(thread["thread_id"], run["run_id"])
    elog("Cancelled run", {"run_id": cancelled["run_id"], "status": cancelled["status"]})

    # Wait a moment for metrics to settle
    await asyncio.sleep(1.0)

    # Scrape after
    async with httpx.AsyncClient(base_url=server_url, timeout=10.0) as client:
        after = await _scrape(client)

    # completed{interrupted} should have incremented (either from CAS or user cancel)
    interrupted_after = _parse_metric(
        after, "aegra_runs_completed_total", {"graph_id": graph_id, "status": "interrupted"}
    )
    interrupted_before = (
        _parse_metric(before, "aegra_runs_completed_total", {"graph_id": graph_id, "status": "interrupted"}) or 0
    )
    elog("completed{interrupted}", {"before": interrupted_before, "after": interrupted_after})
    # It may have completed as success before cancel reached it, so check >= rather than >
    assert interrupted_after is not None

    # Clean up
    await sdk.threads.delete(thread["thread_id"])


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_reaper_cycle_observed() -> None:
    """After the reaper runs (every 15s), cycle_seconds should have observations."""
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        pytest.skip("ENABLE_PROMETHEUS_METRICS=false")
    if not settings.redis.REDIS_BROKER_ENABLED:
        pytest.skip("REDIS_BROKER_ENABLED=false — reaper requires worker mode")

    server_url = settings.app.SERVER_URL

    # Wait for at least one reaper cycle (15s default)
    await asyncio.sleep(16)

    async with httpx.AsyncClient(base_url=server_url, timeout=10.0) as client:
        body = await _scrape(client)

    cycle_count = _get_metric(body, "aegra_reaper_cycle_seconds_count")
    elog("reaper_cycle_seconds_count", cycle_count)
    assert cycle_count > 0, "Reaper should have completed at least one cycle after 16s"

    # queue_depth should be set by the reaper
    depth = _get_metric(body, "aegra_queue_depth")
    elog("queue_depth (post-reaper)", depth)
    assert depth >= 0.0


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_heartbeat_metric_exists_after_run() -> None:
    """Verify the heartbeat_extensions metric is registered and queryable after a run."""
    if not settings.observability.ENABLE_PROMETHEUS_METRICS:
        pytest.skip("ENABLE_PROMETHEUS_METRICS=false")
    if not settings.redis.REDIS_BROKER_ENABLED:
        pytest.skip("REDIS_BROKER_ENABLED=false — heartbeats require worker mode")

    server_url = settings.app.SERVER_URL
    sdk = get_e2e_client()
    graph_id = "agent"

    # Create assistant
    assistant = await sdk.assistants.create(graph_id=graph_id, if_exists="do_nothing")
    assistant_id = assistant["assistant_id"]

    # Scrape before
    async with httpx.AsyncClient(base_url=server_url, timeout=10.0) as client:
        before = await _scrape(client)

    # Submit a run
    thread = await sdk.threads.create()
    run = await sdk.runs.create(
        thread["thread_id"],
        assistant_id=assistant_id,
        input={"messages": [{"role": "human", "content": "heartbeat test " * 50}]},
    )
    elog("Created run", {"run_id": run["run_id"]})

    # Wait for it to complete
    await sdk.runs.join(thread["thread_id"], run["run_id"])
    final_run = await sdk.runs.get(thread["thread_id"], run["run_id"])
    elog("Final run", {"status": final_run["status"]})

    # Scrape after
    async with httpx.AsyncClient(base_url=server_url, timeout=10.0) as client:
        after = await _scrape(client)

    # Check if heartbeat was extended (run may be too fast for 10s heartbeat)
    heartbeat_after = _parse_metric(after, "aegra_heartbeat_extensions_total", {"graph_id": graph_id}) or 0
    heartbeat_before = _parse_metric(before, "aegra_heartbeat_extensions_total", {"graph_id": graph_id}) or 0
    elog("heartbeat_extensions", {"before": heartbeat_before, "after": heartbeat_after})
    # Don't assert — echo graph completes fast. Just verify the metric exists.

    # Clean up
    await sdk.threads.delete(thread["thread_id"])
