"""Global registry of in-flight asyncio tasks for graph executions.

Defined in a dependency-free module so that any layer (API routes, broker
managers, streaming service) can import it without circular dependencies.
"""

import asyncio

active_runs: dict[str, asyncio.Task[None]] = {}

# Explicit API cancellations are marked so worker shutdown remains recoverable.
explicit_run_cancellations: set[str] = set()


def request_local_cancellation(run_id: str) -> bool:
    """Mark explicit cancel and cancel the owning task if this process has it."""
    explicit_run_cancellations.add(run_id)
    task = active_runs.get(run_id)
    if task is None or task.done():
        return False
    task.cancel()
    return True
