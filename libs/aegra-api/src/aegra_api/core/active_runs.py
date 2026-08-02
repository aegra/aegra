"""Global registry of in-flight asyncio tasks for graph executions.

Defined in a dependency-free module so that any layer (API routes, broker
managers, streaming service) can import it without circular dependencies.
"""

import asyncio

active_runs: dict[str, asyncio.Task[None]] = {}

# Run IDs whose worker wrapper was cancelled by an explicit API request.
# Worker shutdown cancellation deliberately does not add entries here, so a
# graceful shutdown still leaves leased runs for crash recovery.
explicit_run_cancellations: set[str] = set()
