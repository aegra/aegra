"""Global registry of in-flight asyncio tasks for graph executions.

Defined in a dependency-free module so that any layer (API routes, broker
managers, streaming service) can import it without circular dependencies.
"""

import asyncio

active_runs: dict[str, asyncio.Task[None]] = {}

# Explicit API cancellations are marked so worker shutdown remains recoverable.
explicit_run_cancellations: set[str] = set()

# Runs inside execute_run here, so sure to reach its finally. A worker lists a run in
# active_runs before taking the lease, so that one cannot tell whether execution will happen.
executing_runs: set[str] = set()

# Runs the API made terminal while this process still executes them; the executor sends their webhook.
reconciled_run_cancellations: set[str] = set()
