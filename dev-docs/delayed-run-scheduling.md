# Delayed run scheduling

Run creation accepts `after_seconds` to delay graph execution while keeping the
run visible through the normal run APIs. A value of `0` preserves immediate
execution.

## Local executor

In development mode, `LocalExecutor` registers one asyncio task per run. A
delayed task sleeps before calling `execute_run`. The task is stored in
`active_runs` for its entire lifetime, so the existing cancellation and
shutdown paths can cancel the timer before graph execution begins.

## Worker executor

In production mode, the run row is committed as `pending` immediately. Delayed
runs store their UTC `not_before` timestamp in Postgres and are not pushed to
Redis at creation time. A lightweight dispatcher checks due rows and pushes
their IDs to the worker queue. Workers also apply the `not_before` predicate
when falling back to Postgres, so a delayed run cannot be claimed early.

The timestamp is persisted rather than held only in memory. After an API or
worker restart, the dispatcher discovers any pending row whose timestamp has
passed and queues it. Queue delivery is intentionally idempotent: the worker's
conditional pending/unclaimed lease update is the execution gate, so a
duplicate queue entry cannot execute a run twice.

Cancellation updates the pending row to `interrupted` before dispatch can claim
it. If a queue entry races with cancellation, the worker lease update rejects
the non-pending row.

The migration adding `runs.not_before` must be applied before enabling delayed
runs in a production deployment.
