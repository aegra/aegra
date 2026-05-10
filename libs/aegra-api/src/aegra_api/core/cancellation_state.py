"""In-process registry of run-cancellation reasons.

``execute_run`` can receive CancelledError from three sources: user
cancel, lease loss, or timeout. They are disambiguated by recording
which reason the run_id was tagged with before ``task.cancel()``.

Lives in ``core/`` (no service dependencies) so any layer — API
routes, broker managers, executors — can use it without circular
imports. Same shape as ``core/active_runs.py``.

THREADING / LOOP MODEL: this registry assumes a single asyncio event
loop per process. Do not invoke from threads, signal handlers, or
``atexit`` callbacks — the underlying dict is not lock-protected.
Cross-instance cancellation is handled by ``RedisBrokerManager``,
which receives pubsub messages on the same loop.

OWNERSHIP: marks are written by ``mark()`` (API cancel, heartbeat
lease-loss). The tag is read twice — first by ``execute_run``'s
CancelledError handler (peek via ``reason_of``, must not consume),
then by the worker after ``await job_task`` raises (atomic read+remove
via ``pop_reason``). Single-layer semantics: there is exactly one tag
per ``run_id`` at a time. Cleanup ownership:

- **Cancel path**: ``worker_executor._execute_with_lease`` calls
  ``pop_reason`` in its CancelledError handler — atomic with the read,
  so a concurrent ``mark`` for the re-run is unblocked at zero awaits.
- **Outermost safety-net**: ``worker_executor._execute_and_release``
  finally calls ``clear`` — covers rare paths that bypass the inner
  except (e.g. CancelledError raised inside ``_acquire_and_load``
  before ``job_task`` exists, or a leaked mark on a success path).
- **LocalExecutor (dev mode)**: ``submit`` adds a done-callback on
  the task that calls ``clear`` — there is no enclosing finally.

Precedence rule: a run is tagged with at most one reason. ``lease_loss``
always wins — once set it is never downgraded to ``"user"`` (only an
explicit ``clear`` removes it). ``execute_run`` reads the tag in this
order and acts on the first hit: ``lease_loss`` > ``user`` > timeout
(default). The precedence is enforced inside ``mark`` so an in-flight
user cancel arriving after a heartbeat-driven lease loss cannot make the
old worker finalize the run as ``interrupted`` and clobber the rerun.
"""

from typing import Literal

CancellationReason = Literal["user", "lease_loss"]


class CancellationRegistry:
    """Per-process registry tagging in-flight runs with a cancel reason."""

    def __init__(self) -> None:
        self._tags: dict[str, CancellationReason] = {}

    def mark(self, run_id: str, reason: CancellationReason) -> None:
        """Tag a run with a cancel reason.

        ``lease_loss`` always wins: a later ``mark(..., "user")`` is
        ignored if a lease-loss tag is already present. Without this
        precedence, a user cancel arriving in the overlap window between
        heartbeat-driven lease loss and the rerun would erase the
        lease-loss tag, and ``execute_run`` on the old worker would
        finalize the run as ``interrupted`` — clobbering the rerun's
        eventual outcome.
        """
        if self._tags.get(run_id) == "lease_loss" and reason != "lease_loss":
            return
        self._tags[run_id] = reason

    def reason_of(self, run_id: str) -> CancellationReason | None:
        """Peek at the tagged reason without removing it.

        Used by ``execute_run`` to classify the CancelledError. The tag must
        survive this read because the worker peeks it again post-job-done —
        use ``pop_reason`` for end-of-life atomic read+remove.
        """
        return self._tags.get(run_id)

    def pop_reason(self, run_id: str) -> CancellationReason | None:
        """Atomically read and remove the tag — single call replacing
        ``reason_of`` followed by ``clear``. Used by the worker's cancel
        handler so a concurrent ``mark`` for a re-run on another worker
        isn't blocked by precedence (lease_loss > user)."""
        return self._tags.pop(run_id, None)

    def clear(self, run_id: str, *, only: CancellationReason | None = None) -> None:
        """Drop the tag for ``run_id``.

        ``only`` restricts removal to a specific reason — used to roll
        back a ``mark(..., "user")`` without disturbing a lease-loss tag
        that arrived in the meantime. Default removes any tag.
        """
        if only is None or self._tags.get(run_id) == only:
            self._tags.pop(run_id, None)

    def clear_all(self) -> None:
        """Drop every tag. Test-only — production code uses ``clear(run_id)``."""
        self._tags.clear()


cancellations = CancellationRegistry()
