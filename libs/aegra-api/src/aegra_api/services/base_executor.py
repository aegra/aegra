"""Abstract interface for run execution dispatch.

Follows the same strategy pattern as the broker abstraction:
one interface, two backends (local asyncio tasks vs Redis workers).
"""

from abc import ABC, abstractmethod

from aegra_api.models.run_job import RunJob


class BaseExecutor(ABC):
    """Dispatches RunJobs for execution and tracks their lifecycle."""

    @abstractmethod
    async def submit(self, job: RunJob) -> None:
        """Schedule a job for execution. Returns immediately, including delays."""

    @abstractmethod
    async def wait_for_completion(self, run_id: str, *, timeout: float = 300.0) -> None:
        """Block until the run reaches a terminal state or timeout expires."""

    @abstractmethod
    async def start(self) -> None:
        """Initialize resources (called during app startup)."""

    @abstractmethod
    async def stop(self) -> None:
        """Drain in-flight work and release resources (called during shutdown)."""
