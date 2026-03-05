"""Process-wide GPU concurrency lock.

Prevents training, inference, and hosting from loading models simultaneously,
which would cause VRAM collisions and OOM crashes.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

log = logging.getLogger(__name__)


class GPULock:
    """Singleton asyncio semaphore for GPU-bound operations."""

    _instance: Optional[GPULock] = None

    def __init__(self, max_concurrent: int = 1):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._current_operation: Optional[str] = None

    @classmethod
    def get(cls, max_concurrent: int = 1) -> GPULock:
        if cls._instance is None:
            cls._instance = cls(max_concurrent)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    async def acquire(self, operation: str = "unknown") -> None:
        if self._semaphore.locked():
            log.info(
                "GPU lock contention: '%s' waiting, '%s' holds lock",
                operation,
                self._current_operation,
            )
        await self._semaphore.acquire()
        self._current_operation = operation
        log.info("GPU lock acquired by '%s'", operation)

    def release(self) -> None:
        op = self._current_operation
        self._current_operation = None
        self._semaphore.release()
        log.info("GPU lock released by '%s'", op)

    async def __aenter__(self) -> GPULock:
        await self.acquire()
        return self

    async def __aexit__(self, *exc) -> None:
        self.release()

    @property
    def is_locked(self) -> bool:
        return self._semaphore.locked()

    @property
    def current_operation(self) -> Optional[str]:
        return self._current_operation
