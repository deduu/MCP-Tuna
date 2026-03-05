"""Tests for the GPU concurrency lock."""
from __future__ import annotations

import asyncio

import pytest

from shared.gpu_lock import GPULock


@pytest.fixture(autouse=True)
def reset_singleton():
    GPULock.reset()
    yield
    GPULock.reset()


class TestGPULock:
    def test_singleton(self):
        a = GPULock.get()
        b = GPULock.get()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = GPULock.get()
        GPULock.reset()
        b = GPULock.get()
        assert a is not b

    async def test_acquire_release(self):
        lock = GPULock.get()
        assert not lock.is_locked
        await lock.acquire("test_op")
        assert lock.is_locked
        assert lock.current_operation == "test_op"
        lock.release()
        assert not lock.is_locked
        assert lock.current_operation is None

    async def test_context_manager(self):
        lock = GPULock.get()
        async with lock:
            assert lock.is_locked
        assert not lock.is_locked

    async def test_serializes_concurrent_access(self):
        lock = GPULock.get()
        order: list[str] = []

        async def task(name: str, delay: float):
            await lock.acquire(name)
            order.append(f"{name}_start")
            await asyncio.sleep(delay)
            order.append(f"{name}_end")
            lock.release()

        t1 = asyncio.create_task(task("first", 0.05))
        await asyncio.sleep(0.01)  # ensure first starts first
        t2 = asyncio.create_task(task("second", 0.01))

        await asyncio.gather(t1, t2)

        assert order == ["first_start", "first_end", "second_start", "second_end"]

    async def test_release_on_exception_in_context(self):
        lock = GPULock.get()
        with pytest.raises(ValueError):
            async with lock:
                raise ValueError("boom")
        assert not lock.is_locked
