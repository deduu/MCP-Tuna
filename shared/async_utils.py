"""Helpers for keeping async integration boundaries non-blocking."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
from typing import Any, Callable


async def run_sync(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    """Run synchronous work in a worker thread while preserving contextvars."""
    ctx = contextvars.copy_context()
    call = functools.partial(ctx.run, func, *args, **kwargs)
    return await asyncio.to_thread(call)


async def call_maybe_async(
    func: Callable[..., Any], /, *args: Any, **kwargs: Any,
) -> Any:
    """Await async callables and offload sync callables."""
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await run_sync(func, *args, **kwargs)
