from __future__ import annotations

import asyncio
import time

import pytest

from shared.async_utils import call_maybe_async


@pytest.mark.asyncio
async def test_call_maybe_async_offloads_sync_work() -> None:
    events: list[str] = []

    def blocking() -> str:
        time.sleep(0.05)
        events.append("blocking")
        return "ok"

    async def marker() -> None:
        await asyncio.sleep(0.01)
        events.append("marker")

    result, _ = await asyncio.gather(call_maybe_async(blocking), marker())

    assert result == "ok"
    assert events == ["marker", "blocking"]
