from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import patch

import pytest


class _FakeFinetuner:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def check_resources(self) -> dict:
        time.sleep(0.05)
        self._events.append("tool")
        return {"success": True, "gpu": {"available": False}}


@pytest.mark.asyncio
async def test_system_check_resources_does_not_block_event_loop() -> None:
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway

        gateway = TunaGateway()

    events: list[str] = []
    gateway._finetuning_svc = _FakeFinetuner(events)

    async def marker() -> None:
        await asyncio.sleep(0.01)
        events.append("marker")

    tool = gateway.mcp._tools["system.check_resources"]["func"]
    result, _ = await asyncio.gather(tool(), marker())

    payload = json.loads(result)
    assert payload["success"] is True
    assert events == ["marker", "tool"]
