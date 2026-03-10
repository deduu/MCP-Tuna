from __future__ import annotations

import json
from unittest.mock import patch

import pytest


class _FakeChatSession:
    last_config = None
    last_provider = None

    def __init__(self, config, provider=None) -> None:
        type(self).last_config = config
        type(self).last_provider = provider
        self._turns = 0

    async def initialize(self):
        return {"success": True}

    async def send_message(self, message: str) -> str:
        self._turns += 1
        return f"echo:{message}"

    def get_info(self):
        return {"turns": self._turns}


@pytest.mark.asyncio
async def test_host_chat_uses_live_provider_for_mcp_deployment():
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway

    gateway = TunaGateway()
    provider = object()
    gateway.hoster._deployments["dep-mcp"] = {
        "id": "dep-mcp",
        "type": "mcp",
        "model_path": "base/model",
        "adapter_path": "./adapter",
        "provider": provider,
        "transport": "http",
        "host": "127.0.0.1",
        "port": 8001,
    }

    host_chat = gateway.mcp._tools["host.chat"]["func"]

    with patch("hosting_pipeline.services.chat_service.ChatSession", _FakeChatSession):
        result = json.loads(await host_chat(message="hello", deployment_id="dep-mcp"))

    assert result["success"] is True
    assert result["deployment_id"] == "dep-mcp"
    assert result["response"] == "echo:hello"
    assert _FakeChatSession.last_provider is provider
    assert _FakeChatSession.last_config.endpoint is None
    assert _FakeChatSession.last_config.model_path == "base/model"
    assert _FakeChatSession.last_config.adapter_path == "./adapter"


@pytest.mark.asyncio
async def test_host_chat_uses_endpoint_for_api_deployment():
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway

    gateway = TunaGateway()
    gateway.hoster._deployments["dep-api"] = {
        "id": "dep-api",
        "type": "api",
        "model_path": "base/model",
        "adapter_path": "./adapter",
        "transport": "http",
        "host": "127.0.0.1",
        "port": 8010,
    }

    host_chat = gateway.mcp._tools["host.chat"]["func"]

    with patch("hosting_pipeline.services.chat_service.ChatSession", _FakeChatSession):
        result = json.loads(await host_chat(message="hello", deployment_id="dep-api"))

    assert result["success"] is True
    assert result["deployment_id"] == "dep-api"
    assert _FakeChatSession.last_provider is None
    assert _FakeChatSession.last_config.endpoint == "http://127.0.0.1:8010"
    assert _FakeChatSession.last_config.model_path is None
