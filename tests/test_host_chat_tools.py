from __future__ import annotations

import json
from unittest.mock import patch

import pytest


class _FakeChatSession:
    last_config = None
    last_provider = None
    last_inference_service = None
    last_messages = None

    def __init__(self, config, provider=None, inference_service=None) -> None:
        type(self).last_config = config
        type(self).last_provider = provider
        type(self).last_inference_service = inference_service
        self._turns = 0

    async def initialize(self):
        return {"success": True}

    async def send_message(self, message: str) -> str:
        self._turns += 1
        return f"echo:{message}"

    async def send_messages(self, messages):
        type(self).last_messages = messages
        self._turns += 1
        return "echo-vlm"

    def get_info(self):
        return {"turns": self._turns, "modality": self.last_config.modality}


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


@pytest.mark.asyncio
async def test_host_chat_normalizes_wildcard_host_for_api_deployment():
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway

    gateway = TunaGateway()
    gateway.hoster._deployments["dep-api-wildcard"] = {
        "id": "dep-api-wildcard",
        "type": "api",
        "model_path": "base/model",
        "adapter_path": None,
        "transport": "http",
        "host": "0.0.0.0",
        "port": 8011,
    }

    host_chat = gateway.mcp._tools["host.chat"]["func"]

    with patch("hosting_pipeline.services.chat_service.ChatSession", _FakeChatSession):
        result = json.loads(await host_chat(message="hello", deployment_id="dep-api-wildcard"))

    assert result["success"] is True
    assert _FakeChatSession.last_config.endpoint == "http://127.0.0.1:8011"


@pytest.mark.asyncio
async def test_host_chat_vlm_uses_endpoint_for_api_deployment():
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway

    gateway = TunaGateway()
    gateway.hoster._deployments["dep-vlm-api"] = {
        "id": "dep-vlm-api",
        "type": "api",
        "modality": "vision-language",
        "model_path": "base/vlm",
        "adapter_path": "./adapter",
        "transport": "http",
        "host": "127.0.0.1",
        "port": 8012,
        "api_path": "/generate_vlm",
    }

    host_chat_vlm = gateway.mcp._tools["host.chat_vlm"]["func"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_path", "image_path": "uploads/images/example.png"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    with patch("hosting_pipeline.services.chat_service.ChatSession", _FakeChatSession):
        result = json.loads(await host_chat_vlm(messages=messages, deployment_id="dep-vlm-api"))

    assert result["success"] is True
    assert result["response"] == "echo-vlm"
    assert _FakeChatSession.last_config.endpoint == "http://127.0.0.1:8012"
    assert _FakeChatSession.last_config.modality == "vision-language"
    assert _FakeChatSession.last_config.api_path == "/generate_vlm"
    assert _FakeChatSession.last_messages == messages


@pytest.mark.asyncio
async def test_host_chat_vlm_uses_inference_service_for_mcp_deployment():
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway

    gateway = TunaGateway()
    inference_service = object()
    gateway.hoster._deployments["dep-vlm-mcp"] = {
        "id": "dep-vlm-mcp",
        "type": "mcp",
        "modality": "vision-language",
        "model_path": "base/vlm",
        "adapter_path": "./adapter",
        "transport": "http",
        "host": "127.0.0.1",
        "port": 8013,
        "inference_service": inference_service,
    }

    host_chat_vlm = gateway.mcp._tools["host.chat_vlm"]["func"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_path", "image_path": "uploads/images/example.png"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    with patch("hosting_pipeline.services.chat_service.ChatSession", _FakeChatSession):
        result = json.loads(await host_chat_vlm(messages=messages, deployment_id="dep-vlm-mcp"))

    assert result["success"] is True
    assert _FakeChatSession.last_config.model_path == "base/vlm"
    assert _FakeChatSession.last_config.adapter_path == "./adapter"
    assert _FakeChatSession.last_inference_service is inference_service


@pytest.mark.asyncio
async def test_host_chat_rejects_vlm_deployments():
    with patch("mcp_gateway.load_dotenv"):
        from mcp_gateway import TunaGateway

    gateway = TunaGateway()
    gateway.hoster._deployments["dep-vlm-api"] = {
        "id": "dep-vlm-api",
        "type": "api",
        "modality": "vision-language",
        "model_path": "base/vlm",
        "adapter_path": "./adapter",
        "transport": "http",
        "host": "127.0.0.1",
        "port": 8014,
    }

    host_chat = gateway.mcp._tools["host.chat"]["func"]
    result = json.loads(await host_chat(message="hello", deployment_id="dep-vlm-api"))

    assert result["success"] is False
    assert "host.chat_vlm" in result["error"]
