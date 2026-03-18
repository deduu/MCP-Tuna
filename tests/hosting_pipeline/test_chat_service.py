"""Unit tests for the chat REPL service."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config import ChatConfig
from hosting_pipeline.services.chat_service import ChatSession


# ──────────────────────────────────────────────
# ChatConfig tests
# ──────────────────────────────────────────────


class TestChatConfig:
    def test_defaults(self):
        cfg = ChatConfig()
        assert cfg.endpoint is None
        assert cfg.model_path is None
        assert cfg.adapter_path is None
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.7
        assert cfg.system_prompt is None
        assert cfg.streaming is True
        assert cfg.modality == "text"
        assert cfg.api_path is None

    def test_api_mode_config(self):
        cfg = ChatConfig(endpoint="http://localhost:8001")
        assert cfg.endpoint == "http://localhost:8001"

    def test_direct_mode_config(self):
        cfg = ChatConfig(model_path="Qwen/Qwen3-1.7B", adapter_path="./output/lora")
        assert cfg.model_path == "Qwen/Qwen3-1.7B"
        assert cfg.adapter_path == "./output/lora"

    def test_custom_params(self):
        cfg = ChatConfig(
            max_new_tokens=256,
            temperature=0.3,
            system_prompt="You are a helpful assistant.",
            streaming=False,
        )
        assert cfg.max_new_tokens == 256
        assert cfg.temperature == 0.3
        assert cfg.system_prompt == "You are a helpful assistant."
        assert cfg.streaming is False


# ──────────────────────────────────────────────
# ChatSession — API mode
# ──────────────────────────────────────────────


class TestChatSessionAPIMode:
    @pytest.mark.asyncio
    async def test_initialize_api_mode_checks_health(self):
        cfg = ChatConfig(endpoint="http://localhost:8001")
        session = ChatSession(cfg)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok", "model": "test-model"}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client_cls.return_value = mock_client

            info = await session.initialize()

        assert info["mode"] == "api"
        assert info["endpoint"] == "http://localhost:8001"

    @pytest.mark.asyncio
    async def test_send_message_api_mode(self):
        cfg = ChatConfig(endpoint="http://localhost:8001")
        session = ChatSession(cfg)
        session._mode = "api"
        session._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello! How can I help?"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        session._http_client = mock_client

        result = await session.send_message("Hello")

        assert result == "Hello! How can I help?"
        assert len(session._history) == 2  # user + assistant
        assert session._history[0]["role"] == "user"
        assert session._history[1]["role"] == "assistant"
        mock_client.post.assert_awaited_once()
        assert mock_client.post.await_args.args[0] == "http://localhost:8001/generate"

    @pytest.mark.asyncio
    async def test_send_message_result_api_mode_returns_metrics(self):
        cfg = ChatConfig(endpoint="http://localhost:8001")
        session = ChatSession(cfg)
        session._mode = "api"
        session._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Hello! How can I help?",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "api-model",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        session._http_client = mock_client

        result = await session.send_message_result("Hello")

        assert result["response"] == "Hello! How can I help?"
        assert result["usage"]["total_tokens"] == 15
        assert result["model_id"] == "api-model"
        assert result["metrics"]["latency_ms"] is not None

    @pytest.mark.asyncio
    async def test_conversation_history_maintained(self):
        cfg = ChatConfig(endpoint="http://localhost:8001")
        session = ChatSession(cfg)
        session._mode = "api"
        session._initialized = True

        mock_client = AsyncMock()
        session._http_client = mock_client

        # First message
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"response": "Hi there!"}
        mock_client.post = AsyncMock(return_value=resp1)
        await session.send_message("Hello")

        # Second message
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"response": "I'm an AI assistant."}
        mock_client.post = AsyncMock(return_value=resp2)
        await session.send_message("Who are you?")

        assert len(session._history) == 4
        assert session._history[0]["content"] == "Hello"
        assert session._history[1]["content"] == "Hi there!"
        assert session._history[2]["content"] == "Who are you?"
        assert session._history[3]["content"] == "I'm an AI assistant."

    @pytest.mark.asyncio
    async def test_system_prompt_in_history(self):
        cfg = ChatConfig(
            endpoint="http://localhost:8001",
            system_prompt="You are a pirate.",
        )
        session = ChatSession(cfg)
        session._mode = "api"
        session._initialized = True

        mock_client = AsyncMock()
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"response": "Ahoy!"}
        mock_client.post = AsyncMock(return_value=resp)
        session._http_client = mock_client

        await session.send_message("Greet me")

        # system prompt should be first in history
        assert session._history[0]["role"] == "system"
        assert session._history[0]["content"] == "You are a pirate."


# ──────────────────────────────────────────────
# ChatSession — Direct mode
# ──────────────────────────────────────────────


class TestChatSessionDirectMode:
    @pytest.mark.asyncio
    async def test_initialize_direct_mode_loads_model(self):
        cfg = ChatConfig(model_path="test/model")
        session = ChatSession(cfg)

        mock_provider = MagicMock()

        with patch(
            "agentsoul.providers.hf.HuggingFaceProvider",
            return_value=mock_provider,
        ) as mock_cls:
            info = await session.initialize()

        mock_cls.assert_called_once_with(model_path="test/model", lora_adapter_path=None)
        assert info["mode"] == "direct"
        assert info["model_path"] == "test/model"

    @pytest.mark.asyncio
    async def test_initialize_direct_mode_with_adapter(self):
        cfg = ChatConfig(model_path="test/model", adapter_path="./lora")
        session = ChatSession(cfg)

        mock_provider = MagicMock()

        with patch(
            "agentsoul.providers.hf.HuggingFaceProvider",
            return_value=mock_provider,
        ) as mock_cls:
            info = await session.initialize()

        mock_cls.assert_called_once_with(model_path="test/model", lora_adapter_path="./lora")
        assert info["adapter_path"] == "./lora"

    @pytest.mark.asyncio
    async def test_initialize_direct_mode_reuses_existing_provider(self):
        cfg = ChatConfig(model_path="test/model", adapter_path="./lora")
        shared_provider = MagicMock()
        session = ChatSession(cfg, provider=shared_provider)

        with patch("agentsoul.providers.hf.HuggingFaceProvider") as mock_cls:
            info = await session.initialize()

        mock_cls.assert_not_called()
        assert session._provider is shared_provider
        assert info["shared_provider"] is True

    @pytest.mark.asyncio
    async def test_send_message_direct_mode(self):
        cfg = ChatConfig(model_path="test/model")
        session = ChatSession(cfg)
        session._mode = "direct"
        session._initialized = True

        mock_resp = MagicMock()
        mock_resp.content = "Direct response"

        mock_provider = AsyncMock()
        mock_provider.chat = AsyncMock(return_value=mock_resp)
        session._provider = mock_provider

        result = await session.send_message("Hi")

        assert result == "Direct response"
        assert len(session._history) == 2

    @pytest.mark.asyncio
    async def test_stream_message_direct_mode(self):
        cfg = ChatConfig(model_path="test/model")
        session = ChatSession(cfg)
        session._mode = "direct"
        session._initialized = True

        async def fake_stream(*args, **kwargs):
            for token in ["Hello", " world", "!"]:
                yield token

        mock_provider = MagicMock()
        mock_provider.stream = fake_stream
        session._provider = mock_provider

        tokens = []
        async for token in session.stream_message("Hi"):
            tokens.append(token)

        assert tokens == ["Hello", " world", "!"]
        assert len(session._history) == 2
        assert session._history[1]["content"] == "Hello world!"

    @pytest.mark.asyncio
    async def test_stream_message_api_mode_falls_back_to_full_response(self):
        cfg = ChatConfig(endpoint="http://localhost:8001")
        session = ChatSession(cfg)
        session._mode = "api"
        session._initialized = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Hello from API"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        session._http_client = mock_client

        tokens = []
        async for token in session.stream_message("Hi"):
            tokens.append(token)

        assert tokens == ["Hello from API"]
        assert session._history[-1]["content"] == "Hello from API"

    @pytest.mark.asyncio
    async def test_shutdown_direct_mode(self):
        cfg = ChatConfig(model_path="test/model")
        session = ChatSession(cfg)
        session._mode = "direct"
        session._initialized = True
        session._provider = MagicMock()

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            await session.shutdown()

        assert session._provider is None
        mock_torch.cuda.empty_cache.assert_called_once()


# ──────────────────────────────────────────────
# ChatSession — Commands & utilities
# ──────────────────────────────────────────────


class TestChatSessionCommands:
    def test_clear_history_resets_messages(self):
        cfg = ChatConfig(system_prompt="Be helpful.")
        session = ChatSession(cfg)
        session._history = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        session.clear_history()

        # system prompt should be preserved
        assert len(session._history) == 1
        assert session._history[0]["role"] == "system"

    def test_clear_history_no_system_prompt(self):
        cfg = ChatConfig()
        session = ChatSession(cfg)
        session._history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        session.clear_history()
        assert len(session._history) == 0

    def test_get_info_api_mode(self):
        cfg = ChatConfig(endpoint="http://localhost:8001")
        session = ChatSession(cfg)
        session._mode = "api"
        session._initialized = True
        session._history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        info = session.get_info()

        assert info["mode"] == "api"
        assert info["endpoint"] == "http://localhost:8001"
        assert info["turns"] == 1

    def test_get_info_direct_mode(self):
        cfg = ChatConfig(model_path="test/model", adapter_path="./lora")
        session = ChatSession(cfg)
        session._mode = "direct"
        session._initialized = True

        info = session.get_info()

        assert info["mode"] == "direct"
        assert info["model_path"] == "test/model"
        assert info["adapter_path"] == "./lora"
        assert info["turns"] == 0

    @pytest.mark.asyncio
    async def test_shutdown_api_mode_closes_client(self):
        cfg = ChatConfig(endpoint="http://localhost:8001")
        session = ChatSession(cfg)
        session._mode = "api"
        session._initialized = True
        session._http_client = AsyncMock()

        await session.shutdown()

        session._http_client.aclose.assert_called_once()


class TestChatSessionVLM:
    @pytest.mark.asyncio
    async def test_send_messages_api_mode_uses_generate_vlm_route(self):
        cfg = ChatConfig(
            endpoint="http://localhost:8001",
            modality="vision-language",
            api_path="/generate_vlm",
        )
        session = ChatSession(cfg)
        session._mode = "api"
        session._initialized = True

        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Image summary"}
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        session._http_client = mock_client

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_path", "image_path": "uploads/images/example.png"},
                    {"type": "text", "text": "Describe the image."},
                ],
            }
        ]

        result = await session.send_messages(messages)

        assert result == "Image summary"
        mock_client.post.assert_awaited_once()
        assert mock_client.post.await_args.args[0] == "http://localhost:8001/generate_vlm"
        assert session._history[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_send_messages_direct_mode_uses_inference_service(self):
        cfg = ChatConfig(model_path="test/vlm", modality="vision-language")
        inference_service = AsyncMock()
        inference_service.run_vlm_inference = AsyncMock(
            return_value={"success": True, "response": "Detected a crack"}
        )
        session = ChatSession(cfg, inference_service=inference_service)
        session._mode = "direct"
        session._initialized = True

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_path", "image_path": "uploads/images/example.png"},
                    {"type": "text", "text": "Describe the defect."},
                ],
            }
        ]

        result = await session.send_messages(messages)

        assert result == "Detected a crack"
        inference_service.run_vlm_inference.assert_awaited_once()
        assert session._history[-1]["content"][0]["text"] == "Detected a crack"

    @pytest.mark.asyncio
    async def test_send_message_rejects_vlm_sessions(self):
        cfg = ChatConfig(model_path="test/vlm", modality="vision-language")
        session = ChatSession(cfg)
        session._mode = "direct"
        session._initialized = True

        with pytest.raises(ValueError, match="send_message only supports text chats"):
            await session.send_message("hello")
