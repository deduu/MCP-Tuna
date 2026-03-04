"""Tests for AnthropicProvider."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentsoul.providers.anthropic import (
    AnthropicProvider,
    _convert_messages,
    _convert_tools,
    _extract_system,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_tool_use_block(tool_id: str, name: str, input_data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block


def _make_thinking_block(thinking: str) -> MagicMock:
    block = MagicMock()
    block.type = "thinking"
    block.thinking = thinking
    return block


def _make_response(content_blocks: list, stop_reason: str = "end_turn") -> MagicMock:
    resp = MagicMock()
    resp.content = content_blocks
    resp.stop_reason = stop_reason
    resp.usage = MagicMock()
    resp.usage.input_tokens = 100
    resp.usage.output_tokens = 50
    return resp


def _make_provider(mock_client: AsyncMock) -> AnthropicProvider:
    return AnthropicProvider(
        model_id="claude-sonnet-4-20250514",
        api_key="test-key",
        client=mock_client,
    )


# ---------------------------------------------------------------------------
# Unit tests — helper functions
# ---------------------------------------------------------------------------

class TestExtractSystem:
    def test_extracts_system_message(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        system, filtered = _extract_system(messages)
        assert system == "You are helpful."
        assert len(filtered) == 1
        assert filtered[0]["role"] == "user"

    def test_no_system_message(self):
        messages = [{"role": "user", "content": "Hi"}]
        system, filtered = _extract_system(messages)
        assert system is None
        assert len(filtered) == 1


class TestConvertMessages:
    def test_user_message(self):
        result = _convert_messages([{"role": "user", "content": "Hello"}])
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_tool_result_message(self):
        result = _convert_messages([
            {"role": "tool", "tool_call_id": "tc_123", "content": "result data"}
        ])
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "tool_result"
        assert result[0]["content"][0]["tool_use_id"] == "tc_123"

    def test_assistant_with_tool_calls(self):
        result = _convert_messages([
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [
                    {"id": "tc_1", "function": {"name": "search", "arguments": {"q": "test"}}}
                ],
            }
        ])
        blocks = result[0]["content"]
        assert blocks[0]["type"] == "text"
        assert blocks[1]["type"] == "tool_use"
        assert blocks[1]["name"] == "search"


class TestConvertTools:
    def test_openai_style_to_anthropic(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        result = _convert_tools(tools)
        assert result[0]["name"] == "get_weather"
        assert "input_schema" in result[0]
        assert result[0]["input_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# Integration tests — chat()
# ---------------------------------------------------------------------------

class TestAnthropicChat:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_response([_make_text_block("Hello!")])
        )
        provider = _make_provider(mock_client)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert result.content == "Hello!"
        assert result.tool_calls is None
        assert result.finish_reason == "end_turn"
        assert result.usage["prompt_tokens"] == 100

    @pytest.mark.asyncio
    async def test_tool_use_response(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_response(
                [_make_tool_use_block("tc_1", "search", {"query": "weather"})],
                stop_reason="tool_use",
            )
        )
        provider = _make_provider(mock_client)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Search"}],
            tools=[{"function": {"name": "search", "description": "Search", "parameters": {}}}],
        )

        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"
        assert result.tool_calls[0].arguments == {"query": "weather"}
        assert result.finish_reason == "tool_use"

    @pytest.mark.asyncio
    async def test_thinking_response(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_response([
                _make_thinking_block("Let me think..."),
                _make_text_block("The answer is 42."),
            ])
        )
        provider = _make_provider(mock_client)

        result = await provider.chat(
            messages=[{"role": "user", "content": "What is the answer?"}],
            enable_thinking=True,
        )

        assert result.content == "The answer is 42."
        assert result.thinking == "Let me think..."

    @pytest.mark.asyncio
    async def test_system_message_extracted(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_response([_make_text_block("ok")])
        )
        provider = _make_provider(mock_client)

        await provider.chat(messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ])

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be concise."
        assert all(m["role"] != "system" for m in call_kwargs["messages"])

    @pytest.mark.asyncio
    async def test_temperature_zero_with_tools(self):
        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(
            return_value=_make_response([_make_text_block("ok")])
        )
        provider = _make_provider(mock_client)

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"function": {"name": "t", "description": "", "parameters": {}}}],
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0


class TestAnthropicProviderMeta:
    def test_supports_tools(self):
        provider = AnthropicProvider(model_id="claude-sonnet-4-20250514", api_key="k")
        assert provider.supports_tools() is True

    def test_supports_thinking(self):
        provider = AnthropicProvider(model_id="claude-sonnet-4-20250514", api_key="k")
        assert provider.supports_thinking() is True

    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="api_key"):
            AnthropicProvider(model_id="claude-sonnet-4-20250514")
