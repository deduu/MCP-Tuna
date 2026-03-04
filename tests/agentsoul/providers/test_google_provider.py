"""Tests for GoogleGeminiProvider."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agentsoul.providers.google import (
    GoogleGeminiProvider,
    _convert_messages,
    _convert_tools,
    _extract_system,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_text_part(text: str) -> MagicMock:
    part = MagicMock()
    part.text = text
    part.function_call = None
    return part


def _make_function_call_part(name: str, args: dict) -> MagicMock:
    part = MagicMock()
    part.text = None
    fc = MagicMock()
    fc.name = name
    fc.args = args
    part.function_call = fc
    return part


def _make_response(parts: list, prompt_tokens: int = 100, completion_tokens: int = 50) -> MagicMock:
    content = MagicMock()
    content.parts = parts

    candidate = MagicMock()
    candidate.content = content

    usage = MagicMock()
    usage.prompt_token_count = prompt_tokens
    usage.candidates_token_count = completion_tokens
    usage.total_token_count = prompt_tokens + completion_tokens

    resp = MagicMock()
    resp.candidates = [candidate]
    resp.usage_metadata = usage
    return resp


def _make_provider(mock_client: MagicMock) -> GoogleGeminiProvider:
    return GoogleGeminiProvider(
        model_id="gemini-2.0-flash",
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

    def test_no_system_message(self):
        messages = [{"role": "user", "content": "Hi"}]
        system, filtered = _extract_system(messages)
        assert system is None
        assert len(filtered) == 1


class TestConvertMessages:
    def test_user_message(self):
        result = _convert_messages([{"role": "user", "content": "Hello"}])
        assert result[0].role == "user"

    def test_assistant_mapped_to_model(self):
        result = _convert_messages([{"role": "assistant", "content": "Hi!"}])
        assert result[0].role == "model"

    def test_tool_result_mapped_to_user(self):
        result = _convert_messages([
            {"role": "tool", "name": "search", "content": "result data"}
        ])
        assert result[0].role == "user"


class TestConvertTools:
    def test_openai_style_to_gemini(self):
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
        assert len(result) == 1  # One Tool wrapper
        decls = result[0].function_declarations
        assert decls[0].name == "get_weather"


# ---------------------------------------------------------------------------
# Integration tests — chat()
# ---------------------------------------------------------------------------

class TestGeminiChat:
    @pytest.mark.asyncio
    async def test_simple_text_response(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_response([_make_text_part("Hello!")])
        )
        provider = _make_provider(mock_client)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert result.content == "Hello!"
        assert result.tool_calls is None
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 100

    @pytest.mark.asyncio
    async def test_function_call_response(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_response([
                _make_function_call_part("search", {"query": "weather"})
            ])
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
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_mixed_text_and_function_call(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_response([
                _make_text_part("Let me search."),
                _make_function_call_part("search", {"q": "test"}),
            ])
        )
        provider = _make_provider(mock_client)

        result = await provider.chat(
            messages=[{"role": "user", "content": "Find info"}],
            tools=[{"function": {"name": "search", "description": "", "parameters": {}}}],
        )

        assert result.content == "Let me search."
        assert result.tool_calls is not None
        assert result.tool_calls[0].name == "search"

    @pytest.mark.asyncio
    async def test_system_instruction_passed(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_response([_make_text_part("ok")])
        )
        provider = _make_provider(mock_client)

        await provider.chat(messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ])

        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"].system_instruction == "Be concise."

    @pytest.mark.asyncio
    async def test_temperature_zero_with_tools(self):
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_response([_make_text_part("ok")])
        )
        provider = _make_provider(mock_client)

        await provider.chat(
            messages=[{"role": "user", "content": "Hi"}],
            tools=[{"function": {"name": "t", "description": "", "parameters": {}}}],
        )

        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"].temperature == 0.0


class TestGeminiProviderMeta:
    def test_supports_tools(self):
        provider = GoogleGeminiProvider(model_id="gemini-2.0-flash", api_key="k")
        assert provider.supports_tools() is True

    def test_supports_thinking(self):
        provider = GoogleGeminiProvider(model_id="gemini-2.0-flash", api_key="k")
        assert provider.supports_thinking() is True

    def test_requires_api_key(self):
        with pytest.raises(ValueError, match="api_key"):
            GoogleGeminiProvider(model_id="gemini-2.0-flash")
