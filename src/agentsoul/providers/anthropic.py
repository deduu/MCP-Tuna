# src/agentsoul/providers/anthropic.py
from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

from anthropic import AsyncAnthropic

from agentsoul.providers.base import BaseLLM
from agentsoul.core.models import LLMResponse, StreamChunk, ToolCall
from agentsoul.utils.logger import get_logger

logger = logging.getLogger(__name__)


def _extract_system(messages: Iterable[Dict[str, Any]]) -> tuple[str | None, list[Dict[str, Any]]]:
    """Separate system message from the rest (Anthropic takes system as a kwarg)."""
    system: str | None = None
    filtered: list[Dict[str, Any]] = []
    for m in messages:
        if hasattr(m, "to_dict"):
            m = m.to_dict()
        if m.get("role") == "system":
            system = m.get("content", "")
        else:
            filtered.append(m)
    return system, filtered


def _convert_messages(messages: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Convert provider-agnostic messages to Anthropic format."""
    out: list[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")

        # Anthropic uses "user" for tool results
        if role == "tool":
            out.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": m.get("content", ""),
                    }
                ],
            })
            continue

        content = m.get("content", "")
        entry: Dict[str, Any] = {"role": role, "content": content}

        # Preserve assistant tool_calls as tool_use content blocks
        if role == "assistant" and m.get("tool_calls"):
            blocks: list[Dict[str, Any]] = []
            if content:
                blocks.append({"type": "text", "text": content if isinstance(content, str) else str(content)})
            for tc in m["tool_calls"]:
                if hasattr(tc, "id"):
                    blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
                elif isinstance(tc, dict):
                    func = tc.get("function", tc)
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args,
                    })
            entry["content"] = blocks

        out.append(entry)
    return out


def _convert_tools(tools: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Convert OpenAI-style tool schemas to Anthropic format."""
    converted: list[Dict[str, Any]] = []
    for t in tools:
        func = t.get("function", t)
        converted.append({
            "name": func["name"],
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


class AnthropicProvider(BaseLLM):

    supports_tool_calls_in_messages = True

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        client: Optional[AsyncAnthropic] = None,
        verbose: bool = False,
    ):
        super().__init__(model_id=model_id)
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__, verbose=verbose)
        if client:
            self.client = client
        else:
            if not api_key:
                raise ValueError("You must provide an api_key if no client is supplied.")
            kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = AsyncAnthropic(**kwargs)

    async def chat(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False,
        **kwargs: Any,
    ) -> LLMResponse:
        self.logger.debug("input to LLM -> %s", messages)
        system, filtered = _extract_system(messages)
        anthropic_messages = _convert_messages(filtered)

        temperature = kwargs.pop("temperature", 0.7)
        request_params: Dict[str, Any] = {
            "model": self.model_id,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system:
            request_params["system"] = system

        if enable_thinking:
            request_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}
        else:
            request_params["temperature"] = temperature

        if tools:
            request_params["tools"] = _convert_tools(tools)
            if not enable_thinking:
                request_params["temperature"] = 0.0

        try:
            response = await self.client.messages.create(**request_params)

            content_text: list[str] = []
            thinking_text: list[str] = []
            tool_calls: list[ToolCall] | None = None

            for block in response.content:
                if block.type == "text":
                    content_text.append(block.text)
                elif block.type == "thinking":
                    thinking_text.append(block.thinking)
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append(
                        ToolCall(id=block.id, name=block.name, arguments=block.input)
                    )

            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                }

            return LLMResponse(
                content="\n".join(content_text) if content_text else None,
                tool_calls=tool_calls,
                thinking="\n".join(thinking_text) if thinking_text else None,
                finish_reason=response.stop_reason,
                usage=usage,
            )

        except Exception:
            logger.exception("Anthropic chat() failed with payload=%s", request_params)
            raise

    async def stream(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        self.logger.debug("input to LLM -> %s", messages)
        system, filtered = _extract_system(messages)
        anthropic_messages = _convert_messages(filtered)

        temperature = kwargs.pop("temperature", 0.7)
        request_params: Dict[str, Any] = {
            "model": self.model_id,
            "messages": anthropic_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system:
            request_params["system"] = system

        if enable_thinking:
            request_params["thinking"] = {"type": "enabled", "budget_tokens": 10000}
        else:
            request_params["temperature"] = temperature

        if tools:
            request_params["tools"] = _convert_tools(tools)
            if not enable_thinking:
                request_params["temperature"] = 0.0

        try:
            async with self.client.messages.stream(**request_params) as stream:
                # Accumulators for tool use blocks
                current_tool: Dict[str, Any] | None = None
                tool_calls: list[ToolCall] = []

                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool = {"id": block.id, "name": block.name, "arguments": ""}

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield StreamChunk(content=delta.text, tool_calls=None, finish_reason=None)
                        elif delta.type == "thinking_delta":
                            # Yield thinking as content with a marker (consumers can filter)
                            pass
                        elif delta.type == "input_json_delta" and current_tool is not None:
                            current_tool["arguments"] += delta.partial_json

                    elif event.type == "content_block_stop":
                        if current_tool is not None:
                            try:
                                parsed = json.loads(current_tool["arguments"])
                            except (json.JSONDecodeError, ValueError):
                                parsed = {}
                            tool_calls.append(
                                ToolCall(
                                    id=current_tool["id"],
                                    name=current_tool["name"],
                                    arguments=parsed,
                                )
                            )
                            current_tool = None

                    elif event.type == "message_delta":
                        usage = None
                        if hasattr(event, "usage") and event.usage:
                            usage = {
                                "prompt_tokens": getattr(event.usage, "input_tokens", 0),
                                "completion_tokens": getattr(event.usage, "output_tokens", 0),
                            }
                        stop_reason = getattr(event.delta, "stop_reason", None)
                        if tool_calls and stop_reason == "tool_use":
                            yield StreamChunk(
                                content=None,
                                tool_calls=tool_calls,
                                finish_reason="tool_calls",
                                usage=usage,
                            )
                            tool_calls = []
                        elif stop_reason:
                            yield StreamChunk(
                                content=None,
                                tool_calls=None,
                                finish_reason="stop",
                                usage=usage,
                            )

        except Exception:
            logger.exception("Anthropic stream() failed with payload=%s", request_params)
            raise

    def supports_tools(self) -> bool:
        return True

    def supports_thinking(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return True

    def get_message_format_config(self) -> Dict[str, Any]:
        return {
            "supports_tool_calls_in_assistant": True,
            "tool_message_format": "full",
        }
