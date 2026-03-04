# src/agentsoul/providers/google.py
from __future__ import annotations

import logging
import uuid
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional

from google import genai
from google.genai import types

from agentsoul.providers.base import BaseLLM
from agentsoul.core.models import LLMResponse, StreamChunk, ToolCall
from agentsoul.utils.logger import get_logger

logger = logging.getLogger(__name__)


def _extract_system(messages: Iterable[Dict[str, Any]]) -> tuple[str | None, list[Dict[str, Any]]]:
    """Separate system message (Gemini takes system_instruction as config)."""
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


def _convert_messages(messages: list[Dict[str, Any]]) -> list[types.Content]:
    """Convert provider-agnostic messages to Gemini Content objects."""
    contents: list[types.Content] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        # Map roles: assistant -> model, tool -> user with function_response
        if role == "tool":
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name=m.get("name", "tool"),
                            response={"result": content if isinstance(content, str) else str(content)},
                        )
                    ],
                )
            )
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts: list[types.Part] = []

        # Handle text content
        if isinstance(content, str) and content:
            parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(types.Part.from_text(text=block.get("text", "")))

        # Handle assistant tool calls as function_call parts
        if role == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if hasattr(tc, "name"):
                    parts.append(types.Part.from_function_call(
                        name=tc.name, args=tc.arguments,
                    ))
                elif isinstance(tc, dict):
                    func = tc.get("function", tc)
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        import json
                        args = json.loads(args)
                    parts.append(types.Part.from_function_call(
                        name=func.get("name", ""), args=args,
                    ))

        if not parts:
            parts.append(types.Part.from_text(text=""))

        contents.append(types.Content(role=gemini_role, parts=parts))
    return contents


def _convert_tools(tools: list[Dict[str, Any]]) -> list[types.Tool]:
    """Convert OpenAI-style tool schemas to Gemini FunctionDeclaration format."""
    declarations: list[types.FunctionDeclaration] = []
    for t in tools:
        func = t.get("function", t)
        params = func.get("parameters", {})

        declarations.append(
            types.FunctionDeclaration(
                name=func["name"],
                description=func.get("description", ""),
                parameters=params,
            )
        )
    return [types.Tool(function_declarations=declarations)]


class GoogleGeminiProvider(BaseLLM):

    supports_tool_calls_in_messages = True

    def __init__(
        self,
        model_id: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        client: Optional[genai.Client] = None,
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
            self.client = genai.Client(api_key=api_key)

    async def chat(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False,
        **kwargs: Any,
    ) -> LLMResponse:
        self.logger.debug("input to LLM -> %s", messages)
        system, filtered = _extract_system(messages)
        contents = _convert_messages(filtered)

        temperature = kwargs.pop("temperature", 0.7)
        config_params: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": kwargs.pop("max_tokens", 4096),
        }

        if system:
            config_params["system_instruction"] = system

        if tools:
            config_params["tools"] = _convert_tools(tools)
            config_params["temperature"] = 0.0

        if enable_thinking:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=10000)

        config = types.GenerateContentConfig(**config_params)

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )

            text_parts: list[str] = []
            tool_calls: list[ToolCall] | None = None

            if response.candidates and response.candidates[0].content:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        if tool_calls is None:
                            tool_calls = []
                        fc = part.function_call
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                            )
                        )

            usage = None
            if response.usage_metadata:
                um = response.usage_metadata
                usage = {
                    "prompt_tokens": getattr(um, "prompt_token_count", 0),
                    "completion_tokens": getattr(um, "candidates_token_count", 0),
                }

            finish_reason = "tool_calls" if tool_calls else "stop"

            return LLMResponse(
                content="\n".join(text_parts) if text_parts else None,
                tool_calls=tool_calls,
                thinking=None,
                finish_reason=finish_reason,
                usage=usage,
            )

        except Exception:
            logger.exception("Gemini chat() failed for model=%s", self.model_id)
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
        contents = _convert_messages(filtered)

        temperature = kwargs.pop("temperature", 0.7)
        config_params: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": kwargs.pop("max_tokens", 4096),
        }

        if system:
            config_params["system_instruction"] = system

        if tools:
            config_params["tools"] = _convert_tools(tools)
            config_params["temperature"] = 0.0

        if enable_thinking:
            config_params["thinking_config"] = types.ThinkingConfig(thinking_budget=10000)

        config = types.GenerateContentConfig(**config_params)

        try:
            async for chunk in await self.client.aio.models.generate_content_stream(
                model=self.model_id,
                contents=contents,
                config=config,
            ):
                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    continue

                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        yield StreamChunk(content=part.text, tool_calls=None, finish_reason=None)
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        yield StreamChunk(
                            content=None,
                            tool_calls=[
                                ToolCall(
                                    id=f"call_{uuid.uuid4().hex[:8]}",
                                    name=fc.name,
                                    arguments=dict(fc.args) if fc.args else {},
                                )
                            ],
                            finish_reason="tool_calls",
                        )

                # Emit usage on last chunk
                if chunk.usage_metadata:
                    um = chunk.usage_metadata
                    usage = {
                        "prompt_tokens": getattr(um, "prompt_token_count", 0),
                        "completion_tokens": getattr(um, "candidates_token_count", 0),
                    }
                    yield StreamChunk(
                        content=None,
                        tool_calls=None,
                        finish_reason="stop",
                        usage=usage,
                    )

        except Exception:
            logger.exception("Gemini stream() failed for model=%s", self.model_id)
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
