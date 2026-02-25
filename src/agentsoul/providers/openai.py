# src/agentsoul/providers/openai.py
import json
import base64
import mimetypes
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, AsyncGenerator, Optional

from openai import AsyncOpenAI

from agentsoul.providers.base import BaseLLM
from agentsoul.core.models import LLMResponse, StreamChunk, ToolCall
from agentsoul.utils.logger import get_logger

logger = logging.getLogger(__name__)


def _guess_mime(name: str, fallback="image/jpeg") -> str:
    mime, _ = mimetypes.guess_type(name)
    return mime or fallback


def _to_data_url(path: str) -> str:
    p = Path(path)
    mime = _guess_mime(p.name)
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _is_public_http(url: str) -> bool:
    return url.startswith("https://") and "localhost" not in url


def _sanitize_openai_content(content: Any) -> List[Dict[str, Any]]:
    """Convert flexible content formats into OpenAI-compatible format."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, (int, float, bool)):
        return [{"type": "text", "text": str(content)}]
    out: List[Dict[str, Any]] = []
    for b in (content or []):
        t = b.get("type")

        if t == "text":
            txt = b.get("text") or ""
            if txt:
                out.append({"type": "text", "text": txt})

        elif t == "image_path":
            p = b.get("image_path") or b.get("path")
            if p and Path(p).exists():
                try:
                    data_url = _to_data_url(p)
                    out.append(
                        {"type": "image_url", "image_url": {"url": data_url}})
                except Exception:
                    logger.exception(
                        "Failed to convert image_path '%s' to data URL", p)

        elif t == "image_url":
            url = (b.get("image_url") or {}).get("url") or b.get("url")
            if not url:
                continue
            if _is_public_http(url) or url.startswith("data:"):
                detail = (b.get("image_url") or {}).get("detail")
                image_url = {"url": url}
                if detail in ("auto", "low", "high"):
                    image_url["detail"] = detail
                out.append({"type": "image_url", "image_url": image_url})
            else:
                logger.warning(
                    "Dropping image_url that is not public or data URL: %s", url)

        elif t == "image_base64":
            b64 = b.get("image_base64")
            if not b64:
                continue
            mime = b.get("mime_type") or "image/jpeg"
            data_url = f"data:{mime};base64,{b64}"
            out.append({"type": "image_url", "image_url": {"url": data_url}})

    if not out:
        out = [{"type": "text", "text": ""}]
    return out


def _sanitize_openai_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        # Handle both Message objects and dict messages
        if hasattr(m, 'to_dict'):
            m_dict = m.to_dict()
        else:
            m_dict = m

        role = m_dict.get("role", "user")
        content = m_dict.get("content")
        message_dict = {"role": role,
                        "content": _sanitize_openai_content(content)}

        # Handle assistant tool calls
        if role == "assistant" and "tool_calls" in m_dict and m_dict["tool_calls"]:
            openai_tool_calls = []
            for tc in m_dict["tool_calls"]:
                if hasattr(tc, "id"):  # our ToolCall dataclass
                    args = tc.arguments
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    openai_tool_calls.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": args},
                    })
                elif isinstance(tc, dict):
                    if "function" in tc:
                        # Already OpenAI-style dict
                        openai_tool_calls.append(tc)
                    else:
                        # Legacy dict {id, name, arguments}
                        args = tc.get("arguments")
                        if isinstance(args, dict):
                            args = json.dumps(args)
                        openai_tool_calls.append({
                            "id": tc.get("id"),
                            "type": "function",
                            "function": {"name": tc.get("name"), "arguments": args},
                        })
            message_dict["tool_calls"] = openai_tool_calls

        # Handle tool role (must include tool_call_id)
        if role == "tool":
            if "tool_call_id" in m_dict:
                message_dict["tool_call_id"] = m_dict["tool_call_id"]
            if "name" in m_dict:
                message_dict["name"] = m_dict["name"]

        out.append(message_dict)

    return out


def normalize_tools_for_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for t in tools:
        if "function" in t:
            normalized.append(t)
        else:
            normalized.append({
                "type": "function",
                "function": {k: v for k, v in t.items() if k != "type"}
            })
    return normalized


class OpenAIProvider(BaseLLM):

    supports_tool_calls_in_messages = True

    def __init__(
        self,
        model_id: str = "gpt-4o",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[AsyncOpenAI] = None,
        verbose: bool = False
    ):
        super().__init__(model_id=model_id)
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__, verbose=verbose)
        if client:
            self.client = client
        else:
            if not api_key:
                raise ValueError(
                    "You must provide an api_key if no client is supplied."
                )
            self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False,
        **kwargs: Any,
    ) -> LLMResponse:
        self.logger.debug(f"input to LLM -> {messages}")
        message_list = _sanitize_openai_messages(messages)

        temperature = kwargs.pop("temperature", 0.7)
        request_params = {
            "model": self.model_id,
            "messages": message_list,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "temperature": temperature,
            **kwargs,
        }

        if tools:
            request_params["tools"] = normalize_tools_for_openai(tools)
            request_params["temperature"] = 0.0  # deterministic for tool calls

        try:
            response = await self.client.chat.completions.create(**request_params)
            message = response.choices[0].message
            tool_calls = None

            # Handle modern tool_calls format (preferred)
            if message.tool_calls:
                tool_calls = []
                for tc in message.tool_calls:
                    tool_calls.append(
                        ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=json.loads(tc.function.arguments or "{}")
                        )
                    )

            # Handle legacy function_call format (fallback)
            elif message.function_call:
                fc = message.function_call
                tool_calls = [
                    ToolCall(
                        id="fc_" + fc.name,   # synthetic id
                        name=fc.name,
                        arguments=json.loads(fc.arguments or "{}")
                    )
                ]

            return LLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                thinking="OpenAI does not expose thinking",
                finish_reason=response.choices[0].finish_reason,
                usage=response.usage.model_dump() if response.usage else None
            )

        except Exception:
            logger.exception(
                "OpenAI chat() failed with payload=%s", request_params)
            raise

    async def stream(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:

        self.logger.debug(f"input to LLM -> {messages}")
        message_list = _sanitize_openai_messages(messages)

        temperature = kwargs.pop("temperature", 0.7)
        request_params = {
            "model": self.model_id,
            "messages": message_list,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": kwargs.pop("max_tokens", 4096),
            "temperature": temperature,
            **kwargs,
        }
        if tools:
            request_params["tools"] = normalize_tools_for_openai(tools)
            request_params["temperature"] = 0.0  # deterministic for tool calls

        try:
            stream_response = await self.client.chat.completions.create(**request_params)

            tool_call_accumulators: Dict[int, Dict[str, Any]] = {}
            usage_data = None

            async for chunk in stream_response:

                if not chunk.choices:
                    if hasattr(chunk, 'usage') and chunk.usage:
                        usage_data = chunk.usage.model_dump()
                    continue
                choice = chunk.choices[0]
                delta = choice.delta

                # Handle tool calls
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index
                        acc = tool_call_accumulators.setdefault(idx, {
                            "id": None,
                            "name": None,
                            "arguments": ""
                        })

                        if tc.id:
                            acc["id"] = tc.id
                        if tc.function and tc.function.name:
                            acc["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            acc["arguments"] += tc.function.arguments

                # When tool calls finish, yield the complete ones
                if choice.finish_reason == "tool_calls":
                    tool_calls = []
                    for acc in tool_call_accumulators.values():
                        try:
                            parsed_args = json.loads(acc["arguments"])
                        except Exception:
                            parsed_args = {}
                        tool_calls.append(
                            ToolCall(
                                id=acc["id"],
                                name=acc["name"],
                                arguments=parsed_args
                            )
                        )
                    yield StreamChunk(content=None, tool_calls=tool_calls, finish_reason="tool_calls", usage=usage_data)
                    tool_call_accumulators.clear()

                # Normal text
                if delta.content:
                    yield StreamChunk(content=delta.content, tool_calls=None, finish_reason=None)

                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_data = chunk.usage.model_dump()

            if usage_data:
                yield StreamChunk(
                    content=None,
                    tool_calls=None,
                    finish_reason="stop",
                    usage=usage_data
                )
            self.logger.debug("Completed streaming with usage: %s", usage_data)

        except Exception:
            logger.exception(
                "OpenAI stream() failed with payload=%s", request_params)
            raise

    def supports_tools(self) -> bool:
        return True

    def supports_thinking(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return True

    def get_message_format_config(self) -> Dict[str, Any]:
        return {
            'supports_tool_calls_in_assistant': True,
            'tool_message_format': 'full'
        }
