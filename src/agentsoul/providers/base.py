from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional
from typing import AsyncGenerator
from agentsoul.core.models import LLMResponse, StreamChunk


class BaseLLM(ABC):
    """
    A unified contract every provider must satisfy.
    Single method handles both regular chat and tool-enabled conversations.
    """
    supports_tool_calls_in_messages: bool = True
    def __init__(self, model_id: str = "unknown"):
        self.model_id = model_id

    @abstractmethod
    async def chat(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Generate a chat response, optionally with tool support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of available tools in OpenAI function format
            enable_thinking: Whether to enable reasoning/thinking
            **kwargs: Provider-specific parameters

        Returns:
            LLMResponse object with content and/or tool_calls
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False,
        **kwargs: Any
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream chat response chunks.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of available tools
            **kwargs: Provider-specific parameters

        Yields:
            StreamChunk objects with incremental content/tool_calls
        """
        ...

    @abstractmethod
    def supports_tools(self) -> bool:
        """Whether the provider supports tool calls"""
        ...

    @abstractmethod
    def supports_thinking(self) -> bool:
        """Whether this provider supports reasoning/thinking"""
        pass

    def get_message_format_config(self) -> Dict[str, Any]:
        """
        Return provider-specific message format configuration.

        Returns:
            Dict with keys:
            - supports_tool_calls_in_assistant: bool (default True)
            - tool_message_format: str, one of:
                - 'full': role, content, name, tool_call_id (OpenAI, Anthropic)
                - 'name_content': role, content, name (HuggingFace)
                - 'user_content': convert to user message with formatted content
        """
        # Default configuration (OpenAI-compatible)
        return {
            'supports_tool_calls_in_assistant': True,
            'tool_message_format': 'full'
        }

    def get_specs(self) -> Dict[str, Any]:
        """
        Return tokenizer + model architecture specification.

        Expected keys:
        - vocab_size
        - context_length
        - hidden_size (embedding dim)
        - num_attention_heads
        - num_layers
        - max_output_tokens (if determinable)
        """
        ...
