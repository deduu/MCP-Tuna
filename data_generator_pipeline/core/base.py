from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional
from typing import AsyncGenerator
from agent_framework.core.models import LLMResponse, StreamChunk


class BaseGenerator(ABC):
    """Abstract base class for generators."""

    def __init__(self,
                 llm: "BaseLLM",
                 prompt_template: str,
                 parser: "BaseParser",
                 *,
                 debug: bool = False):
        self.llm = llm
        self.prompt_template = prompt_template
        self.parser = parser
        self.debug = debug

    @abstractmethod
    async def generate_from_page(
        self,
        text: str,
        **llm_kwargs: Any,
    ):
        """Generates a list of dictionaries from a given text."""
        pass

    async def _call_llm(
            self,
            messages: List[Dict],
            tools: Optional[List[Dict[str, Any]]] = None,
            **llm_kwargs,
    ):
        """Calls the LLM with the given messages and tools."""
        resp = await self.llm.chat(
            messages=messages,
            tools=tools,
            **llm_kwargs,
        )

        # 2) Convert to string for parser
        raw_text = resp.content or ""

        # 3) Debug hooks (safe to keep always)
        if self.debug:
            self._debug_dump(
                messages=messages,
                response=resp,
                raw_text=raw_text,
            )
        return raw_text

    def _debug_dump(self, **kwargs):
        """Debug output."""
        import json
        print("\n" + "="*80)
        print("DEBUG OUTPUT")
        print("="*80)
        for key, value in kwargs.items():
            print(f"\n{key.upper()}:")
            if isinstance(value, (dict, list)):
                print(json.dumps(value, indent=2)[:500])
            else:
                print(str(value)[:500])
        print("="*80 + "\n")


class BaseParser(ABC):
    """Abstract base for parsers."""

    @abstractmethod
    def extract(self, content: str) -> List[Dict]:
        """Extract structured data from content."""
        pass


class BaseLLM(ABC):
    """
    A unified contract every provider must satisfy.
    Single method handles both regular chat and tool-enabled conversations.
    """
    supports_tool_calls_in_messages: bool = True

    @abstractmethod
    async def chat(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False,
        # stream: Optional[bool] = False,
        **kwargs: Any
    ) -> LLMResponse:
        """
        Generate a chat response, optionally with tool support.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of available tools in OpenAI function format
            enable_thinking: Whether to enable reasoning/thinking
            stream: Whether to return streaming response (if supported)
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
