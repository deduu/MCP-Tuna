import os
from typing import Callable, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from agentsoul.providers.base import BaseLLM
from agentsoul.providers.openai import OpenAIProvider
from agentsoul.providers.anthropic import AnthropicProvider
from agentsoul.providers.google import GoogleGeminiProvider

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
_ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
_ANTHROPIC_API_BASE = os.getenv("ANTHROPIC_API_BASE")
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

_MODEL_FACTORIES: Dict[str, Callable[..., BaseLLM]] = {
    "gpt-4o": lambda api_key=_OPENAI_API_KEY, base_url=_OPENAI_API_BASE: OpenAIProvider(
        model_id="gpt-4o", api_key=api_key, base_url=base_url
    ),
    "claude-sonnet-4-20250514": lambda api_key=_ANTHROPIC_API_KEY, base_url=_ANTHROPIC_API_BASE: AnthropicProvider(
        model_id="claude-sonnet-4-20250514", api_key=api_key, base_url=base_url
    ),
    "gemini-2.0-flash": lambda api_key=_GOOGLE_API_KEY, base_url=None: GoogleGeminiProvider(
        model_id="gemini-2.0-flash", api_key=api_key,
    ),
}

_registry = None


class LLMRegistry:

    def __init__(self):
        self.model_factories = _MODEL_FACTORIES

    async def preload(self, items: list[tuple[str, Optional[str]]]) -> None:
        for name, key in items:
            self.model_factories[name](key)

    async def get(self, name: str, api_key: Optional[str] = None, base_url: Optional[str] = None) -> BaseLLM:
        if name not in self.model_factories:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(self.model_factories)}")

        # Only pass non-None values so lambda defaults (from env vars) are preserved
        kwargs = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        return self.model_factories[name](**kwargs)


def get_registry() -> LLMRegistry:
    global _registry
    if _registry is None:
        _registry = LLMRegistry()
    return _registry
