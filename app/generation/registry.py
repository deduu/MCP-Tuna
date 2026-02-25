import os
from typing import Callable, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

from agentsoul.providers.base import BaseLLM
from agentsoul.providers.openai import OpenAIProvider

_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

_MODEL_FACTORIES: Dict[str, Callable[..., BaseLLM]] = {
    "gpt-4o": lambda api_key=_OPENAI_API_KEY, base_url=_OPENAI_API_BASE: OpenAIProvider(
        model_id="gpt-4o", api_key=api_key, base_url=base_url
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

        return self.model_factories[name](api_key, base_url)


def get_registry() -> LLMRegistry:
    global _registry
    if _registry is None:
        _registry = LLMRegistry()
    return _registry
