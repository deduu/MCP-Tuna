from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from AgentY.shared.providers import BaseLLM  # noqa: F401 — re-exported for backwards compat


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

    @classmethod
    def filter_kwargs(cls, kwargs: dict) -> dict:
        """Override in subclasses to whitelist accepted kwargs. Default: none."""
        return {}

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

        raw_text = resp.content or ""

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
