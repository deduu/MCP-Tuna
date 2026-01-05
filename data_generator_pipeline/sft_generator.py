from typing import Any, Dict, List, Optional
from .llm_clients import LLMClient
from .prompt_template_registry import PromptTemplateRegistry
from .parsers_extractor import JsonExtractor
from src.agent_framework.providers.base import BaseLLM


# class SFTGenerator:
#     def __init__(
#         self,
#         llm: LLMClient,
#         prompt_registry: PromptTemplateRegistry,
#         parser: JsonExtractor,
#     ):
#         self.llm = llm
#         self.prompt_registry = prompt_registry
#         self.parser = parser

#     def generate_from_page(self, text: str) -> List[Dict]:
#         prompt = self.prompt_registry.render(text)
#         raw = self.llm.generate(prompt)
#         return self.parser.extract(raw)


class SFTGenerator:
    def __init__(
        self,
        llm: "BaseLLM",
        prompt_registry: "PromptTemplateRegistry",
        parser: "JsonExtractor",
        *,
        debug: bool = False,
    ):
        self.llm = llm
        self.prompt_registry = prompt_registry
        self.parser = parser
        self.debug = debug

    async def generate_from_page(
        self,
        text: str,
        *,
        tools: Optional[List[Dict[str, Any]]] = None,
        **llm_kwargs: Any,
    ) -> List[Dict]:
        prompt = self.prompt_registry.render(text)

        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ]

        # 1) LLM call (async)
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
                prompt=prompt,
                messages=messages,
                response=resp,
                raw_text=raw_text,
            )

        # 4) Parse
        try:
            return self.parser.extract(raw_text)
        except Exception as e:
            if self.debug:
                # Keep the most useful artifacts attached to the error
                raise RuntimeError(
                    f"JsonExtractor failed: {e}\n"
                    f"raw_text_preview={raw_text[:500]!r}"
                ) from e
            raise

    def _debug_dump(self, *, prompt, messages, response, raw_text):
        print("\n=== SFTGenerator DEBUG ===")
        print(f"prompt_preview: {prompt[:400]!r}")
        print(f"messages: {messages}")
        print(
            f"llm_response.finish_reason: {getattr(response,'finish_reason',None)}")
        print(f"llm_response.usage: {getattr(response,'usage',None)}")
        print(
            f"llm_response.tool_calls: {getattr(response,'tool_calls',None)}")
        print(f"raw_text_preview: {raw_text[:800]!r}")
        print("=== END DEBUG ===\n")
