# ============================================================================
# FILE: src/finetuning/generators/sft.py
# ============================================================================

from typing import List, Dict, Any
from ..core.base import BaseGenerator


class SFTGenerator(BaseGenerator):
    """Generates supervised fine-tuning pairs."""

    async def generate_from_page(
        self,
        text: str,
        **llm_kwargs: Any,
    ) -> List[Dict]:
        prompt = self.prompt_template.format(text=text)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates instruction-output pairs for fine-tuning."
            },
            {"role": "user", "content": prompt},
        ]

        raw_text = await self._call_llm(messages, **llm_kwargs)

        try:
            return self.parser.extract(raw_text)
        except Exception as e:
            if self.debug:
                raise RuntimeError(
                    f"JsonExtractor failed: {e}\n"
                    f"raw_text_preview={raw_text[:500]!r}"
                ) from e
            raise
