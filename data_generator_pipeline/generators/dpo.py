# ============================================================================
# FILE: src/finetuning/generators/dpo.py
# ============================================================================

from typing import List, Dict, Any
from ..core.base import BaseGenerator


class DPOGenerator(BaseGenerator):
    """Generates preference pairs for DPO."""

    @classmethod
    def filter_kwargs(cls, kwargs: dict) -> dict:
        """DPO accepts no extra kwargs."""
        return {}

    async def generate_from_page(
        self,
        text: str,
        **llm_kwargs: Any,
    ) -> List[Dict]:
        prompt = self.prompt_template.format(text=text)

        messages = [
            {
                "role": "system",
                "content": (
                    "Generate a prompt and two responses (chosen and rejected) "
                    "based on the given text. Return JSON format: "
                    '[{"prompt": "...", "chosen": "...", "rejected": "..."}]'
                )
            },
            {"role": "user", "content": prompt},
        ]

        raw_text = await self._call_llm(messages, **llm_kwargs)

        try:
            return self.parser.extract(raw_text)
        except Exception as e:
            if self.debug:
                raise RuntimeError(f"DPO parsing failed: {e}") from e
            raise
