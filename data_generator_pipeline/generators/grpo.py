# ============================================================================
# FILE: src/finetuning/generators/grpo.py
# ============================================================================

from typing import List, Dict, Any
from ..core.base import BaseGenerator


class GRPOGenerator(BaseGenerator):
    """Generates multiple responses with rewards for GRPO."""

    def __init__(self, *args, num_responses: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_responses = num_responses

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
                    f"Generate a prompt and {self.num_responses} different responses. "
                    "Return JSON: [{\"prompt\": \"...\", \"responses\": [...]}]"
                )
            },
            {"role": "user", "content": prompt},
        ]

        raw_text = await self._call_llm(messages, **llm_kwargs)

        try:
            parsed = self.parser.extract(raw_text)
            # Add placeholder rewards (calculate separately with reward model)
            for item in parsed:
                if "rewards" not in item:
                    item["rewards"] = [0.0] * len(item.get("responses", []))
            return parsed
        except Exception as e:
            if self.debug:
                raise RuntimeError(f"GRPO parsing failed: {e}") from e
            raise
