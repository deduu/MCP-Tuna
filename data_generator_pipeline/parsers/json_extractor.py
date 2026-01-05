# ============================================================================
# FILE: src/finetuning/parsers/json_extractor.py
# ============================================================================

import json
import re
from ..core.base import BaseParser


class JsonExtractor(BaseParser):
    """Extract JSON from LLM responses."""

    def extract(self, content: str) -> list:
        # Remove thinking tags if present
        if "</think>" in content:
            content = content.split("</think>")[-1]

        # Extract fenced JSON if exists
        match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)

        content = content.strip()

        try:
            parsed = json.loads(content)
            # Ensure it's always a list
            if isinstance(parsed, dict):
                return [parsed]
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON returned:\n{content[:500]}"
            ) from e
