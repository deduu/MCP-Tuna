"""Standalone JSON extractor for parsing LLM responses.

Extracts JSON arrays/objects from LLM output, handling thinking tags
and fenced code blocks. Used by orchestration and data_generator pipelines.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List


class JsonExtractor:
    """Extract JSON from LLM responses."""

    def extract(self, content: str) -> List[Dict]:
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
