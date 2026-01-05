from .base import BaseMetric
from ...providers.base import LLMProvider

import json


class LLMQualityMetric(BaseMetric):
    name = "quality"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def compute(self, dp):
        prompt = f"""
Score the following response quality from 0 to 1.

Instruction:
{dp.full_instruction}

Response:
{dp.output}

Return ONLY valid JSON:
{{"score": float}}
"""
        try:
            raw = self.llm.generate(prompt)
            return float(json.loads(raw)["score"])
        except Exception:
            return 0.5
