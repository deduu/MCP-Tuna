from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from .base import BaseMetric
from ...providers.base import LLMProvider

log = logging.getLogger(__name__)

_LLM_TIMEOUT_S = 30


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
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self.llm.generate, prompt)
                raw = future.result(timeout=_LLM_TIMEOUT_S)
            return float(json.loads(raw)["score"])
        except FuturesTimeout:
            log.warning("LLM quality scoring timed out after %ds", _LLM_TIMEOUT_S)
            return 0.5
        except Exception:
            return 0.5
