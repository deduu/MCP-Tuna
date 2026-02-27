"""Pairwise judge: compares two outputs and picks the better one."""
from __future__ import annotations

import time
from typing import List, Optional

from .base import BaseJudge
from .models import (
    JudgeCriterion,
    JudgeRubric,
    JudgeType,
    PairwiseResult,
)


PAIRWISE_PROMPT = """You are an expert evaluator. Compare the two model outputs below and determine which one better answers the question.

Question: {question}

Output A:
{generated_a}

Output B:
{generated_b}

{reference_section}

Evaluation criteria:
{criteria_section}

Return ONLY a JSON object:
{{
  "winner": "<A or B or tie>",
  "confidence": <0.0-1.0>,
  "reason": "<explanation of your choice>"
}}"""


class PairwiseJudge(BaseJudge):
    """Compares two outputs head-to-head."""

    judge_type = JudgeType.PAIRWISE

    async def evaluate(
        self,
        question: str,
        generated: str,
        reference: Optional[str] = None,
        generated_b: Optional[str] = None,
        rubric: Optional[JudgeRubric] = None,
        criteria: Optional[List[JudgeCriterion]] = None,
    ) -> PairwiseResult:
        if generated_b is None:
            return PairwiseResult(
                judge_id=self.judge_id,
                model=self.llm.model_id,
                winner="A",
                confidence=1.0,
                reason="No second output provided for comparison.",
                success=False,
                error="generated_b is required for pairwise evaluation",
            )

        criteria_list = (rubric.criteria if rubric else criteria) or []
        criteria_section = (
            self._build_criteria_prompt_section(criteria_list)
            if criteria_list
            else "Use your best judgment on overall quality."
        )
        reference_section = (
            f"Reference Answer:\n{reference}"
            if reference
            else "(No reference answer provided)"
        )

        prompt = PAIRWISE_PROMPT.format(
            question=question,
            generated_a=generated,
            generated_b=generated_b,
            reference_section=reference_section,
            criteria_section=criteria_section,
        )

        t0 = time.perf_counter()
        try:
            raw = await self._call_llm(prompt)
            latency = time.perf_counter() - t0
            parsed = self._parse_json_response(raw)

            if parsed is None:
                return PairwiseResult(
                    judge_id=self.judge_id,
                    model=self.llm.model_id,
                    winner="tie",
                    confidence=0.0,
                    reason="JSON parse failure",
                    latency_s=round(latency, 4),
                    success=False,
                    error="JSON parse failure",
                )

            winner = str(parsed.get("winner", "tie")).strip().upper()
            if winner == "TIE":
                winner = "tie"
            elif winner not in ("A", "B"):
                winner = "tie"

            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))

            return PairwiseResult(
                judge_id=self.judge_id,
                model=self.llm.model_id,
                winner=winner,
                confidence=confidence,
                reason=str(parsed.get("reason", "")),
                latency_s=round(latency, 4),
            )
        except Exception as e:
            latency = time.perf_counter() - t0
            return PairwiseResult(
                judge_id=self.judge_id,
                model=self.llm.model_id,
                winner="tie",
                confidence=0.0,
                reason=str(e),
                latency_s=round(latency, 4),
                success=False,
                error=str(e),
            )
