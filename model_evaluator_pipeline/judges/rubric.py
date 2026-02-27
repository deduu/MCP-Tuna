"""Rubric-based judge: evaluates against detailed scoring rubrics with level descriptions."""
from __future__ import annotations

import time
from typing import List, Optional

from .models import (
    CriterionScore,
    JudgeCriterion,
    JudgeRubric,
    JudgeType,
    SingleJudgeResult,
)
from .pointwise import PointwiseJudge


RUBRIC_PROMPT = """You are an expert evaluator using a detailed scoring rubric. For each criterion, match the output to the most appropriate score level described below.

Question: {question}

Generated Answer:
{generated}

{reference_section}

Scoring Rubric:
{criteria_section}

Return ONLY a JSON object with this exact format:
{json_format}

IMPORTANT: For each criterion, your score MUST correspond to one of the defined score levels."""


class RubricJudge(PointwiseJudge):
    """Like PointwiseJudge but uses a rubric-specific prompt emphasizing level matching."""

    judge_type = JudgeType.RUBRIC

    async def evaluate(
        self,
        question: str,
        generated: str,
        reference: Optional[str] = None,
        generated_b: Optional[str] = None,
        rubric: Optional[JudgeRubric] = None,
        criteria: Optional[List[JudgeCriterion]] = None,
    ) -> SingleJudgeResult:
        effective_criteria = (rubric.criteria if rubric else criteria) or []

        if not effective_criteria:
            # Fall back to parent pointwise behavior with default criteria
            result = await super().evaluate(
                question, generated, reference, generated_b, rubric, criteria,
            )
            # Override judge_type to RUBRIC even on fallback
            return result.model_copy(update={"judge_type": JudgeType.RUBRIC})

        template = (
            rubric.prompt_template
            if rubric and rubric.prompt_template
            else RUBRIC_PROMPT
        )
        reference_section = (
            f"Reference Answer:\n{reference}"
            if reference
            else "(No reference provided)"
        )

        prompt = template.format(
            question=question,
            generated=generated,
            reference_section=reference_section,
            criteria_section=self._build_criteria_prompt_section(effective_criteria),
            json_format=self._build_json_format_section(effective_criteria),
        )

        t0 = time.perf_counter()
        try:
            raw = await self._call_llm(prompt)
            latency = time.perf_counter() - t0
            parsed = self._parse_json_response(raw)

            if parsed is None:
                return self._error_result(effective_criteria, latency, "JSON parse failure", raw)

            scores = self._extract_scores(parsed, effective_criteria)
            overall = self._compute_overall(scores, effective_criteria)

            return SingleJudgeResult(
                judge_id=self.judge_id,
                judge_type=JudgeType.RUBRIC,
                model=self.llm.model_id,
                criteria_scores=scores,
                overall_score=overall,
                raw_response=raw,
                latency_s=round(latency, 4),
            )
        except Exception as e:
            latency = time.perf_counter() - t0
            return self._error_result(effective_criteria, latency, str(e))

    def _error_result(
        self,
        criteria: List[JudgeCriterion],
        latency: float,
        error: str,
        raw: str = "",
    ) -> SingleJudgeResult:
        return SingleJudgeResult(
            judge_id=self.judge_id,
            judge_type=JudgeType.RUBRIC,
            model=self.llm.model_id,
            criteria_scores=[
                CriterionScore(
                    criterion=c.name,
                    score=float(c.min_score),
                    reason=error,
                    min_score=c.min_score,
                    max_score=c.max_score,
                )
                for c in criteria
            ],
            overall_score=0.0,
            raw_response=raw,
            latency_s=round(latency, 4),
            success=False,
            error=error,
        )
