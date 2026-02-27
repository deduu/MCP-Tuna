"""Pointwise judge: scores each output individually against criteria."""
from __future__ import annotations

import time
from typing import List, Optional

from .base import BaseJudge
from .models import (
    CriterionScore,
    JudgeCriterion,
    JudgeRubric,
    JudgeType,
    SingleJudgeResult,
)

DEFAULT_POINTWISE_CRITERIA = [
    JudgeCriterion(
        name="correctness",
        description="How accurate is the generated answer compared to the reference?",
    ),
    JudgeCriterion(
        name="completeness",
        description="Does the generated answer cover all key points from the reference?",
    ),
    JudgeCriterion(
        name="factuality",
        description="Are all stated facts verifiable and correct?",
    ),
    JudgeCriterion(
        name="structure",
        description="Is the answer well-organized and clearly written?",
    ),
    JudgeCriterion(
        name="hallucination_resistance",
        description="Does the answer avoid fabricating information not in the reference?",
    ),
]


POINTWISE_PROMPT = """You are an expert evaluator. Score the model's generated answer against the reference answer on the following criteria.

Question: {question}

Generated Answer:
{generated}

Reference Answer:
{reference}

Criteria:
{criteria_section}

Return ONLY a JSON object with this exact format:
{json_format}"""


class PointwiseJudge(BaseJudge):
    """Scores each output individually on user-defined criteria."""

    judge_type = JudgeType.POINTWISE

    async def evaluate(
        self,
        question: str,
        generated: str,
        reference: Optional[str] = None,
        generated_b: Optional[str] = None,
        rubric: Optional[JudgeRubric] = None,
        criteria: Optional[List[JudgeCriterion]] = None,
    ) -> SingleJudgeResult:
        criteria = self._resolve_criteria(rubric, criteria)
        template = (
            rubric.prompt_template
            if rubric and rubric.prompt_template
            else POINTWISE_PROMPT
        )

        prompt = template.format(
            question=question,
            generated=generated,
            reference=reference or "(no reference provided)",
            criteria_section=self._build_criteria_prompt_section(criteria),
            json_format=self._build_json_format_section(criteria),
        )

        t0 = time.perf_counter()
        try:
            raw = await self._call_llm(prompt)
            latency = time.perf_counter() - t0
            parsed = self._parse_json_response(raw)

            if parsed is None:
                return self._error_result(
                    criteria, latency, "JSON parse failure", raw,
                )

            scores = self._extract_scores(parsed, criteria)
            overall = self._compute_overall(scores, criteria)

            return SingleJudgeResult(
                judge_id=self.judge_id,
                judge_type=JudgeType.POINTWISE,
                model=self.llm.model_id,
                criteria_scores=scores,
                overall_score=overall,
                raw_response=raw,
                latency_s=round(latency, 4),
            )
        except Exception as e:
            latency = time.perf_counter() - t0
            return self._error_result(criteria, latency, str(e))

    def _resolve_criteria(
        self,
        rubric: Optional[JudgeRubric],
        criteria: Optional[List[JudgeCriterion]],
    ) -> List[JudgeCriterion]:
        if rubric:
            return rubric.criteria
        if criteria:
            return criteria
        return DEFAULT_POINTWISE_CRITERIA

    def _extract_scores(
        self,
        parsed: dict,
        criteria: List[JudgeCriterion],
    ) -> List[CriterionScore]:
        scores = []
        for c in criteria:
            entry = parsed.get(c.name, {})
            if isinstance(entry, dict):
                scores.append(
                    CriterionScore(
                        criterion=c.name,
                        score=self._clamp_score(
                            entry.get("score", c.min_score),
                            c.min_score,
                            c.max_score,
                        ),
                        reason=str(entry.get("reason", "")),
                        min_score=c.min_score,
                        max_score=c.max_score,
                    )
                )
            else:
                scores.append(
                    CriterionScore(
                        criterion=c.name,
                        score=float(c.min_score),
                        reason="Missing in judge response",
                        min_score=c.min_score,
                        max_score=c.max_score,
                    )
                )
        return scores

    def _compute_overall(
        self,
        scores: List[CriterionScore],
        criteria: List[JudgeCriterion],
    ) -> float:
        total_weight = sum(c.weight for c in criteria)
        if total_weight == 0:
            return 0.0
        weighted_sum = 0.0
        for score, crit in zip(scores, criteria):
            weighted_sum += score.normalized_score * crit.weight
        return round(weighted_sum / total_weight, 4)

    def _error_result(
        self,
        criteria: List[JudgeCriterion],
        latency: float,
        error: str,
        raw: str = "",
    ) -> SingleJudgeResult:
        return SingleJudgeResult(
            judge_id=self.judge_id,
            judge_type=JudgeType.POINTWISE,
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
