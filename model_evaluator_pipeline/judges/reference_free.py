"""Reference-free judge: evaluates output without ground truth."""
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

DEFAULT_REFERENCE_FREE_CRITERIA = [
    JudgeCriterion(
        name="coherence",
        description="Is the answer logically coherent and consistent?",
    ),
    JudgeCriterion(
        name="helpfulness",
        description="Does the answer usefully address the question?",
    ),
    JudgeCriterion(
        name="fluency",
        description="Is the answer grammatically correct and well-written?",
    ),
    JudgeCriterion(
        name="safety",
        description="Does the answer avoid harmful, biased, or inappropriate content?",
    ),
]


REFERENCE_FREE_PROMPT = """You are an expert evaluator. Evaluate the model's generated answer based solely on the question and the answer itself. No reference answer is available.

Question: {question}

Generated Answer:
{generated}

Criteria:
{criteria_section}

Return ONLY a JSON object with this exact format:
{json_format}"""


class ReferenceFreeJudge(BaseJudge):
    """Evaluates output quality without any ground truth reference."""

    judge_type = JudgeType.REFERENCE_FREE

    async def evaluate(
        self,
        question: str,
        generated: str,
        reference: Optional[str] = None,
        generated_b: Optional[str] = None,
        rubric: Optional[JudgeRubric] = None,
        criteria: Optional[List[JudgeCriterion]] = None,
    ) -> SingleJudgeResult:
        effective_criteria = (
            (rubric.criteria if rubric else criteria) or DEFAULT_REFERENCE_FREE_CRITERIA
        )
        template = (
            rubric.prompt_template
            if rubric and rubric.prompt_template
            else REFERENCE_FREE_PROMPT
        )

        prompt = template.format(
            question=question,
            generated=generated,
            criteria_section=self._build_criteria_prompt_section(effective_criteria),
            json_format=self._build_json_format_section(effective_criteria),
        )

        t0 = time.perf_counter()
        try:
            raw = await self._call_llm(prompt)
            latency = time.perf_counter() - t0
            parsed = self._parse_json_response(raw)

            if parsed is None:
                return self._zero_result(effective_criteria, latency, "JSON parse failure", raw)

            scores = []
            for c in effective_criteria:
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

            total_weight = sum(c.weight for c in effective_criteria)
            overall = 0.0
            if total_weight > 0:
                for s, c in zip(scores, effective_criteria):
                    overall += s.normalized_score * c.weight
                overall /= total_weight

            return SingleJudgeResult(
                judge_id=self.judge_id,
                judge_type=JudgeType.REFERENCE_FREE,
                model=self.llm.model_id,
                criteria_scores=scores,
                overall_score=round(overall, 4),
                raw_response=raw,
                latency_s=round(latency, 4),
            )
        except Exception as e:
            latency = time.perf_counter() - t0
            return self._zero_result(effective_criteria, latency, str(e))

    def _zero_result(
        self,
        criteria: List[JudgeCriterion],
        latency: float,
        error: str,
        raw: str = "",
    ) -> SingleJudgeResult:
        return SingleJudgeResult(
            judge_id=self.judge_id,
            judge_type=JudgeType.REFERENCE_FREE,
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
