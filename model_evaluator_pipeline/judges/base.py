"""Abstract base class for all judge types."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from shared.providers import BaseLLM
from shared.registry import judge_registry

from .models import (
    JudgeCriterion,
    JudgeRubric,
    JudgeType,
    PairwiseResult,
    SingleJudgeResult,
)


class BaseJudge(ABC):
    """Abstract base class for all LLM-as-a-judge implementations.

    Subclasses must set ``judge_type`` class attribute and implement ``evaluate()``.
    Auto-registers concrete subclasses in ``judge_registry``.
    """

    judge_type: JudgeType

    def __init__(self, llm: BaseLLM, judge_id: Optional[str] = None):
        self.llm = llm
        self.judge_id = judge_id or llm.model_id

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "judge_type") and isinstance(cls.judge_type, JudgeType):
            judge_registry.add(cls.judge_type.value, cls)

    @abstractmethod
    async def evaluate(
        self,
        question: str,
        generated: str,
        reference: Optional[str] = None,
        generated_b: Optional[str] = None,
        rubric: Optional[JudgeRubric] = None,
        criteria: Optional[List[JudgeCriterion]] = None,
    ) -> Union[SingleJudgeResult, PairwiseResult]:
        """Run evaluation. Subclasses implement the specific paradigm."""
        ...

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the raw text response."""
        response = await self.llm.chat(
            [{"role": "user", "content": prompt}],
        )
        return response.content or ""

    def _parse_json_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response, handling markdown code fences."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

    def _build_criteria_prompt_section(
        self, criteria: List[JudgeCriterion],
    ) -> str:
        """Build the criteria description section for the prompt."""
        lines = []
        for c in criteria:
            lines.append(
                f"- {c.name} ({c.min_score}-{c.max_score}): {c.description}"
            )
            if c.score_levels:
                for level in c.score_levels:
                    lines.append(
                        f"    {level.min_score}-{level.max_score}: {level.description}"
                    )
        return "\n".join(lines)

    def _build_json_format_section(
        self, criteria: List[JudgeCriterion],
    ) -> str:
        """Build the expected JSON format section for the prompt."""
        entries = []
        for c in criteria:
            entries.append(
                f'  "{c.name}": {{"score": <{c.min_score}-{c.max_score}>, '
                f'"reason": "<brief explanation>"}}'
            )
        return "{\n" + ",\n".join(entries) + "\n}"

    def _clamp_score(self, value: Any, min_s: int, max_s: int) -> float:
        """Clamp a score to the criterion's range."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float(min_s)
        return max(float(min_s), min(float(max_s), v))
