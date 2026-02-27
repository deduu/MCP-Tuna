"""Tests for RubricJudge."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from model_evaluator_pipeline.judges.models import (
    JudgeCriterion,
    JudgeRubric,
    JudgeType,
    ScoreLevel,
    SingleJudgeResult,
)
from model_evaluator_pipeline.judges.rubric import RubricJudge


# ── Helpers ────────────────────────────────────────────


def _make_mock_llm(response_content: str, model_id: str = "gpt-4o") -> MagicMock:
    llm = MagicMock()
    llm.model_id = model_id
    resp = MagicMock()
    resp.content = response_content
    llm.chat = AsyncMock(return_value=resp)
    return llm


MEDICAL_RUBRIC = JudgeRubric(
    name="medical_evaluation",
    description="Evaluate medical answers",
    criteria=[
        JudgeCriterion(
            name="accuracy",
            description="Medical accuracy",
            min_score=1,
            max_score=4,
            score_levels=[
                ScoreLevel(min_score=1, max_score=1, description="Incorrect facts"),
                ScoreLevel(min_score=2, max_score=2, description="Partially correct"),
                ScoreLevel(min_score=3, max_score=3, description="Mostly correct"),
                ScoreLevel(min_score=4, max_score=4, description="Fully accurate"),
            ],
        ),
        JudgeCriterion(
            name="safety",
            description="Safety warnings",
            min_score=1,
            max_score=4,
        ),
    ],
)


# ── Basic functionality ───────────────────────────────


class TestRubricJudge:
    @pytest.mark.asyncio
    async def test_returns_single_judge_result(self):
        response = json.dumps({
            "accuracy": {"score": 3, "reason": "Mostly correct"},
            "safety": {"score": 4, "reason": "Good warnings"},
        })
        llm = _make_mock_llm(response)
        judge = RubricJudge(llm=llm)

        result = await judge.evaluate(
            question="Side effects of aspirin?",
            generated="Aspirin can cause stomach irritation.",
            reference="Aspirin may cause GI bleeding, tinnitus.",
            rubric=MEDICAL_RUBRIC,
        )

        assert isinstance(result, SingleJudgeResult)
        assert result.judge_type == JudgeType.RUBRIC
        assert result.success is True
        assert len(result.criteria_scores) == 2

    @pytest.mark.asyncio
    async def test_prompt_contains_rubric_keyword(self):
        response = json.dumps({
            "accuracy": {"score": 3, "reason": "OK"},
            "safety": {"score": 4, "reason": "OK"},
        })
        llm = _make_mock_llm(response)
        judge = RubricJudge(llm=llm)

        await judge.evaluate(
            question="Q",
            generated="A",
            reference="R",
            rubric=MEDICAL_RUBRIC,
        )

        call_messages = llm.chat.call_args[0][0]
        prompt_text = call_messages[0]["content"].lower()
        assert "rubric" in prompt_text

    @pytest.mark.asyncio
    async def test_score_levels_in_prompt(self):
        response = json.dumps({
            "accuracy": {"score": 3, "reason": "OK"},
            "safety": {"score": 4, "reason": "OK"},
        })
        llm = _make_mock_llm(response)
        judge = RubricJudge(llm=llm)

        await judge.evaluate(
            question="Q", generated="A", rubric=MEDICAL_RUBRIC,
        )

        call_messages = llm.chat.call_args[0][0]
        prompt_text = call_messages[0]["content"]
        assert "Incorrect facts" in prompt_text
        assert "Fully accurate" in prompt_text

    @pytest.mark.asyncio
    async def test_falls_back_to_pointwise_without_rubric(self):
        """Without rubric or criteria, falls back to pointwise defaults."""
        response = json.dumps({
            "correctness": {"score": 8, "reason": "Good"},
            "completeness": {"score": 7, "reason": "OK"},
            "factuality": {"score": 9, "reason": "OK"},
            "structure": {"score": 9, "reason": "OK"},
            "hallucination_resistance": {"score": 10, "reason": "OK"},
        })
        llm = _make_mock_llm(response)
        judge = RubricJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        # Falls back to pointwise default criteria
        assert result.success is True
        assert len(result.criteria_scores) == 5

    @pytest.mark.asyncio
    async def test_custom_prompt_template(self):
        rubric = JudgeRubric(
            name="custom",
            criteria=[
                JudgeCriterion(name="c1", description="d1", min_score=1, max_score=5),
            ],
            prompt_template="Custom prompt for {question}: {generated}. Criteria: {criteria_section}. Format: {json_format}",
        )
        response = json.dumps({"c1": {"score": 3, "reason": "OK"}})
        llm = _make_mock_llm(response)
        judge = RubricJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", rubric=rubric,
        )

        assert result.success is True
        call_messages = llm.chat.call_args[0][0]
        assert "Custom prompt for Q" in call_messages[0]["content"]


# ── Error handling ────────────────────────────────────


class TestRubricJudgeErrors:
    @pytest.mark.asyncio
    async def test_json_parse_failure(self):
        llm = _make_mock_llm("Not JSON")
        judge = RubricJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", rubric=MEDICAL_RUBRIC,
        )

        assert result.success is False
