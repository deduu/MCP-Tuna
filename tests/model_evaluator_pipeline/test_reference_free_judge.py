"""Tests for ReferenceFreeJudge."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from model_evaluator_pipeline.judges.models import (
    JudgeCriterion,
    JudgeType,
    SingleJudgeResult,
)
from model_evaluator_pipeline.judges.reference_free import (
    DEFAULT_REFERENCE_FREE_CRITERIA,
    ReferenceFreeJudge,
)


# ── Helpers ────────────────────────────────────────────


def _make_mock_llm(response_content: str, model_id: str = "gpt-4o") -> MagicMock:
    llm = MagicMock()
    llm.model_id = model_id
    resp = MagicMock()
    resp.content = response_content
    llm.chat = AsyncMock(return_value=resp)
    return llm


SAMPLE_REFERENCE_FREE_RESPONSE = json.dumps({
    "coherence": {"score": 8, "reason": "Logical flow"},
    "helpfulness": {"score": 7, "reason": "Addresses question"},
    "fluency": {"score": 9, "reason": "Well written"},
    "safety": {"score": 10, "reason": "No harmful content"},
})


# ── Default criteria ──────────────────────────────────


class TestDefaultCriteria:
    def test_default_criteria_names(self):
        names = [c.name for c in DEFAULT_REFERENCE_FREE_CRITERIA]
        assert names == ["coherence", "helpfulness", "fluency", "safety"]


# ── Basic functionality ───────────────────────────────


class TestReferenceFreeJudge:
    @pytest.mark.asyncio
    async def test_returns_single_judge_result(self):
        llm = _make_mock_llm(SAMPLE_REFERENCE_FREE_RESPONSE)
        judge = ReferenceFreeJudge(llm=llm)

        result = await judge.evaluate(
            question="What is Python?",
            generated="Python is a programming language.",
        )

        assert isinstance(result, SingleJudgeResult)
        assert result.judge_type == JudgeType.REFERENCE_FREE
        assert result.success is True
        assert len(result.criteria_scores) == 4

    @pytest.mark.asyncio
    async def test_reference_ignored(self):
        llm = _make_mock_llm(SAMPLE_REFERENCE_FREE_RESPONSE)
        judge = ReferenceFreeJudge(llm=llm)

        result = await judge.evaluate(
            question="Q",
            generated="A",
            reference="This should be ignored",
        )

        assert result.success is True
        # Reference should not appear in the prompt
        call_messages = llm.chat.call_args[0][0]
        assert "This should be ignored" not in call_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_overall_score_computed(self):
        llm = _make_mock_llm(SAMPLE_REFERENCE_FREE_RESPONSE)
        judge = ReferenceFreeJudge(llm=llm)

        result = await judge.evaluate(question="Q", generated="A")

        assert result.overall_score is not None
        assert 0.0 <= result.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_custom_criteria(self):
        response = json.dumps({
            "tone": {"score": 4, "reason": "Professional"},
        })
        llm = _make_mock_llm(response)
        judge = ReferenceFreeJudge(llm=llm)

        result = await judge.evaluate(
            question="Q",
            generated="A",
            criteria=[
                JudgeCriterion(
                    name="tone",
                    description="Is the tone professional?",
                    min_score=1,
                    max_score=5,
                ),
            ],
        )

        assert len(result.criteria_scores) == 1
        assert result.criteria_scores[0].criterion == "tone"
        assert result.criteria_scores[0].score == 4


# ── Error handling ────────────────────────────────────


class TestReferenceFreeJudgeErrors:
    @pytest.mark.asyncio
    async def test_json_parse_failure(self):
        llm = _make_mock_llm("Not JSON")
        judge = ReferenceFreeJudge(llm=llm)

        result = await judge.evaluate(question="Q", generated="A")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_llm_exception(self):
        llm = MagicMock()
        llm.model_id = "gpt-4o"
        llm.chat = AsyncMock(side_effect=RuntimeError("Timeout"))
        judge = ReferenceFreeJudge(llm=llm)

        result = await judge.evaluate(question="Q", generated="A")

        assert result.success is False
