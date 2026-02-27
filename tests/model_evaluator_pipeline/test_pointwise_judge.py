"""Tests for BaseJudge and PointwiseJudge."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from model_evaluator_pipeline.judges.models import (
    CriterionScore,
    JudgeCriterion,
    JudgeRubric,
    JudgeType,
    SingleJudgeResult,
)
from model_evaluator_pipeline.judges.pointwise import (
    DEFAULT_POINTWISE_CRITERIA,
    PointwiseJudge,
)


# ── Helpers ────────────────────────────────────────────


def _make_mock_llm(response_content: str, model_id: str = "gpt-4o") -> MagicMock:
    """Create a mock BaseLLM that returns the given content."""
    llm = MagicMock()
    llm.model_id = model_id
    resp = MagicMock()
    resp.content = response_content
    llm.chat = AsyncMock(return_value=resp)
    return llm


SAMPLE_POINTWISE_RESPONSE = json.dumps({
    "correctness": {"score": 8, "reason": "Accurate"},
    "completeness": {"score": 7, "reason": "Mostly complete"},
    "factuality": {"score": 9, "reason": "Correct facts"},
    "structure": {"score": 9, "reason": "Well organized"},
    "hallucination_resistance": {"score": 10, "reason": "No hallucinations"},
})


# ── Default criteria ──────────────────────────────────


class TestDefaultCriteria:
    def test_default_criteria_match_original_five(self):
        names = [c.name for c in DEFAULT_POINTWISE_CRITERIA]
        assert names == [
            "correctness",
            "completeness",
            "factuality",
            "structure",
            "hallucination_resistance",
        ]

    def test_default_criteria_ranges(self):
        for c in DEFAULT_POINTWISE_CRITERIA:
            assert c.min_score == 1
            assert c.max_score == 10
            assert c.weight == 1.0


# ── PointwiseJudge with default criteria ──────────────


class TestPointwiseJudgeDefault:
    @pytest.mark.asyncio
    async def test_evaluate_returns_single_judge_result(self):
        llm = _make_mock_llm(SAMPLE_POINTWISE_RESPONSE)
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="What is Python?",
            generated="Python is a programming language.",
            reference="Python is a high-level programming language.",
        )

        assert isinstance(result, SingleJudgeResult)
        assert result.judge_type == JudgeType.POINTWISE
        assert result.success is True
        assert result.model == "gpt-4o"
        assert len(result.criteria_scores) == 5

    @pytest.mark.asyncio
    async def test_scores_parsed_correctly(self):
        llm = _make_mock_llm(SAMPLE_POINTWISE_RESPONSE)
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        scores_by_name = {s.criterion: s for s in result.criteria_scores}
        assert scores_by_name["correctness"].score == 8
        assert scores_by_name["completeness"].score == 7
        assert scores_by_name["factuality"].score == 9
        assert scores_by_name["hallucination_resistance"].score == 10

    @pytest.mark.asyncio
    async def test_overall_score_computed(self):
        llm = _make_mock_llm(SAMPLE_POINTWISE_RESPONSE)
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        assert result.overall_score is not None
        assert 0.0 <= result.overall_score <= 1.0

    @pytest.mark.asyncio
    async def test_latency_recorded(self):
        llm = _make_mock_llm(SAMPLE_POINTWISE_RESPONSE)
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        assert result.latency_s >= 0.0

    @pytest.mark.asyncio
    async def test_llm_called_with_prompt(self):
        llm = _make_mock_llm(SAMPLE_POINTWISE_RESPONSE)
        judge = PointwiseJudge(llm=llm)

        await judge.evaluate(question="Q", generated="A", reference="R")

        llm.chat.assert_called_once()
        call_messages = llm.chat.call_args[0][0]
        assert len(call_messages) == 1
        assert "Q" in call_messages[0]["content"]


# ── PointwiseJudge with custom criteria ───────────────


class TestPointwiseJudgeCustomCriteria:
    @pytest.mark.asyncio
    async def test_custom_criteria_used(self):
        custom_response = json.dumps({
            "medical_accuracy": {"score": 4, "reason": "Good"},
            "safety": {"score": 5, "reason": "Includes warnings"},
        })
        llm = _make_mock_llm(custom_response)
        judge = PointwiseJudge(llm=llm)

        custom_criteria = [
            JudgeCriterion(
                name="medical_accuracy",
                description="Are medical facts correct?",
                min_score=1,
                max_score=5,
            ),
            JudgeCriterion(
                name="safety",
                description="Safety warnings included?",
                min_score=1,
                max_score=5,
            ),
        ]

        result = await judge.evaluate(
            question="Side effects?",
            generated="Answer",
            reference="Ref",
            criteria=custom_criteria,
        )

        assert len(result.criteria_scores) == 2
        scores_by_name = {s.criterion: s for s in result.criteria_scores}
        assert scores_by_name["medical_accuracy"].score == 4
        assert scores_by_name["medical_accuracy"].max_score == 5
        assert scores_by_name["safety"].score == 5

    @pytest.mark.asyncio
    async def test_rubric_overrides_criteria(self):
        custom_response = json.dumps({
            "clarity": {"score": 3, "reason": "OK"},
        })
        llm = _make_mock_llm(custom_response)
        judge = PointwiseJudge(llm=llm)

        rubric = JudgeRubric(
            name="simple",
            criteria=[
                JudgeCriterion(
                    name="clarity",
                    description="Is it clear?",
                    min_score=1,
                    max_score=5,
                ),
            ],
        )

        result = await judge.evaluate(
            question="Q", generated="A", reference="R", rubric=rubric,
        )

        assert len(result.criteria_scores) == 1
        assert result.criteria_scores[0].criterion == "clarity"


# ── Error handling ────────────────────────────────────


class TestPointwiseJudgeErrors:
    @pytest.mark.asyncio
    async def test_json_parse_failure(self):
        llm = _make_mock_llm("This is not JSON at all")
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        assert result.success is False
        assert "parse" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_criteria_in_response(self):
        partial_response = json.dumps({
            "correctness": {"score": 8, "reason": "Good"},
            # missing other criteria
        })
        llm = _make_mock_llm(partial_response)
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        assert result.success is True  # still succeeds, missing get min_score
        scores_by_name = {s.criterion: s for s in result.criteria_scores}
        assert scores_by_name["correctness"].score == 8
        assert scores_by_name["completeness"].score == 1  # min_score fallback

    @pytest.mark.asyncio
    async def test_llm_exception_handled(self):
        llm = MagicMock()
        llm.model_id = "gpt-4o"
        llm.chat = AsyncMock(side_effect=RuntimeError("API timeout"))
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        assert result.success is False
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_score_clamping(self):
        response = json.dumps({
            "correctness": {"score": 99, "reason": "Over max"},
            "completeness": {"score": -5, "reason": "Under min"},
            "factuality": {"score": 5, "reason": "Normal"},
            "structure": {"score": "not_a_number", "reason": "Bad type"},
            "hallucination_resistance": {"score": 8, "reason": "OK"},
        })
        llm = _make_mock_llm(response)
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        scores_by_name = {s.criterion: s for s in result.criteria_scores}
        assert scores_by_name["correctness"].score == 10  # clamped to max
        assert scores_by_name["completeness"].score == 1  # clamped to min
        assert scores_by_name["factuality"].score == 5
        assert scores_by_name["structure"].score == 1  # fallback for bad type
        assert scores_by_name["hallucination_resistance"].score == 8

    @pytest.mark.asyncio
    async def test_markdown_code_fence_parsed(self):
        fenced = f"```json\n{SAMPLE_POINTWISE_RESPONSE}\n```"
        llm = _make_mock_llm(fenced)
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        assert result.success is True
        assert len(result.criteria_scores) == 5


# ── Custom judge_id ───────────────────────────────────


class TestPointwiseJudgeId:
    @pytest.mark.asyncio
    async def test_custom_judge_id(self):
        llm = _make_mock_llm(SAMPLE_POINTWISE_RESPONSE)
        judge = PointwiseJudge(llm=llm, judge_id="primary-judge")

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        assert result.judge_id == "primary-judge"

    @pytest.mark.asyncio
    async def test_default_judge_id_from_model(self):
        llm = _make_mock_llm(SAMPLE_POINTWISE_RESPONSE, model_id="claude-3")
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference="R",
        )

        assert result.judge_id == "claude-3"


# ── No reference provided ────────────────────────────


class TestPointwiseNoReference:
    @pytest.mark.asyncio
    async def test_no_reference_handled(self):
        llm = _make_mock_llm(SAMPLE_POINTWISE_RESPONSE)
        judge = PointwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", reference=None,
        )

        assert result.success is True
        # Check prompt includes "(no reference provided)"
        call_messages = llm.chat.call_args[0][0]
        assert "no reference provided" in call_messages[0]["content"]
