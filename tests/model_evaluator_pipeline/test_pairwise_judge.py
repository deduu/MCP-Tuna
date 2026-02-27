"""Tests for PairwiseJudge."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from model_evaluator_pipeline.judges.models import (
    JudgeCriterion,
    JudgeType,
    PairwiseResult,
)
from model_evaluator_pipeline.judges.pairwise import PairwiseJudge


# ── Helpers ────────────────────────────────────────────


def _make_mock_llm(response_content: str, model_id: str = "gpt-4o") -> MagicMock:
    llm = MagicMock()
    llm.model_id = model_id
    resp = MagicMock()
    resp.content = response_content
    llm.chat = AsyncMock(return_value=resp)
    return llm


SAMPLE_PAIRWISE_RESPONSE = json.dumps({
    "winner": "A",
    "confidence": 0.85,
    "reason": "Output A is more detailed and accurate",
})


# ── Basic functionality ───────────────────────────────


class TestPairwiseJudge:
    @pytest.mark.asyncio
    async def test_returns_pairwise_result(self):
        llm = _make_mock_llm(SAMPLE_PAIRWISE_RESPONSE)
        judge = PairwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="What is Python?",
            generated="Answer A",
            generated_b="Answer B",
        )

        assert isinstance(result, PairwiseResult)
        assert result.winner == "A"
        assert result.confidence == 0.85
        assert result.success is True

    @pytest.mark.asyncio
    async def test_winner_b(self):
        response = json.dumps({"winner": "B", "confidence": 0.9, "reason": "Better"})
        llm = _make_mock_llm(response)
        judge = PairwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", generated_b="B",
        )

        assert result.winner == "B"

    @pytest.mark.asyncio
    async def test_tie_result(self):
        response = json.dumps({"winner": "tie", "confidence": 0.5, "reason": "Equal"})
        llm = _make_mock_llm(response)
        judge = PairwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", generated_b="B",
        )

        assert result.winner == "tie"

    @pytest.mark.asyncio
    async def test_missing_generated_b(self):
        llm = _make_mock_llm(SAMPLE_PAIRWISE_RESPONSE)
        judge = PairwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A",
        )

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_with_reference(self):
        llm = _make_mock_llm(SAMPLE_PAIRWISE_RESPONSE)
        judge = PairwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", generated_b="B", reference="Ref",
        )

        assert result.success is True
        call_messages = llm.chat.call_args[0][0]
        assert "Ref" in call_messages[0]["content"]

    @pytest.mark.asyncio
    async def test_with_criteria(self):
        llm = _make_mock_llm(SAMPLE_PAIRWISE_RESPONSE)
        judge = PairwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q",
            generated="A",
            generated_b="B",
            criteria=[
                JudgeCriterion(name="accuracy", description="Is it accurate?"),
            ],
        )

        assert result.success is True
        call_messages = llm.chat.call_args[0][0]
        assert "accuracy" in call_messages[0]["content"]


# ── Error handling ────────────────────────────────────


class TestPairwiseJudgeErrors:
    @pytest.mark.asyncio
    async def test_json_parse_failure(self):
        llm = _make_mock_llm("Not valid JSON")
        judge = PairwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", generated_b="B",
        )

        assert result.success is False
        assert result.winner == "tie"

    @pytest.mark.asyncio
    async def test_llm_exception(self):
        llm = MagicMock()
        llm.model_id = "gpt-4o"
        llm.chat = AsyncMock(side_effect=RuntimeError("Connection failed"))
        judge = PairwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", generated_b="B",
        )

        assert result.success is False
        assert result.winner == "tie"

    @pytest.mark.asyncio
    async def test_invalid_winner_normalized(self):
        response = json.dumps({"winner": "X", "confidence": 0.5, "reason": "?"})
        llm = _make_mock_llm(response)
        judge = PairwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", generated_b="B",
        )

        assert result.winner == "tie"

    @pytest.mark.asyncio
    async def test_confidence_clamped(self):
        response = json.dumps({"winner": "A", "confidence": 5.0, "reason": "?"})
        llm = _make_mock_llm(response)
        judge = PairwiseJudge(llm=llm)

        result = await judge.evaluate(
            question="Q", generated="A", generated_b="B",
        )

        assert result.confidence == 1.0
