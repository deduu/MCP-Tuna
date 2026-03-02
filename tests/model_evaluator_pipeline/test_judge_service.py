"""Tests for AdvancedJudgeService."""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_evaluator_pipeline.services.judge_service import AdvancedJudgeService
from shared.config import AdvancedJudgeConfig


# ── Helpers ────────────────────────────────────────────


def _make_mock_llm(response_content: str, model_id: str = "gpt-4o") -> MagicMock:
    llm = MagicMock()
    llm.model_id = model_id
    resp = MagicMock()
    resp.content = response_content
    llm.chat = AsyncMock(return_value=resp)
    return llm


POINTWISE_RESPONSE = json.dumps({
    "correctness": {"score": 8, "reason": "Good"},
    "completeness": {"score": 7, "reason": "OK"},
    "factuality": {"score": 9, "reason": "Correct"},
    "structure": {"score": 9, "reason": "Clear"},
    "hallucination_resistance": {"score": 10, "reason": "None"},
})

PAIRWISE_RESPONSE = json.dumps({
    "winner": "A",
    "confidence": 0.85,
    "reason": "Better quality",
})


# ── Config ─────────────────────────────────────────────


class TestAdvancedJudgeConfig:
    def test_defaults(self):
        cfg = AdvancedJudgeConfig()
        assert cfg.default_judge_type == "pointwise"
        assert cfg.default_judge_model == "gpt-4o"
        assert cfg.default_aggregation == "mean"
        assert cfg.timeout_s == 30.0


# ── Single evaluation ─────────────────────────────────


class TestEvaluateSingle:
    @pytest.mark.asyncio
    async def test_basic_pointwise(self):
        svc = AdvancedJudgeService()
        mock_llm = _make_mock_llm(POINTWISE_RESPONSE)

        with patch.object(svc, "_get_llm", return_value=mock_llm):
            result = await svc.evaluate_single(
                question="What is Python?",
                generated="Python is a language.",
                reference="Python is a high-level programming language.",
            )

        assert result["success"] is True
        assert "result" in result

    @pytest.mark.asyncio
    async def test_pairwise_single(self):
        svc = AdvancedJudgeService()
        mock_llm = _make_mock_llm(PAIRWISE_RESPONSE)

        with patch.object(svc, "_get_llm", return_value=mock_llm):
            result = await svc.evaluate_single(
                question="Q",
                generated="A",
                generated_b="B",
                judge_type="pairwise",
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_custom_criteria(self):
        custom_response = json.dumps({
            "accuracy": {"score": 4, "reason": "OK"},
        })
        svc = AdvancedJudgeService()
        mock_llm = _make_mock_llm(custom_response)

        with patch.object(svc, "_get_llm", return_value=mock_llm):
            result = await svc.evaluate_single(
                question="Q",
                generated="A",
                reference="R",
                criteria=[
                    {"name": "accuracy", "description": "Is it accurate?", "min_score": 1, "max_score": 5},
                ],
            )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_with_rubric(self):
        rubric_response = json.dumps({
            "quality": {"score": 3, "reason": "Decent"},
        })
        svc = AdvancedJudgeService()
        mock_llm = _make_mock_llm(rubric_response)

        with patch.object(svc, "_get_llm", return_value=mock_llm):
            result = await svc.evaluate_single(
                question="Q",
                generated="A",
                judge_type="rubric",
                rubric={
                    "name": "test_rubric",
                    "criteria": [
                        {"name": "quality", "description": "Overall quality", "min_score": 1, "max_score": 5},
                    ],
                },
            )

        assert result["success"] is True


# ── Multi-judge evaluation ────────────────────────────


class TestEvaluateMultiJudge:
    @pytest.mark.asyncio
    async def test_two_judges_aggregated(self):
        svc = AdvancedJudgeService()
        mock_llm = _make_mock_llm(POINTWISE_RESPONSE)

        with patch.object(svc, "_get_llm", return_value=mock_llm):
            result = await svc.evaluate_multi_judge(
                question="Q",
                generated="A",
                reference="R",
                judges=[
                    {"model": "gpt-4o", "judge_id": "judge1"},
                    {"model": "gpt-4o-mini", "judge_id": "judge2"},
                ],
            )

        assert result["success"] is True
        assert "aggregated" in result

    @pytest.mark.asyncio
    async def test_pairwise_multi_judge(self):
        svc = AdvancedJudgeService()
        mock_llm = _make_mock_llm(PAIRWISE_RESPONSE)

        with patch.object(svc, "_get_llm", return_value=mock_llm):
            result = await svc.evaluate_multi_judge(
                question="Q",
                generated="A",
                generated_b="B",
                judge_type="pairwise",
                judges=[
                    {"model": "gpt-4o"},
                    {"model": "gpt-4o-mini"},
                ],
            )

        assert result["success"] is True
        assert "aggregated" in result

    @pytest.mark.asyncio
    async def test_weighted_aggregation(self):
        svc = AdvancedJudgeService()
        mock_llm = _make_mock_llm(POINTWISE_RESPONSE)

        with patch.object(svc, "_get_llm", return_value=mock_llm):
            result = await svc.evaluate_multi_judge(
                question="Q",
                generated="A",
                reference="R",
                judges=[
                    {"model": "gpt-4o", "weight": 2.0},
                    {"model": "gpt-4o-mini", "weight": 1.0},
                ],
                aggregation="weighted",
            )

        assert result["success"] is True


# ── Batch evaluation ──────────────────────────────────


class TestEvaluateBatch:
    @pytest.mark.asyncio
    async def test_batch_evaluation(self):
        svc = AdvancedJudgeService()
        mock_llm = _make_mock_llm(POINTWISE_RESPONSE)

        test_data = [
            {"instruction": "Q1", "generated": "A1", "output": "R1"},
            {"instruction": "Q2", "generated": "A2", "output": "R2"},
        ]

        with patch.object(svc, "_get_llm", return_value=mock_llm):
            result = await svc.evaluate_batch(test_data=test_data)

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["results"]) == 2
        assert "summary" in result

    @pytest.mark.asyncio
    async def test_batch_with_custom_criteria(self):
        custom_response = json.dumps({
            "clarity": {"score": 8, "reason": "Clear"},
        })
        svc = AdvancedJudgeService()
        mock_llm = _make_mock_llm(custom_response)

        test_data = [
            {"instruction": "Q1", "generated": "A1", "output": "R1"},
        ]

        with patch.object(svc, "_get_llm", return_value=mock_llm):
            result = await svc.evaluate_batch(
                test_data=test_data,
                criteria=[{"name": "clarity", "description": "Is it clear?"}],
            )

        assert result["success"] is True


# ── List judge types ──────────────────────────────────


class TestListJudgeTypes:
    def test_returns_all_types(self):
        svc = AdvancedJudgeService()
        result = svc.list_judge_types()

        assert result["success"] is True
        assert "pointwise" in result["judge_types"]
        assert "pairwise" in result["judge_types"]
        assert "reference_free" in result["judge_types"]
        assert "rubric" in result["judge_types"]

    def test_returns_descriptions(self):
        svc = AdvancedJudgeService()
        result = svc.list_judge_types()

        assert "details" in result
        assert "pointwise" in result["details"]


# ── Export ─────────────────────────────────────────────


class TestExport:
    @pytest.mark.asyncio
    async def test_export_jsonl(self):
        svc = AdvancedJudgeService()
        results = [
            {"question": "Q1", "score": 8},
            {"question": "Q2", "score": 7},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            result = await svc.export_results(results, path, format="jsonl")
            assert result["success"] is True
            assert result["num_results"] == 2

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_export_json(self):
        svc = AdvancedJudgeService()
        results = [{"question": "Q1"}]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            result = await svc.export_results(results, path, format="json")
            assert result["success"] is True

            with open(path) as f:
                data = json.load(f)
            assert len(data) == 1
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_export_unsupported_format(self):
        svc = AdvancedJudgeService()
        result = await svc.export_results([], "out.csv", format="csv")
        assert result["success"] is False
