"""Unit tests for the model evaluation pipeline.

Tests cover: ROUGE scoring, BERTScore, LLM-as-Judge,
batch evaluation, export (JSONL + Excel), and summary statistics.
"""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config import ModelEvaluationConfig


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_mock_llm_response(content: str):
    """Create a mock LLM response object with optional logprobs."""
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = None
    resp.usage = {"prompt_tokens": 100, "completion_tokens": 50}
    return resp


SAMPLE_JUDGE_RESPONSE = json.dumps({
    "correctness": {"score": 8, "reason": "Accurate answer"},
    "completeness": {"score": 7, "reason": "Mostly complete"},
    "factuality": {"score": 9, "reason": "Facts are correct"},
    "structure": {"score": 9, "reason": "Well organized"},
    "hallucination_resistance": {"score": 10, "reason": "No hallucinations"},
})


# ──────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────

class TestModelEvaluationConfig:
    def test_defaults(self):
        cfg = ModelEvaluationConfig()
        assert cfg.model == "gpt-4o"
        assert cfg.metrics == ["rouge", "bertscore", "llm_judge"]
        assert cfg.max_new_tokens == 1024
        assert cfg.temperature == 0.1
        assert cfg.judge_model == "gpt-4o"

    def test_custom_values(self):
        cfg = ModelEvaluationConfig(
            model="gpt-4o-mini",
            metrics=["rouge"],
            max_new_tokens=512,
            temperature=0.5,
        )
        assert cfg.model == "gpt-4o-mini"
        assert cfg.metrics == ["rouge"]
        assert cfg.max_new_tokens == 512


# ──────────────────────────────────────────────
# ROUGE metric tests
# ──────────────────────────────────────────────

class TestRougeMetric:
    @pytest.mark.asyncio
    async def test_compute_rouge_identical(self):
        from model_evaluator_pipeline.metrics.rouge import compute_rouge

        result = await compute_rouge(
            generated="The cat sat on the mat.",
            reference="The cat sat on the mat.",
        )
        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result
        assert "rougeLsum" in result
        # Identical strings should yield perfect scores
        assert result["rouge1"] == pytest.approx(1.0)
        assert result["rouge2"] == pytest.approx(1.0)
        assert result["rougeL"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_compute_rouge_different(self):
        from model_evaluator_pipeline.metrics.rouge import compute_rouge

        result = await compute_rouge(
            generated="The dog ran across the field.",
            reference="The cat sat on the mat.",
        )
        # Different strings should yield lower scores
        assert result["rouge1"] < 1.0
        assert result["rouge2"] < 1.0

    @pytest.mark.asyncio
    async def test_compute_rouge_empty(self):
        from model_evaluator_pipeline.metrics.rouge import compute_rouge

        result = await compute_rouge(generated="", reference="Some text here.")
        assert result["rouge1"] == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_compute_rouge_returns_floats(self):
        from model_evaluator_pipeline.metrics.rouge import compute_rouge

        result = await compute_rouge(
            generated="Hello world test",
            reference="Hello world example",
        )
        for key in ("rouge1", "rouge2", "rougeL", "rougeLsum"):
            assert isinstance(result[key], float)
            assert 0.0 <= result[key] <= 1.0


# ──────────────────────────────────────────────
# BERTScore metric tests
# ──────────────────────────────────────────────

class TestBertScoreMetric:
    @pytest.mark.asyncio
    async def test_compute_bertscore_returns_keys(self):
        from model_evaluator_pipeline.metrics.bertscore import compute_bertscore

        with patch("model_evaluator_pipeline.metrics.bertscore.bert_score.score") as mock_score:
            # Mock BERTScore to avoid downloading models in tests
            import torch
            mock_score.return_value = (
                torch.tensor([0.92]),   # precision
                torch.tensor([0.88]),   # recall
                torch.tensor([0.90]),   # f1
            )

            result = await compute_bertscore(
                generated="The cat sat on the mat.",
                reference="The cat sat on the mat.",
            )

        assert "bert_precision" in result
        assert "bert_recall" in result
        assert "bert_f1" in result

    @pytest.mark.asyncio
    async def test_compute_bertscore_values(self):
        from model_evaluator_pipeline.metrics.bertscore import compute_bertscore

        with patch("model_evaluator_pipeline.metrics.bertscore.bert_score.score") as mock_score:
            import torch
            mock_score.return_value = (
                torch.tensor([0.95]),
                torch.tensor([0.93]),
                torch.tensor([0.94]),
            )

            result = await compute_bertscore(
                generated="test output",
                reference="test reference",
            )

        assert result["bert_precision"] == pytest.approx(0.95, abs=0.01)
        assert result["bert_recall"] == pytest.approx(0.93, abs=0.01)
        assert result["bert_f1"] == pytest.approx(0.94, abs=0.01)

    @pytest.mark.asyncio
    async def test_compute_bertscore_custom_model(self):
        from model_evaluator_pipeline.metrics.bertscore import compute_bertscore

        with patch("model_evaluator_pipeline.metrics.bertscore.bert_score.score") as mock_score:
            import torch
            mock_score.return_value = (
                torch.tensor([0.85]),
                torch.tensor([0.80]),
                torch.tensor([0.82]),
            )

            await compute_bertscore(
                generated="test",
                reference="test",
                model_type="distilbert-base-uncased",
            )

            # Verify custom model was passed
            mock_score.assert_called_once()
            call_kwargs = mock_score.call_args
            assert call_kwargs[1]["model_type"] == "distilbert-base-uncased"


# ──────────────────────────────────────────────
# LLM-as-Judge metric tests
# ──────────────────────────────────────────────

class TestLLMJudge:
    @pytest.mark.asyncio
    async def test_llm_judge_parses_response(self):
        from model_evaluator_pipeline.metrics.llm_judge import llm_judge

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value=_make_mock_llm_response(SAMPLE_JUDGE_RESPONSE)
        )

        result = await llm_judge(
            question="What is Python?",
            generated="Python is a programming language.",
            reference="Python is a high-level programming language.",
            llm=mock_llm,
        )

        assert result["correctness"]["score"] == 8
        assert result["completeness"]["score"] == 7
        assert result["factuality"]["score"] == 9
        assert result["structure"]["score"] == 9
        assert result["hallucination_resistance"]["score"] == 10

    @pytest.mark.asyncio
    async def test_llm_judge_handles_bad_json(self):
        from model_evaluator_pipeline.metrics.llm_judge import llm_judge

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value=_make_mock_llm_response("not valid json at all")
        )

        result = await llm_judge(
            question="What is X?",
            generated="X is Y.",
            reference="X is Z.",
            llm=mock_llm,
        )

        # Should return zero scores on parse failure
        assert result["correctness"]["score"] == 0
        assert result["completeness"]["score"] == 0

    @pytest.mark.asyncio
    async def test_llm_judge_criteria_keys(self):
        from model_evaluator_pipeline.metrics.llm_judge import llm_judge, JUDGE_CRITERIA

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value=_make_mock_llm_response(SAMPLE_JUDGE_RESPONSE)
        )

        result = await llm_judge(
            question="Q", generated="G", reference="R", llm=mock_llm,
        )

        for criterion in JUDGE_CRITERIA:
            assert criterion in result
            assert "score" in result[criterion]
            assert "reason" in result[criterion]

    @pytest.mark.asyncio
    async def test_llm_judge_scores_clamped(self):
        from model_evaluator_pipeline.metrics.llm_judge import llm_judge

        bad_scores = json.dumps({
            "correctness": {"score": 15, "reason": "over"},
            "completeness": {"score": -3, "reason": "under"},
            "factuality": {"score": 5, "reason": "ok"},
            "structure": {"score": 5, "reason": "ok"},
            "hallucination_resistance": {"score": 5, "reason": "ok"},
        })

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value=_make_mock_llm_response(bad_scores)
        )

        result = await llm_judge(
            question="Q", generated="G", reference="R", llm=mock_llm,
        )

        assert result["correctness"]["score"] == 10  # clamped to max
        assert result["completeness"]["score"] == 1   # clamped to min


# ──────────────────────────────────────────────
# ModelEvaluationService tests
# ──────────────────────────────────────────────

class TestModelEvaluationService:
    @pytest.mark.asyncio
    async def test_evaluate_single_rouge_only(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig(metrics=["rouge"])
        svc = ModelEvaluationService(config)

        result = await svc.evaluate_single(
            question="What is Python?",
            generated="Python is a programming language.",
            reference="Python is a programming language.",
            metrics=["rouge"],
        )

        assert result["success"] is True
        assert "rouge" in result["scores"]
        assert result["scores"]["rouge"]["rouge1"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_evaluate_single_bertscore_mocked(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig(metrics=["bertscore"])
        svc = ModelEvaluationService(config)

        with patch("model_evaluator_pipeline.metrics.bertscore.bert_score.score") as mock_score:
            import torch
            mock_score.return_value = (
                torch.tensor([0.9]),
                torch.tensor([0.85]),
                torch.tensor([0.87]),
            )

            result = await svc.evaluate_single(
                question="Q",
                generated="A generated answer.",
                reference="A reference answer.",
                metrics=["bertscore"],
            )

        assert result["success"] is True
        assert "bertscore" in result["scores"]
        assert result["scores"]["bertscore"]["bert_f1"] == pytest.approx(0.87, abs=0.01)

    @pytest.mark.asyncio
    async def test_evaluate_single_llm_judge(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value=_make_mock_llm_response(SAMPLE_JUDGE_RESPONSE)
        )

        config = ModelEvaluationConfig(metrics=["llm_judge"])
        svc = ModelEvaluationService(config, llm=mock_llm)

        result = await svc.evaluate_single(
            question="Q",
            generated="G",
            reference="R",
            metrics=["llm_judge"],
        )

        assert result["success"] is True
        assert "llm_judge" in result["scores"]
        assert result["scores"]["llm_judge"]["correctness"]["score"] == 8

    @pytest.mark.asyncio
    async def test_evaluate_single_all_metrics(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        mock_llm = MagicMock()
        mock_llm.chat = AsyncMock(
            return_value=_make_mock_llm_response(SAMPLE_JUDGE_RESPONSE)
        )

        config = ModelEvaluationConfig(metrics=["rouge", "bertscore", "llm_judge"])
        svc = ModelEvaluationService(config, llm=mock_llm)

        with patch("model_evaluator_pipeline.metrics.bertscore.bert_score.score") as mock_score:
            import torch
            mock_score.return_value = (
                torch.tensor([0.9]),
                torch.tensor([0.85]),
                torch.tensor([0.87]),
            )

            result = await svc.evaluate_single(
                question="What is Python?",
                generated="Python is a language.",
                reference="Python is a programming language.",
            )

        assert result["success"] is True
        assert "rouge" in result["scores"]
        assert "bertscore" in result["scores"]
        assert "llm_judge" in result["scores"]

    @pytest.mark.asyncio
    async def test_evaluate_batch_without_inference(self):
        """Test batch evaluation with pre-generated outputs (no model inference)."""
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig(metrics=["rouge"])
        svc = ModelEvaluationService(config)

        test_data = [
            {
                "instruction": "What is Python?",
                "output": "Python is a programming language.",
                "generated": "Python is a programming language.",
            },
            {
                "instruction": "What is Java?",
                "output": "Java is a programming language.",
                "generated": "Java is a language for building apps.",
            },
        ]

        result = await svc.evaluate_batch(
            test_data=test_data,
            metrics=["rouge"],
        )

        assert result["success"] is True
        assert len(result["results"]) == 2
        assert "summary" in result
        assert "rouge" in result["summary"]
        # First result should have perfect rouge (identical strings)
        assert result["results"][0]["scores"]["rouge"]["rouge1"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_evaluate_batch_summary_stats(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig(metrics=["rouge"])
        svc = ModelEvaluationService(config)

        test_data = [
            {
                "instruction": "Q1",
                "output": "Reference A",
                "generated": "Reference A",
            },
            {
                "instruction": "Q2",
                "output": "Reference B",
                "generated": "Something completely different",
            },
        ]

        result = await svc.evaluate_batch(test_data=test_data, metrics=["rouge"])

        summary = result["summary"]
        assert "rouge" in summary
        rouge_stats = summary["rouge"]
        assert "rouge1" in rouge_stats
        assert "min" in rouge_stats["rouge1"]
        assert "max" in rouge_stats["rouge1"]
        assert "mean" in rouge_stats["rouge1"]
        assert "stdev" in rouge_stats["rouge1"]
        # One perfect match, one poor match: max should be 1.0
        assert rouge_stats["rouge1"]["max"] == pytest.approx(1.0)


# ──────────────────────────────────────────────
# Export tests
# ──────────────────────────────────────────────

class TestExport:
    @pytest.mark.asyncio
    async def test_export_jsonl(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig()
        svc = ModelEvaluationService(config)

        results = [
            {"instruction": "Q1", "generated": "A1", "reference": "R1", "scores": {"rouge": {"rouge1": 0.9}}},
            {"instruction": "Q2", "generated": "A2", "reference": "R2", "scores": {"rouge": {"rouge1": 0.7}}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.jsonl")
            result = await svc.export_results(results, path, format="jsonl")

            assert result["success"] is True
            assert result["num_results"] == 2
            assert os.path.exists(path)

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
            parsed = json.loads(lines[0])
            assert parsed["instruction"] == "Q1"

    @pytest.mark.asyncio
    async def test_export_xlsx(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig()
        svc = ModelEvaluationService(config)

        results = [
            {"instruction": "Q1", "generated": "A1", "reference": "R1", "scores": {"rouge": {"rouge1": 0.9}}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.xlsx")
            result = await svc.export_results(results, path, format="xlsx")

            assert result["success"] is True
            assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_export_json(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig()
        svc = ModelEvaluationService(config)

        results = [
            {"instruction": "Q1", "generated": "A1", "reference": "R1", "scores": {}},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            result = await svc.export_results(results, path, format="json")

            assert result["success"] is True
            assert os.path.exists(path)

            with open(path) as f:
                data = json.load(f)
            assert len(data) == 1


# ──────────────────────────────────────────────
# Summary statistics tests
# ──────────────────────────────────────────────

class TestSummaryStatistics:
    @pytest.mark.asyncio
    async def test_compute_summary_rouge(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig()
        svc = ModelEvaluationService(config)

        results = [
            {"scores": {"rouge": {"rouge1": 0.9, "rouge2": 0.8, "rougeL": 0.85, "rougeLsum": 0.85}}},
            {"scores": {"rouge": {"rouge1": 0.7, "rouge2": 0.6, "rougeL": 0.65, "rougeLsum": 0.65}}},
            {"scores": {"rouge": {"rouge1": 0.5, "rouge2": 0.4, "rougeL": 0.45, "rougeLsum": 0.45}}},
        ]

        summary = svc.compute_summary(results)

        assert "rouge" in summary
        assert summary["rouge"]["rouge1"]["mean"] == pytest.approx(0.7, abs=0.01)
        assert summary["rouge"]["rouge1"]["min"] == pytest.approx(0.5, abs=0.01)
        assert summary["rouge"]["rouge1"]["max"] == pytest.approx(0.9, abs=0.01)

    @pytest.mark.asyncio
    async def test_compute_summary_llm_judge(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig()
        svc = ModelEvaluationService(config)

        results = [
            {"scores": {"llm_judge": {
                "correctness": {"score": 8, "reason": "good"},
                "completeness": {"score": 7, "reason": "ok"},
                "factuality": {"score": 9, "reason": "great"},
                "structure": {"score": 8, "reason": "fine"},
                "hallucination_resistance": {"score": 10, "reason": "none"},
            }}},
            {"scores": {"llm_judge": {
                "correctness": {"score": 6, "reason": "ok"},
                "completeness": {"score": 5, "reason": "partial"},
                "factuality": {"score": 7, "reason": "mostly"},
                "structure": {"score": 6, "reason": "ok"},
                "hallucination_resistance": {"score": 8, "reason": "minor"},
            }}},
        ]

        summary = svc.compute_summary(results)

        assert "llm_judge" in summary
        assert summary["llm_judge"]["correctness"]["mean"] == pytest.approx(7.0)
        assert summary["llm_judge"]["correctness"]["min"] == pytest.approx(6.0)
        assert summary["llm_judge"]["correctness"]["max"] == pytest.approx(8.0)

    @pytest.mark.asyncio
    async def test_compute_summary_empty(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig()
        svc = ModelEvaluationService(config)

        summary = svc.compute_summary([])
        assert summary == {}


# ──────────────────────────────────────────────
# Perplexity metric integration tests
# ──────────────────────────────────────────────

class TestPerplexityMetric:
    @pytest.mark.asyncio
    async def test_evaluate_single_perplexity(self):
        """Perplexity metric is invoked when listed in metrics."""
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig(metrics=["perplexity"], api_key="test-key")
        svc = ModelEvaluationService(config)

        with patch("model_evaluator_pipeline.services.evaluation_service.compute_perplexity") as mock_ppl:
            mock_ppl.return_value = {"perplexity": 5.2, "mean_logprob": -1.6, "num_tokens": 10}

            result = await svc.evaluate_single(
                question="What is X?", generated="X is Y.", reference="X is Y.",
                metrics=["perplexity"],
            )

        assert result["success"] is True
        assert "perplexity" in result["scores"]
        assert result["scores"]["perplexity"]["perplexity"] == pytest.approx(5.2)


# ──────────────────────────────────────────────
# Flatten result tests
# ──────────────────────────────────────────────

class TestFlattenResult:
    def test_flatten_rouge(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        svc = ModelEvaluationService()
        row = {
            "instruction": "Q", "generated": "A", "reference": "R",
            "scores": {"rouge": {"rouge1": 0.9, "rouge2": 0.8, "rougeL": 0.85, "rougeLsum": 0.85}},
        }
        flat = svc.flatten_result(row)
        assert flat["rouge1"] == 0.9
        assert flat["rouge2"] == 0.8
        assert flat["rougeL"] == 0.85
        assert "scores" not in flat

    def test_flatten_llm_judge(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        svc = ModelEvaluationService()
        row = {
            "instruction": "Q", "generated": "A", "reference": "R",
            "scores": {"llm_judge": {
                "correctness": {"score": 8, "reason": "good"},
                "completeness": {"score": 7, "reason": "ok"},
            }},
        }
        flat = svc.flatten_result(row)
        assert flat["llm_judge_correctness"] == 8
        assert flat["llm_judge_completeness"] == 7
        assert "scores" not in flat

    @pytest.mark.asyncio
    async def test_batch_with_flatten(self):
        from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService

        config = ModelEvaluationConfig(metrics=["rouge"])
        svc = ModelEvaluationService(config)

        test_data = [
            {"instruction": "Q1", "output": "Ref A", "generated": "Ref A"},
            {"instruction": "Q2", "output": "Ref B", "generated": "Different"},
        ]

        result = await svc.evaluate_batch(test_data=test_data, metrics=["rouge"], flatten=True)

        assert result["success"] is True
        # Flattened results should have rouge1 at top level, not nested under scores
        for row in result["results"]:
            assert "rouge1" in row
            assert "scores" not in row
