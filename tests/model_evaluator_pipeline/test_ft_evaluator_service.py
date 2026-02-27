"""Unit tests for the FT evaluator service (domain knowledge judge)."""
from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.config import FTEvaluatorConfig


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_chat_response(content: str):
    """Build a mock OpenAI ChatCompletion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


PASS_RESPONSE = json.dumps({
    "verdict": "PASS",
    "failure_type": None,
    "severity": None,
    "suggested_action": None,
    "reasoning": "Answer correctly matches reference.",
})

FAIL_RESPONSE = json.dumps({
    "verdict": "FAIL",
    "failure_type": "K_HALLUCINATION",
    "severity": "CRITICAL",
    "suggested_action": "DPO_ALIGNMENT",
    "reasoning": "Model hallucinated facts not in reference.",
})

FAIL_GAP_RESPONSE = json.dumps({
    "verdict": "FAIL",
    "failure_type": "K_GAP",
    "severity": "MAJOR",
    "suggested_action": "ADD_FINETUNING_DATA",
    "reasoning": "Model lacks knowledge present in training data.",
})


# ──────────────────────────────────────────────
# Config tests
# ──────────────────────────────────────────────

class TestFTEvaluatorConfig:
    def test_defaults(self):
        cfg = FTEvaluatorConfig()
        assert cfg.judge_models == ["gpt-4o"]
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 2048
        assert cfg.system_prompt_path is None

    def test_custom_values(self):
        cfg = FTEvaluatorConfig(
            judge_models=["gpt-4o", "deepseek-v3"],
            temperature=0.1,
            max_tokens=4096,
        )
        assert len(cfg.judge_models) == 2
        assert cfg.max_tokens == 4096


# ──────────────────────────────────────────────
# Single evaluation tests
# ──────────────────────────────────────────────

class TestEvaluateSingle:
    @pytest.mark.asyncio
    async def test_single_pass(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService

        svc = FTEvaluatorService(FTEvaluatorConfig())
        mock_response = _make_chat_response(PASS_RESPONSE)

        with patch("model_evaluator_pipeline.services.ft_evaluator_service.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client

            verdict = await svc.evaluate_single(
                instruction="What is X?",
                generated="X is Y.",
                reference="X is Y.",
                judge_model="gpt-4o",
            )

        assert verdict.verdict.value == "PASS"
        assert verdict.failure_type is None
        assert verdict.judge_model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_single_fail(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService

        svc = FTEvaluatorService(FTEvaluatorConfig())
        mock_response = _make_chat_response(FAIL_RESPONSE)

        with patch("model_evaluator_pipeline.services.ft_evaluator_service.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client

            verdict = await svc.evaluate_single(
                instruction="What is Z?",
                generated="Z is wrong.",
                reference="Z is correct.",
                judge_model="gpt-4o",
            )

        assert verdict.verdict.value == "FAIL"
        assert verdict.failure_type.value == "K_HALLUCINATION"
        assert verdict.severity.value == "CRITICAL"

    @pytest.mark.asyncio
    async def test_single_bad_json_fallback(self):
        """Malformed JSON returns a FAIL verdict with error reasoning."""
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService

        svc = FTEvaluatorService(FTEvaluatorConfig())
        mock_response = _make_chat_response("not valid json at all")

        with patch("model_evaluator_pipeline.services.ft_evaluator_service.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=mock_response)
            MockClient.return_value = client

            verdict = await svc.evaluate_single(
                instruction="Q", generated="A", reference="R",
                judge_model="gpt-4o",
            )

        assert verdict.verdict.value == "FAIL"
        assert "parse" in verdict.reasoning.lower() or "json" in verdict.reasoning.lower()


# ──────────────────────────────────────────────
# Multi-judge tests
# ──────────────────────────────────────────────

class TestEvaluateMultiJudge:
    @pytest.mark.asyncio
    async def test_multi_judge_consensus_pass(self):
        """Majority PASS → consensus PASS."""
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService

        svc = FTEvaluatorService(FTEvaluatorConfig(judge_models=["gpt-4o", "deepseek"]))

        pass_resp = _make_chat_response(PASS_RESPONSE)
        fail_resp = _make_chat_response(FAIL_RESPONSE)

        with patch("model_evaluator_pipeline.services.ft_evaluator_service.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            # First call PASS, second call PASS (both judges)
            client.chat.completions.create = AsyncMock(side_effect=[pass_resp, pass_resp])
            MockClient.return_value = client

            result = await svc.evaluate_multi_judge(
                instruction="Q", generated="A", reference="R",
                judge_models=["gpt-4o", "deepseek"],
            )

        assert result.consensus_verdict == "PASS"
        assert len(result.verdicts) == 2

    @pytest.mark.asyncio
    async def test_multi_judge_consensus_fail(self):
        """Majority FAIL → consensus FAIL."""
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService

        svc = FTEvaluatorService(FTEvaluatorConfig())

        pass_resp = _make_chat_response(PASS_RESPONSE)
        fail_resp = _make_chat_response(FAIL_RESPONSE)

        with patch("model_evaluator_pipeline.services.ft_evaluator_service.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(side_effect=[fail_resp, fail_resp, pass_resp])
            MockClient.return_value = client

            result = await svc.evaluate_multi_judge(
                instruction="Q", generated="A", reference="R",
                judge_models=["gpt-4o", "deepseek", "claude"],
            )

        assert result.consensus_verdict == "FAIL"

    @pytest.mark.asyncio
    async def test_multi_judge_with_ksmi_label(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService

        svc = FTEvaluatorService(FTEvaluatorConfig())

        pass_resp = _make_chat_response(PASS_RESPONSE)

        with patch("model_evaluator_pipeline.services.ft_evaluator_service.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(return_value=pass_resp)
            MockClient.return_value = client

            result = await svc.evaluate_multi_judge(
                instruction="Q", generated="A", reference="R",
                ksmi_label="DOC_ANSWERABLE",
                judge_models=["gpt-4o"],
            )

        assert result.ksmi_label.value == "DOC_ANSWERABLE"


# ──────────────────────────────────────────────
# Batch evaluation tests
# ──────────────────────────────────────────────

class TestEvaluateBatch:
    @pytest.mark.asyncio
    async def test_batch_evaluation(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService

        svc = FTEvaluatorService(FTEvaluatorConfig(judge_models=["gpt-4o"]))

        pass_resp = _make_chat_response(PASS_RESPONSE)
        fail_resp = _make_chat_response(FAIL_RESPONSE)

        test_data = [
            {"instruction": "Q1", "generated": "A1", "reference": "R1"},
            {"instruction": "Q2", "generated": "A2", "reference": "R2"},
        ]

        with patch("model_evaluator_pipeline.services.ft_evaluator_service.AsyncOpenAI") as MockClient:
            client = AsyncMock()
            client.chat.completions.create = AsyncMock(side_effect=[pass_resp, fail_resp])
            MockClient.return_value = client

            result = await svc.evaluate_batch(test_data=test_data)

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_batch_summary_statistics(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService
        from model_evaluator_pipeline.models.ft_evaluator import (
            FTEvalResult, FTEvalVerdict, Verdict, FailureType, Severity,
            SuggestedAction, KSMILabel,
        )

        svc = FTEvaluatorService(FTEvaluatorConfig())

        results = [
            FTEvalResult(
                instruction="Q1", generated="A1", reference="R1",
                ksmi_label=KSMILabel.DOC_ANSWERABLE,
                verdicts=[
                    FTEvalVerdict(verdict=Verdict.PASS, reasoning="ok", judge_model="gpt-4o"),
                ],
                consensus_verdict=Verdict.PASS,
            ),
            FTEvalResult(
                instruction="Q2", generated="A2", reference="R2",
                ksmi_label=KSMILabel.DOC_ANSWERABLE,
                verdicts=[
                    FTEvalVerdict(
                        verdict=Verdict.FAIL, failure_type=FailureType.K_GAP,
                        severity=Severity.MAJOR,
                        suggested_action=SuggestedAction.ADD_FINETUNING_DATA,
                        reasoning="gap", judge_model="gpt-4o",
                    ),
                ],
                consensus_verdict=Verdict.FAIL,
            ),
            FTEvalResult(
                instruction="Q3", generated="A3", reference="R3",
                ksmi_label=KSMILabel.EXPERT_OOD,
                verdicts=[
                    FTEvalVerdict(verdict=Verdict.PASS, reasoning="ok", judge_model="gpt-4o"),
                ],
                consensus_verdict=Verdict.PASS,
            ),
        ]

        summary = svc.compute_summary(results)

        assert summary.total_samples == 3
        assert summary.pass_count == 2
        assert summary.fail_count == 1
        assert summary.pass_rate == pytest.approx(2 / 3, rel=0.01)
        assert summary.failure_type_distribution.get("K_GAP", 0) == 1
        assert summary.severity_distribution.get("MAJOR", 0) == 1
        assert "DOC_ANSWERABLE" in summary.pass_rate_by_ksmi_label
        assert "EXPERT_OOD" in summary.pass_rate_by_ksmi_label

    @pytest.mark.asyncio
    async def test_batch_ksmi_breakdown(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService
        from model_evaluator_pipeline.models.ft_evaluator import (
            FTEvalResult, FTEvalVerdict, Verdict, KSMILabel,
        )

        svc = FTEvaluatorService(FTEvaluatorConfig())

        results = [
            FTEvalResult(
                instruction="Q1", generated="A1", reference="R1",
                ksmi_label=KSMILabel.DOC_ANSWERABLE,
                verdicts=[FTEvalVerdict(verdict=Verdict.PASS, reasoning="ok", judge_model="gpt-4o")],
                consensus_verdict=Verdict.PASS,
            ),
            FTEvalResult(
                instruction="Q2", generated="A2", reference="R2",
                ksmi_label=KSMILabel.DOC_ANSWERABLE,
                verdicts=[FTEvalVerdict(verdict=Verdict.FAIL, reasoning="no", judge_model="gpt-4o")],
                consensus_verdict=Verdict.FAIL,
            ),
        ]

        summary = svc.compute_summary(results)
        assert summary.pass_rate_by_ksmi_label["DOC_ANSWERABLE"] == pytest.approx(0.5)


# ──────────────────────────────────────────────
# Export tests
# ──────────────────────────────────────────────

class TestExport:
    @pytest.mark.asyncio
    async def test_export_jsonl(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService
        from model_evaluator_pipeline.models.ft_evaluator import (
            FTEvalResult, FTEvalVerdict, Verdict,
        )

        svc = FTEvaluatorService(FTEvaluatorConfig())

        results = [
            FTEvalResult(
                instruction="Q1", generated="A1", reference="R1",
                verdicts=[FTEvalVerdict(verdict=Verdict.PASS, reasoning="ok", judge_model="gpt-4o")],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.jsonl")
            export_result = await svc.export_results(results, path, format="jsonl")

            assert export_result["success"] is True
            assert export_result["num_results"] == 1
            assert os.path.exists(path)

            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["instruction"] == "Q1"

    @pytest.mark.asyncio
    async def test_export_json(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService
        from model_evaluator_pipeline.models.ft_evaluator import (
            FTEvalResult, FTEvalVerdict, Verdict,
        )

        svc = FTEvaluatorService(FTEvaluatorConfig())

        results = [
            FTEvalResult(
                instruction="Q1", generated="A1", reference="R1",
                verdicts=[FTEvalVerdict(verdict=Verdict.PASS, reasoning="ok", judge_model="gpt-4o")],
            ),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "results.json")
            export_result = await svc.export_results(results, path, format="json")

            assert export_result["success"] is True
            with open(path) as f:
                data = json.load(f)
            assert len(data) == 1


# ──────────────────────────────────────────────
# Prompt loading tests
# ──────────────────────────────────────────────

class TestPromptLoading:
    @pytest.mark.asyncio
    async def test_loads_bundled_prompt(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService

        svc = FTEvaluatorService(FTEvaluatorConfig())
        prompt = svc._load_system_prompt()
        assert "Domain Knowledge Judge" in prompt
        assert "K_GAP" in prompt

    @pytest.mark.asyncio
    async def test_loads_custom_prompt(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Custom prompt for testing")
            custom_path = f.name

        try:
            svc = FTEvaluatorService(FTEvaluatorConfig(system_prompt_path=custom_path))
            prompt = svc._load_system_prompt()
            assert prompt == "Custom prompt for testing"
        finally:
            os.unlink(custom_path)

    @pytest.mark.asyncio
    async def test_per_judge_pass_rate(self):
        from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService
        from model_evaluator_pipeline.models.ft_evaluator import (
            FTEvalResult, FTEvalVerdict, Verdict,
        )

        svc = FTEvaluatorService(FTEvaluatorConfig())

        results = [
            FTEvalResult(
                instruction="Q1", generated="A1", reference="R1",
                verdicts=[
                    FTEvalVerdict(verdict=Verdict.PASS, reasoning="ok", judge_model="gpt-4o"),
                    FTEvalVerdict(verdict=Verdict.FAIL, reasoning="no", judge_model="deepseek"),
                ],
                consensus_verdict=Verdict.PASS,
            ),
            FTEvalResult(
                instruction="Q2", generated="A2", reference="R2",
                verdicts=[
                    FTEvalVerdict(verdict=Verdict.PASS, reasoning="ok", judge_model="gpt-4o"),
                    FTEvalVerdict(verdict=Verdict.PASS, reasoning="ok", judge_model="deepseek"),
                ],
                consensus_verdict=Verdict.PASS,
            ),
        ]

        summary = svc.compute_summary(results)
        assert summary.per_judge_pass_rate["gpt-4o"] == pytest.approx(1.0)
        assert summary.per_judge_pass_rate["deepseek"] == pytest.approx(0.5)
