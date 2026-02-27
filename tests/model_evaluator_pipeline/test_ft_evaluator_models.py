"""Unit tests for FT evaluator Pydantic models and enums."""
from __future__ import annotations

import pytest

from model_evaluator_pipeline.models.ft_evaluator import (
    FailureType,
    FTEvalResult,
    FTEvalSummary,
    FTEvalVerdict,
    KSMILabel,
    Severity,
    SuggestedAction,
    Verdict,
)


# ──────────────────────────────────────────────
# Enum tests
# ──────────────────────────────────────────────

class TestEnums:
    def test_verdict_values(self):
        assert Verdict.PASS == "PASS"
        assert Verdict.FAIL == "FAIL"

    def test_failure_type_values(self):
        assert FailureType.K_GAP == "K_GAP"
        assert FailureType.K_HALLUCINATION == "K_HALLUCINATION"
        assert FailureType.K_OUTDATED == "K_OUTDATED"
        assert FailureType.K_LEAKAGE == "K_LEAKAGE"
        assert FailureType.K_INSTRUCTION == "K_INSTRUCTION"

    def test_severity_values(self):
        assert Severity.CRITICAL == "CRITICAL"
        assert Severity.MAJOR == "MAJOR"
        assert Severity.MINOR == "MINOR"

    def test_suggested_action_values(self):
        assert SuggestedAction.ADD_FINETUNING_DATA == "ADD_FINETUNING_DATA"
        assert SuggestedAction.KNOWLEDGE_PATCHING == "KNOWLEDGE_PATCHING"
        assert SuggestedAction.DPO_ALIGNMENT == "DPO_ALIGNMENT"
        assert SuggestedAction.ADD_NEGATIVE_CONSTRAINTS == "ADD_NEGATIVE_CONSTRAINTS"

    def test_ksmi_label_values(self):
        assert KSMILabel.DOC_ANSWERABLE == "DOC_ANSWERABLE"
        assert KSMILabel.EXPERT_OOD == "EXPERT_OOD"


# ──────────────────────────────────────────────
# FTEvalVerdict tests
# ──────────────────────────────────────────────

class TestFTEvalVerdict:
    def test_pass_verdict(self):
        v = FTEvalVerdict(
            verdict=Verdict.PASS,
            reasoning="Answer matches reference.",
            judge_model="gpt-4o",
        )
        assert v.verdict == Verdict.PASS
        assert v.failure_type is None
        assert v.severity is None
        assert v.suggested_action is None
        assert v.judge_model == "gpt-4o"

    def test_fail_verdict_with_all_fields(self):
        v = FTEvalVerdict(
            verdict=Verdict.FAIL,
            failure_type=FailureType.K_HALLUCINATION,
            severity=Severity.CRITICAL,
            suggested_action=SuggestedAction.DPO_ALIGNMENT,
            reasoning="Model hallucinated facts.",
            judge_model="gpt-4o",
        )
        assert v.verdict == Verdict.FAIL
        assert v.failure_type == FailureType.K_HALLUCINATION
        assert v.severity == Severity.CRITICAL
        assert v.suggested_action == SuggestedAction.DPO_ALIGNMENT

    def test_verdict_serialization(self):
        v = FTEvalVerdict(
            verdict=Verdict.FAIL,
            failure_type=FailureType.K_GAP,
            severity=Severity.MAJOR,
            suggested_action=SuggestedAction.ADD_FINETUNING_DATA,
            reasoning="Missing knowledge.",
            judge_model="deepseek-v3",
        )
        d = v.model_dump()
        assert d["verdict"] == "FAIL"
        assert d["failure_type"] == "K_GAP"
        assert d["severity"] == "MAJOR"
        assert d["judge_model"] == "deepseek-v3"


# ──────────────────────────────────────────────
# FTEvalResult tests
# ──────────────────────────────────────────────

class TestFTEvalResult:
    def test_result_with_single_verdict(self):
        v = FTEvalVerdict(
            verdict=Verdict.PASS, reasoning="Good.", judge_model="gpt-4o",
        )
        r = FTEvalResult(
            instruction="What is X?",
            generated="X is Y.",
            reference="X is Y.",
            verdicts=[v],
        )
        assert len(r.verdicts) == 1
        assert r.consensus_verdict is None

    def test_consensus_verdict_majority_pass(self):
        verdicts = [
            FTEvalVerdict(verdict=Verdict.PASS, reasoning="ok", judge_model="gpt-4o"),
            FTEvalVerdict(verdict=Verdict.PASS, reasoning="ok", judge_model="deepseek"),
            FTEvalVerdict(verdict=Verdict.FAIL, reasoning="no", judge_model="claude"),
        ]
        r = FTEvalResult(
            instruction="Q", generated="A", reference="R",
            verdicts=verdicts,
            consensus_verdict=Verdict.PASS,
        )
        assert r.consensus_verdict == Verdict.PASS

    def test_result_with_ksmi_label(self):
        r = FTEvalResult(
            instruction="Q", generated="A", reference="R",
            ksmi_label=KSMILabel.EXPERT_OOD,
            verdicts=[
                FTEvalVerdict(verdict=Verdict.FAIL, reasoning="OOD", judge_model="gpt-4o"),
            ],
        )
        assert r.ksmi_label == KSMILabel.EXPERT_OOD


# ──────────────────────────────────────────────
# FTEvalSummary tests
# ──────────────────────────────────────────────

class TestFTEvalSummary:
    def test_summary_creation(self):
        s = FTEvalSummary(
            total_samples=10,
            pass_count=7,
            fail_count=3,
            pass_rate=0.7,
            failure_type_distribution={"K_GAP": 2, "K_HALLUCINATION": 1},
            severity_distribution={"CRITICAL": 1, "MAJOR": 2},
            action_distribution={"ADD_FINETUNING_DATA": 2, "DPO_ALIGNMENT": 1},
            pass_rate_by_ksmi_label={"DOC_ANSWERABLE": 0.8, "EXPERT_OOD": 0.5},
            per_judge_pass_rate={"gpt-4o": 0.75, "deepseek": 0.65},
        )
        assert s.total_samples == 10
        assert s.pass_rate == pytest.approx(0.7)
        assert s.failure_type_distribution["K_GAP"] == 2

    def test_summary_serialization_roundtrip(self):
        s = FTEvalSummary(
            total_samples=5,
            pass_count=3,
            fail_count=2,
            pass_rate=0.6,
            failure_type_distribution={},
            severity_distribution={},
            action_distribution={},
            pass_rate_by_ksmi_label={},
            per_judge_pass_rate={},
        )
        d = s.model_dump()
        s2 = FTEvalSummary(**d)
        assert s2.total_samples == 5
        assert s2.pass_rate == pytest.approx(0.6)
