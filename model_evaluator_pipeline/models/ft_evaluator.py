"""Domain knowledge evaluator data models for fine-tuned model evaluation.

Implements the K-Type failure taxonomy:
  K-Gap, K-Hallucination, K-Outdated, K-Leakage, K-Instruction

And KSMI answerability labels:
  DOC_ANSWERABLE, EXPERT_OOD
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class FailureType(str, Enum):
    K_GAP = "K_GAP"
    K_HALLUCINATION = "K_HALLUCINATION"
    K_OUTDATED = "K_OUTDATED"
    K_LEAKAGE = "K_LEAKAGE"
    K_INSTRUCTION = "K_INSTRUCTION"


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


class SuggestedAction(str, Enum):
    ADD_FINETUNING_DATA = "ADD_FINETUNING_DATA"
    KNOWLEDGE_PATCHING = "KNOWLEDGE_PATCHING"
    DPO_ALIGNMENT = "DPO_ALIGNMENT"
    ADD_NEGATIVE_CONSTRAINTS = "ADD_NEGATIVE_CONSTRAINTS"


class KSMILabel(str, Enum):
    DOC_ANSWERABLE = "DOC_ANSWERABLE"
    EXPERT_OOD = "EXPERT_OOD"


class FTEvalVerdict(BaseModel):
    """One judge's verdict on one sample."""

    verdict: Verdict
    failure_type: Optional[FailureType] = None
    severity: Optional[Severity] = None
    suggested_action: Optional[SuggestedAction] = None
    reasoning: str
    judge_model: str


class FTEvalResult(BaseModel):
    """One sample with all judges' verdicts."""

    instruction: str
    generated: str
    reference: str
    ksmi_label: Optional[KSMILabel] = None
    verdicts: List[FTEvalVerdict]
    consensus_verdict: Optional[Verdict] = None


class FTEvalSummary(BaseModel):
    """Batch statistics for domain knowledge evaluation."""

    total_samples: int
    pass_count: int
    fail_count: int
    pass_rate: float
    failure_type_distribution: Dict[str, int]
    severity_distribution: Dict[str, int]
    action_distribution: Dict[str, int]
    pass_rate_by_ksmi_label: Dict[str, float]
    per_judge_pass_rate: Dict[str, float]
