"""Pydantic v2 models for the advanced LLM-as-a-judge evaluation system."""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ── Enums ─────────────────────────────────────────────


class JudgeType(str, Enum):
    """Supported judge evaluation paradigms."""

    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    REFERENCE_FREE = "reference_free"
    RUBRIC = "rubric"


class AggregationStrategy(str, Enum):
    """How to combine scores from multiple judges."""

    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED = "weighted"
    MAJORITY = "majority"


# ── Criteria / Rubrics ────────────────────────────────


class ScoreLevel(BaseModel):
    """A single level within a rubric score range."""

    min_score: int
    max_score: int
    description: str


class JudgeCriterion(BaseModel):
    """A single evaluation criterion (user-definable)."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1)
    min_score: int = Field(default=1, ge=0)
    max_score: int = Field(default=10, ge=1)
    weight: float = Field(default=1.0, ge=0.0)
    score_levels: Optional[List[ScoreLevel]] = None

    @model_validator(mode="after")
    def check_score_range(self) -> JudgeCriterion:
        if self.min_score >= self.max_score:
            msg = f"min_score ({self.min_score}) must be < max_score ({self.max_score})"
            raise ValueError(msg)
        return self


class JudgeRubric(BaseModel):
    """A named collection of criteria forming a complete rubric."""

    name: str = Field(..., min_length=1)
    description: str = ""
    criteria: List[JudgeCriterion] = Field(..., min_length=1)
    prompt_template: Optional[str] = None

    @property
    def criteria_names(self) -> List[str]:
        return [c.name for c in self.criteria]

    @property
    def total_weight(self) -> float:
        return sum(c.weight for c in self.criteria)


# ── Judge Configuration ───────────────────────────────


class JudgeModelConfig(BaseModel):
    """Configuration for a single judge LLM."""

    model: str = "gpt-4o"
    judge_id: Optional[str] = None
    temperature: float = 0.1
    timeout_s: float = 30.0
    weight: float = 1.0

    @model_validator(mode="after")
    def set_default_judge_id(self) -> JudgeModelConfig:
        if self.judge_id is None:
            self.judge_id = self.model
        return self


class MultiJudgeConfig(BaseModel):
    """Configuration for running multiple judges."""

    judges: List[JudgeModelConfig] = Field(
        default_factory=lambda: [JudgeModelConfig()]
    )
    aggregation: AggregationStrategy = AggregationStrategy.MEAN


# ── Results ───────────────────────────────────────────


class CriterionScore(BaseModel):
    """Score for a single criterion from a single judge."""

    criterion: str
    score: float
    reason: str = ""
    min_score: int = 1
    max_score: int = 10

    @property
    def normalized_score(self) -> float:
        """Normalize score to [0, 1] range."""
        span = self.max_score - self.min_score
        if span <= 0:
            return 0.0
        return (self.score - self.min_score) / span


class SingleJudgeResult(BaseModel):
    """Result from one judge evaluating one sample."""

    judge_id: str
    judge_type: JudgeType
    model: str
    criteria_scores: List[CriterionScore]
    overall_score: Optional[float] = None
    raw_response: Optional[str] = None
    latency_s: float = 0.0
    success: bool = True
    error: Optional[str] = None


class PairwiseResult(BaseModel):
    """Result from a pairwise comparison judge."""

    judge_id: str
    model: str
    winner: str  # "A", "B", or "tie"
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    latency_s: float = 0.0
    success: bool = True
    error: Optional[str] = None


class AggregatedResult(BaseModel):
    """Aggregated result from multiple judges on one sample."""

    judge_results: List[SingleJudgeResult]
    aggregated_scores: Dict[str, float]
    overall_score: float
    agreement: Dict[str, float]
    aggregation_strategy: AggregationStrategy


class PairwiseAggregatedResult(BaseModel):
    """Aggregated pairwise results from multiple judges."""

    pairwise_results: List[PairwiseResult]
    winner: str  # "A", "B", or "tie"
    agreement_rate: float
    win_counts: Dict[str, int]


class JudgeEvaluationResult(BaseModel):
    """Full result for a single evaluated sample across all judges."""

    question: str
    generated: str
    reference: Optional[str] = None
    generated_b: Optional[str] = None
    per_judge: List[SingleJudgeResult] = Field(default_factory=list)
    pairwise: Optional[PairwiseAggregatedResult] = None
    aggregated: Optional[AggregatedResult] = None


class BatchJudgeResult(BaseModel):
    """Result for a batch evaluation."""

    success: bool
    results: List[JudgeEvaluationResult]
    summary: Dict[str, Any] = Field(default_factory=dict)
    count: int = 0
