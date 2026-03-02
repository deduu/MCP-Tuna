"""Tests for Pydantic v2 judge data models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from model_evaluator_pipeline.judges.models import (
    AggregatedResult,
    AggregationStrategy,
    CriterionScore,
    JudgeCriterion,
    JudgeModelConfig,
    JudgeRubric,
    JudgeType,
    MultiJudgeConfig,
    PairwiseResult,
    ScoreLevel,
    SingleJudgeResult,
)


# ── JudgeCriterion ─────────────────────────────────────


class TestJudgeCriterion:
    def test_defaults(self):
        c = JudgeCriterion(name="accuracy", description="How accurate?")
        assert c.min_score == 1
        assert c.max_score == 10
        assert c.weight == 1.0
        assert c.score_levels is None

    def test_custom_range(self):
        c = JudgeCriterion(
            name="quality", description="Overall quality", min_score=0, max_score=5
        )
        assert c.min_score == 0
        assert c.max_score == 5

    def test_invalid_range_raises(self):
        with pytest.raises(ValidationError):
            JudgeCriterion(
                name="bad", description="min >= max", min_score=10, max_score=5
            )

    def test_equal_range_raises(self):
        with pytest.raises(ValidationError):
            JudgeCriterion(
                name="bad", description="min == max", min_score=5, max_score=5
            )

    def test_with_score_levels(self):
        c = JudgeCriterion(
            name="quality",
            description="Quality metric",
            min_score=1,
            max_score=4,
            score_levels=[
                ScoreLevel(min_score=1, max_score=1, description="Poor"),
                ScoreLevel(min_score=2, max_score=2, description="Fair"),
                ScoreLevel(min_score=3, max_score=3, description="Good"),
                ScoreLevel(min_score=4, max_score=4, description="Excellent"),
            ],
        )
        assert len(c.score_levels) == 4

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            JudgeCriterion(name="", description="No name")

    def test_serialization_roundtrip(self):
        c = JudgeCriterion(name="test", description="Test criterion", weight=2.0)
        data = c.model_dump()
        c2 = JudgeCriterion(**data)
        assert c == c2


# ── JudgeRubric ────────────────────────────────────────


class TestJudgeRubric:
    def test_basic_rubric(self):
        r = JudgeRubric(
            name="medical",
            criteria=[
                JudgeCriterion(name="accuracy", description="Medical accuracy"),
                JudgeCriterion(name="safety", description="Safety warnings"),
            ],
        )
        assert r.criteria_names == ["accuracy", "safety"]
        assert r.total_weight == 2.0

    def test_empty_criteria_raises(self):
        with pytest.raises(ValidationError):
            JudgeRubric(name="bad", criteria=[])

    def test_custom_prompt_template(self):
        r = JudgeRubric(
            name="custom",
            criteria=[JudgeCriterion(name="c1", description="d1")],
            prompt_template="Custom template: {question}",
        )
        assert r.prompt_template is not None

    def test_weighted_criteria(self):
        r = JudgeRubric(
            name="weighted",
            criteria=[
                JudgeCriterion(name="a", description="d", weight=3.0),
                JudgeCriterion(name="b", description="d", weight=1.0),
            ],
        )
        assert r.total_weight == 4.0


# ── JudgeModelConfig ───────────────────────────────────


class TestJudgeModelConfig:
    def test_defaults(self):
        c = JudgeModelConfig()
        assert c.model == "gpt-4o"
        assert c.judge_id == "gpt-4o"  # auto-derived
        assert c.temperature == 0.1
        assert c.timeout_s == 30.0
        assert c.weight == 1.0

    def test_custom_judge_id(self):
        c = JudgeModelConfig(model="gpt-4o", judge_id="primary-judge")
        assert c.judge_id == "primary-judge"

    def test_auto_judge_id(self):
        c = JudgeModelConfig(model="claude-3-opus")
        assert c.judge_id == "claude-3-opus"


# ── MultiJudgeConfig ──────────────────────────────────


class TestMultiJudgeConfig:
    def test_defaults(self):
        c = MultiJudgeConfig()
        assert len(c.judges) == 1
        assert c.aggregation == AggregationStrategy.MEAN

    def test_multiple_judges(self):
        c = MultiJudgeConfig(
            judges=[
                JudgeModelConfig(model="gpt-4o"),
                JudgeModelConfig(model="gpt-4o-mini"),
            ],
            aggregation=AggregationStrategy.WEIGHTED,
        )
        assert len(c.judges) == 2


# ── CriterionScore ─────────────────────────────────────


class TestCriterionScore:
    def test_normalized_score_default_range(self):
        s = CriterionScore(criterion="test", score=5.5, min_score=1, max_score=10)
        assert s.normalized_score == pytest.approx(0.5)

    def test_normalized_score_min(self):
        s = CriterionScore(criterion="test", score=1, min_score=1, max_score=10)
        assert s.normalized_score == pytest.approx(0.0)

    def test_normalized_score_max(self):
        s = CriterionScore(criterion="test", score=10, min_score=1, max_score=10)
        assert s.normalized_score == pytest.approx(1.0)

    def test_normalized_score_custom_range(self):
        s = CriterionScore(criterion="test", score=3, min_score=0, max_score=5)
        assert s.normalized_score == pytest.approx(0.6)

    def test_normalized_score_zero_span(self):
        s = CriterionScore(criterion="test", score=5, min_score=5, max_score=5)
        assert s.normalized_score == 0.0


# ── SingleJudgeResult ──────────────────────────────────


class TestSingleJudgeResult:
    def test_success_result(self):
        r = SingleJudgeResult(
            judge_id="gpt-4o",
            judge_type=JudgeType.POINTWISE,
            model="gpt-4o",
            criteria_scores=[
                CriterionScore(criterion="c1", score=8, reason="Good"),
            ],
            overall_score=0.778,
            latency_s=1.23,
        )
        assert r.success is True
        assert r.error is None

    def test_error_result(self):
        r = SingleJudgeResult(
            judge_id="gpt-4o",
            judge_type=JudgeType.POINTWISE,
            model="gpt-4o",
            criteria_scores=[],
            overall_score=0.0,
            success=False,
            error="Timeout",
        )
        assert r.success is False


# ── PairwiseResult ─────────────────────────────────────


class TestPairwiseResult:
    def test_valid_result(self):
        r = PairwiseResult(
            judge_id="gpt-4o",
            model="gpt-4o",
            winner="A",
            confidence=0.85,
            reason="Output A is more accurate",
        )
        assert r.winner == "A"
        assert r.success is True

    def test_confidence_clamped(self):
        with pytest.raises(ValidationError):
            PairwiseResult(
                judge_id="gpt-4o",
                model="gpt-4o",
                winner="A",
                confidence=1.5,
            )


# ── AggregatedResult ──────────────────────────────────


class TestAggregatedResult:
    def test_basic_aggregation(self):
        r = AggregatedResult(
            judge_results=[],
            aggregated_scores={"accuracy": 0.75, "completeness": 0.80},
            overall_score=0.775,
            agreement={"accuracy": 0.95, "completeness": 0.88},
            aggregation_strategy=AggregationStrategy.MEAN,
        )
        assert r.overall_score == 0.775


# ── Enums ──────────────────────────────────────────────


class TestEnums:
    def test_judge_types(self):
        assert JudgeType.POINTWISE.value == "pointwise"
        assert JudgeType.PAIRWISE.value == "pairwise"
        assert JudgeType.REFERENCE_FREE.value == "reference_free"
        assert JudgeType.RUBRIC.value == "rubric"

    def test_aggregation_strategies(self):
        assert AggregationStrategy.MEAN.value == "mean"
        assert AggregationStrategy.MEDIAN.value == "median"
        assert AggregationStrategy.WEIGHTED.value == "weighted"
        assert AggregationStrategy.MAJORITY.value == "majority"
