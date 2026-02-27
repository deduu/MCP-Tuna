"""Tests for multi-judge aggregation logic."""
from __future__ import annotations

import pytest

from model_evaluator_pipeline.judges.aggregation import (
    aggregate_pairwise,
    aggregate_pointwise,
)
from model_evaluator_pipeline.judges.models import (
    AggregatedResult,
    AggregationStrategy,
    CriterionScore,
    JudgeModelConfig,
    JudgeType,
    PairwiseAggregatedResult,
    PairwiseResult,
    SingleJudgeResult,
)


# ── Helpers ────────────────────────────────────────────


def _make_pointwise_result(
    judge_id: str,
    scores: dict[str, float],
    success: bool = True,
) -> SingleJudgeResult:
    """Create a SingleJudgeResult with given criterion scores (1-10 range)."""
    return SingleJudgeResult(
        judge_id=judge_id,
        judge_type=JudgeType.POINTWISE,
        model=judge_id,
        criteria_scores=[
            CriterionScore(
                criterion=name,
                score=score,
                reason="test",
                min_score=1,
                max_score=10,
            )
            for name, score in scores.items()
        ],
        overall_score=0.5,
        success=success,
    )


def _make_pairwise_result(
    judge_id: str,
    winner: str,
    confidence: float = 0.8,
    success: bool = True,
) -> PairwiseResult:
    return PairwiseResult(
        judge_id=judge_id,
        model=judge_id,
        winner=winner,
        confidence=confidence,
        reason="test",
        success=success,
    )


# ── Pointwise Aggregation: MEAN ──────────────────────


class TestAggregatePointwiseMean:
    def test_single_judge(self):
        results = [
            _make_pointwise_result("j1", {"accuracy": 8, "completeness": 6}),
        ]
        agg = aggregate_pointwise(results, AggregationStrategy.MEAN)

        assert isinstance(agg, AggregatedResult)
        assert agg.aggregation_strategy == AggregationStrategy.MEAN
        # Normalized: (8-1)/9 ≈ 0.7778, (6-1)/9 ≈ 0.5556
        assert agg.aggregated_scores["accuracy"] == pytest.approx(0.7778, abs=0.001)
        assert agg.aggregated_scores["completeness"] == pytest.approx(0.5556, abs=0.001)

    def test_two_judges_averaged(self):
        results = [
            _make_pointwise_result("j1", {"accuracy": 8}),
            _make_pointwise_result("j2", {"accuracy": 6}),
        ]
        agg = aggregate_pointwise(results, AggregationStrategy.MEAN)

        # Mean of normalized: (7/9 + 5/9) / 2 = 12/18 = 2/3 ≈ 0.6667
        assert agg.aggregated_scores["accuracy"] == pytest.approx(0.6667, abs=0.001)

    def test_failed_judges_excluded(self):
        results = [
            _make_pointwise_result("j1", {"accuracy": 8}),
            _make_pointwise_result("j2", {"accuracy": 2}, success=False),
        ]
        agg = aggregate_pointwise(results, AggregationStrategy.MEAN)

        # Only j1 should be counted
        assert agg.aggregated_scores["accuracy"] == pytest.approx(0.7778, abs=0.001)

    def test_all_failed_returns_empty(self):
        results = [
            _make_pointwise_result("j1", {"accuracy": 8}, success=False),
        ]
        agg = aggregate_pointwise(results, AggregationStrategy.MEAN)

        assert agg.aggregated_scores == {}
        assert agg.overall_score == 0.0


# ── Pointwise Aggregation: MEDIAN ────────────────────


class TestAggregatePointwiseMedian:
    def test_three_judges_median(self):
        results = [
            _make_pointwise_result("j1", {"accuracy": 3}),
            _make_pointwise_result("j2", {"accuracy": 8}),
            _make_pointwise_result("j3", {"accuracy": 5}),
        ]
        agg = aggregate_pointwise(results, AggregationStrategy.MEDIAN)

        # Normalized: 2/9, 7/9, 4/9 → median = 4/9 ≈ 0.4444
        assert agg.aggregated_scores["accuracy"] == pytest.approx(0.4444, abs=0.001)


# ── Pointwise Aggregation: WEIGHTED ──────────────────


class TestAggregatePointwiseWeighted:
    def test_weighted_average(self):
        results = [
            _make_pointwise_result("gpt-4o", {"accuracy": 10}),
            _make_pointwise_result("gpt-4o-mini", {"accuracy": 4}),
        ]
        configs = [
            JudgeModelConfig(model="gpt-4o", judge_id="gpt-4o", weight=2.0),
            JudgeModelConfig(model="gpt-4o-mini", judge_id="gpt-4o-mini", weight=1.0),
        ]
        agg = aggregate_pointwise(
            results, AggregationStrategy.WEIGHTED, judge_configs=configs,
        )

        # gpt-4o: normalized 9/9=1.0, weight 2
        # gpt-4o-mini: normalized 3/9≈0.333, weight 1
        # weighted = (1.0*2 + 0.333*1) / 3 ≈ 0.7778
        assert agg.aggregated_scores["accuracy"] == pytest.approx(0.7778, abs=0.001)


# ── Pointwise Aggregation: MAJORITY ──────────────────


class TestAggregatePointwiseMajority:
    def test_majority_vote(self):
        results = [
            _make_pointwise_result("j1", {"accuracy": 8}),
            _make_pointwise_result("j2", {"accuracy": 8}),
            _make_pointwise_result("j3", {"accuracy": 4}),
        ]
        agg = aggregate_pointwise(results, AggregationStrategy.MAJORITY)

        # Normalized then scaled to 0-10: j1=8, j2=8, j3=4 → mode = 8 → 0.8
        # Actually: normalized = 7/9, 7/9, 3/9
        # Rounded to 0-10: 8, 8, 3 → mode = 8 → 8/10 = 0.8
        assert agg.aggregated_scores["accuracy"] == pytest.approx(0.8, abs=0.01)


# ── Agreement metric ─────────────────────────────────


class TestAgreement:
    def test_perfect_agreement(self):
        results = [
            _make_pointwise_result("j1", {"accuracy": 8}),
            _make_pointwise_result("j2", {"accuracy": 8}),
        ]
        agg = aggregate_pointwise(results, AggregationStrategy.MEAN)

        assert agg.agreement["accuracy"] == pytest.approx(1.0)

    def test_low_agreement(self):
        results = [
            _make_pointwise_result("j1", {"accuracy": 1}),
            _make_pointwise_result("j2", {"accuracy": 10}),
        ]
        agg = aggregate_pointwise(results, AggregationStrategy.MEAN)

        # Normalized: 0/9 and 9/9=1.0 → stdev=0.5 → agreement=1-0.5/0.5=0.0
        assert agg.agreement["accuracy"] == pytest.approx(0.0, abs=0.01)

    def test_single_judge_full_agreement(self):
        results = [
            _make_pointwise_result("j1", {"accuracy": 8}),
        ]
        agg = aggregate_pointwise(results, AggregationStrategy.MEAN)

        assert agg.agreement["accuracy"] == 1.0


# ── Pairwise Aggregation ─────────────────────────────


class TestAggregatePairwise:
    def test_unanimous_winner(self):
        results = [
            _make_pairwise_result("j1", "A"),
            _make_pairwise_result("j2", "A"),
            _make_pairwise_result("j3", "A"),
        ]
        agg = aggregate_pairwise(results)

        assert isinstance(agg, PairwiseAggregatedResult)
        assert agg.winner == "A"
        assert agg.agreement_rate == pytest.approx(1.0)
        assert agg.win_counts == {"A": 3, "B": 0, "tie": 0}

    def test_majority_winner(self):
        results = [
            _make_pairwise_result("j1", "A"),
            _make_pairwise_result("j2", "B"),
            _make_pairwise_result("j3", "A"),
        ]
        agg = aggregate_pairwise(results)

        assert agg.winner == "A"
        assert agg.agreement_rate == pytest.approx(2 / 3, abs=0.001)
        assert agg.win_counts == {"A": 2, "B": 1, "tie": 0}

    def test_tie_result(self):
        results = [
            _make_pairwise_result("j1", "A"),
            _make_pairwise_result("j2", "B"),
        ]
        agg = aggregate_pairwise(results)

        # Both have 1 vote, "A" wins due to max() picking first
        assert agg.agreement_rate == pytest.approx(0.5)

    def test_failed_judges_excluded(self):
        results = [
            _make_pairwise_result("j1", "A"),
            _make_pairwise_result("j2", "B", success=False),
        ]
        agg = aggregate_pairwise(results)

        assert agg.winner == "A"
        assert agg.win_counts["A"] == 1

    def test_all_failed_returns_tie(self):
        results = [
            _make_pairwise_result("j1", "A", success=False),
        ]
        agg = aggregate_pairwise(results)

        assert agg.winner == "tie"
        assert agg.agreement_rate == 0.0

    def test_empty_results(self):
        agg = aggregate_pairwise([])

        assert agg.winner == "tie"
        assert agg.agreement_rate == 0.0
