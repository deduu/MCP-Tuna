"""Multi-judge aggregation: combine scores from multiple judges."""
from __future__ import annotations

import statistics
from typing import Dict, List, Optional

from .models import (
    AggregatedResult,
    AggregationStrategy,
    JudgeModelConfig,
    PairwiseAggregatedResult,
    PairwiseResult,
    SingleJudgeResult,
)


def aggregate_pointwise(
    results: List[SingleJudgeResult],
    strategy: AggregationStrategy = AggregationStrategy.MEAN,
    judge_configs: Optional[List[JudgeModelConfig]] = None,
) -> AggregatedResult:
    """Aggregate scores from multiple pointwise judges on one sample.

    Args:
        results: Per-judge results for a single evaluated sample.
        strategy: How to combine scores.
        judge_configs: Used for weighted aggregation (maps by judge_id).

    Returns:
        AggregatedResult with aggregated_scores, overall_score, and agreement.
    """
    successful = [r for r in results if r.success]
    if not successful:
        return AggregatedResult(
            judge_results=results,
            aggregated_scores={},
            overall_score=0.0,
            agreement={},
            aggregation_strategy=strategy,
        )

    # Collect normalized scores per criterion
    criterion_normalized: Dict[str, List[float]] = {}
    for r in successful:
        for cs in r.criteria_scores:
            criterion_normalized.setdefault(cs.criterion, []).append(
                cs.normalized_score
            )

    # Build weight map from configs
    weight_map: Dict[str, float] = {}
    if judge_configs:
        for jc in judge_configs:
            weight_map[jc.judge_id or jc.model] = jc.weight

    # Aggregate per criterion
    aggregated: Dict[str, float] = {}
    for crit, scores in criterion_normalized.items():
        if strategy == AggregationStrategy.MEAN:
            aggregated[crit] = round(statistics.mean(scores), 4)
        elif strategy == AggregationStrategy.MEDIAN:
            aggregated[crit] = round(statistics.median(scores), 4)
        elif strategy == AggregationStrategy.WEIGHTED:
            weighted_sum = 0.0
            total_w = 0.0
            for i, r in enumerate(successful):
                w = weight_map.get(r.judge_id, 1.0)
                weighted_sum += scores[i] * w
                total_w += w
            aggregated[crit] = round(weighted_sum / total_w, 4) if total_w > 0 else 0.0
        elif strategy == AggregationStrategy.MAJORITY:
            rounded = [round(s * 10) for s in scores]
            try:
                mode_val = statistics.mode(rounded)
            except statistics.StatisticsError:
                mode_val = round(statistics.mean(rounded))
            aggregated[crit] = round(mode_val / 10.0, 4)

    # Agreement: 1 - (stdev / 0.5) on normalized [0,1] scores
    agreement: Dict[str, float] = {}
    for crit, scores in criterion_normalized.items():
        if len(scores) < 2:
            agreement[crit] = 1.0
        else:
            stdev = statistics.stdev(scores)
            agreement[crit] = round(max(0.0, 1.0 - stdev / 0.5), 4)

    overall = round(statistics.mean(aggregated.values()), 4) if aggregated else 0.0

    return AggregatedResult(
        judge_results=results,
        aggregated_scores=aggregated,
        overall_score=overall,
        agreement=agreement,
        aggregation_strategy=strategy,
    )


def aggregate_pairwise(
    results: List[PairwiseResult],
) -> PairwiseAggregatedResult:
    """Aggregate pairwise comparison results from multiple judges.

    Uses majority vote to determine the winner.
    """
    win_counts: Dict[str, int] = {"A": 0, "B": 0, "tie": 0}
    for r in results:
        if r.success:
            key = r.winner if r.winner in win_counts else "tie"
            win_counts[key] += 1

    total_votes = sum(win_counts.values())
    if total_votes == 0:
        winner = "tie"
        agreement_rate = 0.0
    else:
        winner = max(win_counts, key=lambda k: win_counts[k])
        agreement_rate = round(win_counts[winner] / total_votes, 4)

    return PairwiseAggregatedResult(
        pairwise_results=results,
        winner=winner,
        agreement_rate=agreement_rate,
        win_counts=win_counts,
    )
