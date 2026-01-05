from typing import List, Dict
from .metrics.base import BaseMetric
from .data import DataPoint


class MetricEvaluator:
    def __init__(self, metrics: List[BaseMetric], weights: Dict[str, float]):
        self.metrics = metrics
        self.weights = weights

    def evaluate(self, dp: DataPoint) -> Dict[str, float]:
        scores = {m.name: m.compute(dp) for m in self.metrics}

        scores["combined_score"] = sum(
            scores[k] * self.weights.get(k, 0.0)
            for k in scores
            if k != "combined_score"
        )
        return scores

    def evaluate_batch(self, dataset: List[DataPoint]):
        return [self.evaluate(dp) for dp in dataset]
