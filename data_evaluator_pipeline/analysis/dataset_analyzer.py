import numpy as np

from ..core.evaluator import MetricEvaluator


class DatasetAnalyzer:
    def __init__(self, evaluator: MetricEvaluator):
        self.evaluator = evaluator

    def analyze(self, dataset):
        metrics = self.evaluator.evaluate_batch(dataset)

        # ✅ REQUIRED GUARD
        if not metrics:
            return {
                "count": 0,
                "warning": "No samples to analyze"
            }

        summary = {}
        keys = metrics[0].keys()

        for key in keys:
            vals = np.array([m[key] for m in metrics if key in m])
            summary[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

        summary["count"] = len(metrics)
        return summary
