"""Evaluator service layer — scoring, filtering, and statistics."""

import statistics
from typing import Any, Dict, List, Optional

from shared.config import EvaluatorConfig
from shared.providers import SyncLLMAdapter
from shared.provider_factory import create_llm
from shared.registry import metric_registry

from ..core.data import DataPoint
from ..core.evaluator import MetricEvaluator
from ..core.metrics.complexity import ComplexityMetric
from ..core.metrics.ifd import InstructionFollowingDifficultyMetric
from ..core.metrics.quality import LLMQualityMetric


class EvaluatorService:
    """MCP-ready service for evaluating fine-tuning datasets."""

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        self.config = config or EvaluatorConfig()
        llm = create_llm(self.config)
        self.sync_llm = SyncLLMAdapter(llm)
        self._evaluator = self._build_evaluator()

    def _build_evaluator(self) -> MetricEvaluator:
        metrics = [
            ComplexityMetric(language_code=self.config.language),
            InstructionFollowingDifficultyMetric(),
            LLMQualityMetric(self.sync_llm),
        ]
        return MetricEvaluator(metrics, self.config.weights)

    async def evaluate_dataset(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Score all data points with configured metrics."""
        dps = [
            DataPoint(
                instruction=dp.get("instruction", ""),
                input=dp.get("input", ""),
                output=dp.get("output", ""),
                metadata=dp.get("metadata"),
            )
            for dp in data_points
        ]

        scored = []
        for dp in dps:
            scores = {}
            for metric in self._evaluator.metrics:
                scores[metric.name] = metric.compute(dp)
            weighted = sum(
                scores.get(k, 0) * w
                for k, w in self._evaluator.weights.items()
            )

            entry = {
                "instruction": dp.instruction,
                "input": dp.input,
                "output": dp.output,
                "scores": scores,
                "weighted_score": weighted,
                "metadata": dp.metadata,
            }
            scored.append(entry)

        return {
            "success": True,
            "data_points": scored,
            "count": len(scored),
        }

    async def filter_by_quality(
        self,
        data_points: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Return only entries above the quality threshold."""
        threshold = threshold if threshold is not None else self.config.threshold

        # If data_points don't have scores yet, evaluate them first
        if data_points and "weighted_score" not in data_points[0]:
            eval_result = await self.evaluate_dataset(data_points)
            data_points = eval_result["data_points"]

        filtered = [dp for dp in data_points if dp.get("weighted_score", 0) >= threshold]

        return {
            "success": True,
            "data_points": filtered,
            "original_count": len(data_points),
            "filtered_count": len(filtered),
            "threshold": threshold,
        }

    async def analyze_statistics(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return per-metric min/max/mean/std."""
        if data_points and "scores" not in data_points[0]:
            eval_result = await self.evaluate_dataset(data_points)
            data_points = eval_result["data_points"]

        all_scores: Dict[str, List[float]] = {}
        weighted_scores: List[float] = []

        for dp in data_points:
            for metric_name, score in dp.get("scores", {}).items():
                all_scores.setdefault(metric_name, []).append(score)
            weighted_scores.append(dp.get("weighted_score", 0))

        stats: Dict[str, Any] = {}
        for name, values in all_scores.items():
            stats[name] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0,
            }

        stats["weighted"] = {
            "min": min(weighted_scores) if weighted_scores else 0,
            "max": max(weighted_scores) if weighted_scores else 0,
            "mean": statistics.mean(weighted_scores) if weighted_scores else 0,
            "stdev": statistics.stdev(weighted_scores) if len(weighted_scores) > 1 else 0,
        }

        return {
            "success": True,
            "statistics": stats,
            "total_data_points": len(data_points),
        }

    def list_metrics(self) -> Dict[str, Any]:
        """List all registered metrics from the shared metric_registry."""
        return {
            "success": True,
            "metrics": metric_registry.list_keys(),
            "all": metric_registry.list_all(),
        }
