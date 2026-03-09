"""Evaluator service layer — scoring, filtering, and statistics."""

import statistics
from typing import Any, Dict, List, Optional

from shared.async_utils import run_sync
from shared.config import EvaluatorConfig
from shared.registry import metric_registry

from ..core.data import DataPoint
from ..core.evaluator import MetricEvaluator

METRIC_CATALOG: Dict[str, Dict[str, Any]] = {
    "complexity": {
        "description": "Vocabulary richness, structure, and semantic density of the response.",
        "kind": "heuristic",
        "warmup": "Loads spaCy + tokenizer resources on first evaluation run.",
    },
    "ifd": {
        "description": "Instruction-following difficulty based on specificity, output effort, and alignment.",
        "kind": "heuristic",
        "warmup": "Lightweight text scoring.",
    },
    "quality": {
        "description": "LLM-judged response quality scored from 0 to 1.",
        "kind": "llm",
        "warmup": "Uses the configured LLM provider and may be slower than heuristic metrics.",
    },
}


class EvaluatorService:
    """MCP-ready service for evaluating fine-tuning datasets."""

    def __init__(self, config: Optional[EvaluatorConfig] = None):
        self.config = config or EvaluatorConfig()
        self.sync_llm = None
        self._evaluator: Optional[MetricEvaluator] = None

    def _get_evaluator(self) -> MetricEvaluator:
        if self._evaluator is None:
            self._evaluator = self._build_evaluator()
        return self._evaluator

    def _build_evaluator(self) -> MetricEvaluator:
        from shared.providers import SyncLLMAdapter
        from shared.provider_factory import create_llm
        from ..core.metrics.complexity import ComplexityMetric
        from ..core.metrics.ifd import InstructionFollowingDifficultyMetric
        from ..core.metrics.quality import LLMQualityMetric

        if self.sync_llm is None:
            llm = create_llm(self.config)
            self.sync_llm = SyncLLMAdapter(llm)

        metrics = [
            ComplexityMetric(language_code=self.config.language),
            InstructionFollowingDifficultyMetric(),
            LLMQualityMetric(self.sync_llm),
        ]
        return MetricEvaluator(metrics, self.config.weights)

    def _resolve_metrics(
        self,
        metrics: Optional[List[str]],
    ) -> tuple[Optional[List[Any]], Optional[List[str]]]:
        evaluator = self._get_evaluator()
        if not metrics:
            return list(evaluator.metrics), None

        available = {metric.name: metric for metric in evaluator.metrics}
        requested = [name for name in metrics if isinstance(name, str) and name]
        unknown = [name for name in requested if name not in available]
        if unknown:
            return None, unknown

        return [available[name] for name in requested], None

    async def evaluate_dataset(
        self,
        data_points: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Score all data points with configured metrics."""
        return await run_sync(self._evaluate_dataset_sync, data_points, metrics)

    def _evaluate_dataset_sync(
        self,
        data_points: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        selected_metrics, unknown = self._resolve_metrics(metrics)
        if unknown:
            return {
                "success": False,
                "error": f"Unknown metrics requested: {', '.join(unknown)}",
            }
        if not selected_metrics:
            return {"success": False, "error": "No valid metrics selected"}
        evaluator = self._get_evaluator()

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
            for metric in selected_metrics:
                scores[metric.name] = metric.compute(dp)
            weighted = sum(
                scores.get(k, 0) * w
                for k, w in evaluator.weights.items()
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
            "metrics_used": [metric.name for metric in selected_metrics],
        }

    async def filter_by_quality(
        self,
        data_points: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return only entries above the quality threshold."""
        return await run_sync(self._filter_by_quality_sync, data_points, threshold, metrics)

    def _filter_by_quality_sync(
        self,
        data_points: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        threshold = threshold if threshold is not None else self.config.threshold

        # If metrics are provided, always rescore with selected metrics.
        if metrics or (data_points and "weighted_score" not in data_points[0]):
            eval_result = self._evaluate_dataset_sync(data_points, metrics=metrics)
            if not eval_result.get("success"):
                return eval_result
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
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return per-metric min/max/mean/std."""
        return await run_sync(self._analyze_statistics_sync, data_points, metrics)

    def _analyze_statistics_sync(
        self,
        data_points: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if metrics or (data_points and "scores" not in data_points[0]):
            eval_result = self._evaluate_dataset_sync(data_points, metrics=metrics)
            if not eval_result.get("success"):
                return eval_result
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

    def get_config(self) -> Dict[str, Any]:
        return {
            "success": True,
            "config": self.config.model_dump(),
        }

    async def update_config(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await run_sync(self._update_config_sync, weights, threshold, language)

    def _update_config_sync(
        self,
        weights: Optional[Dict[str, float]] = None,
        threshold: Optional[float] = None,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        merged_weights = dict(self.config.weights)
        if weights is not None:
            for name, value in weights.items():
                if not isinstance(name, str) or not name.strip():
                    return {"success": False, "error": "Weight names must be non-empty strings"}
                if not isinstance(value, (int, float)):
                    return {
                        "success": False,
                        "error": f"Weight for '{name}' must be numeric",
                    }
                merged_weights[name.strip()] = float(value)

        if threshold is not None and not 0 <= float(threshold) <= 1:
            return {"success": False, "error": "Threshold must be between 0 and 1"}

        if language is not None and not language.strip():
            return {"success": False, "error": "Language must be a non-empty code"}

        update: Dict[str, Any] = {"weights": merged_weights}
        if threshold is not None:
            update["threshold"] = float(threshold)
        if language is not None:
            update["language"] = language.strip()

        self.config = self.config.model_copy(update=update)
        self._evaluator = None

        return {
            "success": True,
            "config": self.config.model_dump(),
        }

    def list_metrics(self) -> Dict[str, Any]:
        """List all supported metrics without forcing evaluator warm-up."""
        metrics = list(METRIC_CATALOG.keys()) or metric_registry.list_keys()
        return {
            "success": True,
            "metrics": metrics,
            "all": {name: METRIC_CATALOG.get(name, {}) for name in metrics},
        }
