"""Advanced judge service: orchestrates multi-judge, multi-criteria evaluation."""
from __future__ import annotations

import asyncio
import json
import statistics as stats_mod
from typing import Any, Dict, List, Optional

from shared.config import AdvancedJudgeConfig
from shared.provider_factory import create_llm
from shared.providers import BaseLLM

from ..judges.base import BaseJudge
from ..judges.models import (
    AggregationStrategy,
    JudgeCriterion,
    JudgeModelConfig,
    JudgeRubric,
    JudgeType,
    PairwiseResult,
    SingleJudgeResult,
)
from ..judges.aggregation import aggregate_pairwise, aggregate_pointwise

# Import registry to ensure all judge types are auto-registered
from ..judges.registry import judge_registry  # noqa: F401


class AdvancedJudgeService:
    """MCP-ready service for advanced LLM-as-a-judge evaluation.

    Supports:
    - Custom criteria and rubrics
    - Multiple judge models with aggregation
    - Pluggable judge types (pointwise, pairwise, reference-free, rubric)
    """

    def __init__(
        self,
        config: Optional[AdvancedJudgeConfig] = None,
    ):
        self.config = config or AdvancedJudgeConfig()
        self._llm_cache: Dict[str, BaseLLM] = {}

    def _get_llm(self, model: str) -> BaseLLM:
        """Get or create an LLM provider, caching by model name."""
        if model not in self._llm_cache:
            from shared.config import PipelineConfig

            self._llm_cache[model] = create_llm(PipelineConfig(model=model))
        return self._llm_cache[model]

    def _create_judge(
        self,
        judge_type: JudgeType,
        llm: BaseLLM,
        judge_id: Optional[str] = None,
    ) -> BaseJudge:
        """Instantiate a judge from the registry."""
        cls, _meta = judge_registry.get(judge_type.value)
        return cls(llm=llm, judge_id=judge_id)

    # ────────────────────────────────────────────────────
    # Single evaluation with one judge
    # ────────────────────────────────────────────────────
    async def evaluate_single(
        self,
        question: str,
        generated: str,
        reference: Optional[str] = None,
        generated_b: Optional[str] = None,
        judge_type: str = "pointwise",
        judge_model: str = "gpt-4o",
        criteria: Optional[List[Dict[str, Any]]] = None,
        rubric: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single judge evaluation on one sample."""
        jt = JudgeType(judge_type)
        rubric_obj = JudgeRubric(**rubric) if rubric else None
        criteria_obj = [JudgeCriterion(**c) for c in criteria] if criteria else None

        llm = self._get_llm(judge_model)
        judge = self._create_judge(jt, llm)

        result = await judge.evaluate(
            question=question,
            generated=generated,
            reference=reference,
            generated_b=generated_b,
            rubric=rubric_obj,
            criteria=criteria_obj,
        )

        return {
            "success": True,
            "result": result.model_dump(),
        }

    # ────────────────────────────────────────────────────
    # Multi-judge evaluation on one sample
    # ────────────────────────────────────────────────────
    async def evaluate_multi_judge(
        self,
        question: str,
        generated: str,
        reference: Optional[str] = None,
        generated_b: Optional[str] = None,
        judge_type: str = "pointwise",
        judges: Optional[List[Dict[str, Any]]] = None,
        criteria: Optional[List[Dict[str, Any]]] = None,
        rubric: Optional[Dict[str, Any]] = None,
        aggregation: str = "mean",
    ) -> Dict[str, Any]:
        """Run multiple judges in parallel and aggregate results."""
        jt = JudgeType(judge_type)
        agg_strategy = AggregationStrategy(aggregation)
        rubric_obj = JudgeRubric(**rubric) if rubric else None
        criteria_obj = [JudgeCriterion(**c) for c in criteria] if criteria else None
        judge_configs = (
            [JudgeModelConfig(**j) for j in judges]
            if judges
            else [JudgeModelConfig()]
        )

        # Create all judges
        judge_instances = []
        for jc in judge_configs:
            llm = self._get_llm(jc.model)
            judge_instances.append(
                self._create_judge(jt, llm, judge_id=jc.judge_id)
            )

        # Run all judges in parallel
        tasks = [
            j.evaluate(
                question=question,
                generated=generated,
                reference=reference,
                generated_b=generated_b,
                rubric=rubric_obj,
                criteria=criteria_obj,
            )
            for j in judge_instances
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle pairwise vs pointwise aggregation
        if jt == JudgeType.PAIRWISE:
            pairwise_results: List[PairwiseResult] = []
            for r in results:
                if isinstance(r, Exception):
                    pairwise_results.append(
                        PairwiseResult(
                            judge_id="error",
                            model="unknown",
                            winner="tie",
                            confidence=0.0,
                            reason=str(r),
                            success=False,
                            error=str(r),
                        )
                    )
                else:
                    pairwise_results.append(r)

            aggregated = aggregate_pairwise(pairwise_results)
            return {
                "success": True,
                "aggregated": aggregated.model_dump(),
            }
        else:
            judge_results: List[SingleJudgeResult] = []
            for r in results:
                if isinstance(r, Exception):
                    judge_results.append(
                        SingleJudgeResult(
                            judge_id="error",
                            judge_type=jt,
                            model="unknown",
                            criteria_scores=[],
                            overall_score=0.0,
                            success=False,
                            error=str(r),
                        )
                    )
                else:
                    judge_results.append(r)

            aggregated = aggregate_pointwise(
                judge_results, agg_strategy, judge_configs,
            )
            return {
                "success": True,
                "aggregated": aggregated.model_dump(),
            }

    # ────────────────────────────────────────────────────
    # Batch evaluation
    # ────────────────────────────────────────────────────
    async def evaluate_batch(
        self,
        test_data: List[Dict[str, Any]],
        judge_type: str = "pointwise",
        judges: Optional[List[Dict[str, Any]]] = None,
        criteria: Optional[List[Dict[str, Any]]] = None,
        rubric: Optional[Dict[str, Any]] = None,
        aggregation: str = "mean",
    ) -> Dict[str, Any]:
        """Evaluate a batch of samples using multi-judge evaluation."""
        results = []
        for row in test_data:
            question = row.get("instruction", row.get("question", ""))
            generated = row.get("generated", "")
            reference = row.get("output", row.get("reference", None))
            generated_b = row.get("generated_b", None)

            eval_result = await self.evaluate_multi_judge(
                question=question,
                generated=generated,
                reference=reference,
                generated_b=generated_b,
                judge_type=judge_type,
                judges=judges,
                criteria=criteria,
                rubric=rubric,
                aggregation=aggregation,
            )
            eval_result["question"] = question
            eval_result["generated"] = generated
            eval_result["reference"] = reference
            results.append(eval_result)

        summary = self._compute_batch_summary(results, judge_type)

        return {
            "success": True,
            "results": results,
            "summary": summary,
            "count": len(results),
        }

    def _compute_batch_summary(
        self,
        results: List[Dict[str, Any]],
        judge_type: str,
    ) -> Dict[str, Any]:
        """Compute aggregate statistics from batch results."""
        if not results:
            return {}

        if judge_type == "pairwise":
            win_totals: Dict[str, int] = {"A": 0, "B": 0, "tie": 0}
            agreement_rates: List[float] = []
            for r in results:
                agg = r.get("aggregated", {})
                wc = agg.get("win_counts", {})
                for k in win_totals:
                    win_totals[k] += wc.get(k, 0)
                agreement_rates.append(agg.get("agreement_rate", 0.0))
            return {
                "total_win_counts": win_totals,
                "mean_agreement_rate": (
                    round(stats_mod.mean(agreement_rates), 4)
                    if agreement_rates
                    else 0.0
                ),
            }

        # Pointwise / reference_free / rubric
        overall_scores: List[float] = []
        criterion_scores: Dict[str, List[float]] = {}
        agreement_values: Dict[str, List[float]] = {}

        for r in results:
            agg = r.get("aggregated", {})
            overall = agg.get("overall_score", 0.0)
            overall_scores.append(overall)
            for crit, val in agg.get("aggregated_scores", {}).items():
                criterion_scores.setdefault(crit, []).append(val)
            for crit, val in agg.get("agreement", {}).items():
                agreement_values.setdefault(crit, []).append(val)

        def _stats(values: List[float]) -> Dict[str, float]:
            return {
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "mean": round(stats_mod.mean(values), 4),
                "stdev": (
                    round(stats_mod.stdev(values), 4) if len(values) > 1 else 0.0
                ),
            }

        return {
            "overall": _stats(overall_scores) if overall_scores else {},
            "per_criterion": {
                crit: _stats(vals) for crit, vals in criterion_scores.items()
            },
            "agreement": {
                crit: _stats(vals) for crit, vals in agreement_values.items()
            },
        }

    # ────────────────────────────────────────────────────
    # List available judge types
    # ────────────────────────────────────────────────────
    def list_judge_types(self) -> Dict[str, Any]:
        """Return available judge types from the registry."""
        return {
            "success": True,
            "judge_types": judge_registry.list_keys(),
            "details": {
                "pointwise": "Score each output individually against criteria",
                "pairwise": "Compare two outputs head-to-head",
                "reference_free": "Evaluate without ground truth reference",
                "rubric": "Score against detailed scoring rubrics with level descriptions",
            },
        }

    # ────────────────────────────────────────────────────
    # Export
    # ────────────────────────────────────────────────────
    async def export_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        format: str = "jsonl",
    ) -> Dict[str, Any]:
        """Export judge results to file."""
        try:
            if format == "jsonl":
                with open(output_path, "w", encoding="utf-8") as f:
                    for row in results:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                return {"success": False, "error": f"Unsupported format: {format}"}

            return {
                "success": True,
                "output_path": output_path,
                "format": format,
                "num_results": len(results),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
