"""Advanced judge service: orchestrates multi-judge, multi-criteria evaluation."""
from __future__ import annotations

import asyncio
import json
import statistics as stats_mod
import time
from typing import Any, Dict, List, Optional

from shared.config import AdvancedJudgeConfig
from shared.multimodal_models import count_image_blocks, normalize_content_blocks
from shared.provider_factory import create_llm
from shared.providers import BaseLLM

from ..judges.base import BaseJudge
from ..judges.models import (
    AggregationStrategy,
    CriterionScore,
    JudgeCriterion,
    JudgeModelConfig,
    JudgeRubric,
    JudgeType,
    PairwiseResult,
    SingleJudgeResult,
)
from ..judges.aggregation import aggregate_pairwise, aggregate_pointwise
from ..judges.pointwise import DEFAULT_POINTWISE_CRITERIA

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

    async def evaluate_vlm_single(
        self,
        messages: List[Dict[str, Any]],
        generated: str,
        reference: Optional[str] = None,
        judge_model: str = "gpt-4o",
        criteria: Optional[List[Dict[str, Any]]] = None,
        rubric: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single multimodal pointwise judge evaluation."""
        try:
            normalized_messages = self._normalize_vlm_messages(messages)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        rubric_obj = JudgeRubric(**rubric) if rubric else None
        criteria_obj = [JudgeCriterion(**c) for c in criteria] if criteria else None
        criteria_list = rubric_obj.criteria if rubric_obj else criteria_obj or DEFAULT_POINTWISE_CRITERIA
        llm = self._get_llm(judge_model)

        start_time = time.perf_counter()
        try:
            raw = await llm.chat(
                [
                    {
                        "role": "system",
                        "content": self._build_vlm_pointwise_system_prompt(criteria_list),
                    },
                    {
                        "role": "user",
                        "content": self._build_vlm_pointwise_user_content(
                            normalized_messages,
                            generated=generated,
                            reference=reference,
                        ),
                    },
                ],
                temperature=0.0,
            )
            latency = time.perf_counter() - start_time
            parsed = self._parse_json_response(raw.content or "")
            if parsed is None:
                result = self._build_vlm_pointwise_error_result(
                    llm=llm,
                    criteria=criteria_list,
                    latency=latency,
                    error="JSON parse failure",
                    raw=raw.content or "",
                )
            else:
                scores = self._extract_pointwise_scores(parsed, criteria_list)
                result = SingleJudgeResult(
                    judge_id=llm.model_id,
                    judge_type=JudgeType.POINTWISE,
                    model=llm.model_id,
                    criteria_scores=scores,
                    overall_score=self._compute_pointwise_overall(scores, criteria_list),
                    raw_response=raw.content or "",
                    latency_s=round(latency, 4),
                )
        except Exception as exc:
            latency = time.perf_counter() - start_time
            result = self._build_vlm_pointwise_error_result(
                llm=llm,
                criteria=criteria_list,
                latency=latency,
                error=str(exc),
            )

        return {
            "success": True,
            "result": result.model_dump(),
            "modality": "vision-language",
            "image_count": count_image_blocks(normalized_messages),
        }

    async def compare_vlm(
        self,
        messages: List[Dict[str, Any]],
        generated_a: str,
        generated_b: str,
        reference: Optional[str] = None,
        judge_model: str = "gpt-4o",
        criteria: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run a single multimodal pairwise comparison."""
        try:
            normalized_messages = self._normalize_vlm_messages(messages)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        criteria_obj = [JudgeCriterion(**c) for c in criteria] if criteria else []
        llm = self._get_llm(judge_model)
        start_time = time.perf_counter()
        try:
            raw = await llm.chat(
                [
                    {
                        "role": "system",
                        "content": self._build_vlm_pairwise_system_prompt(criteria_obj),
                    },
                    {
                        "role": "user",
                        "content": self._build_vlm_pairwise_user_content(
                            normalized_messages,
                            generated_a=generated_a,
                            generated_b=generated_b,
                            reference=reference,
                        ),
                    },
                ],
                temperature=0.0,
            )
            latency = time.perf_counter() - start_time
            parsed = self._parse_json_response(raw.content or "")
            result = self._parse_pairwise_result(
                parsed=parsed,
                llm=llm,
                latency=latency,
                raw=raw.content or "",
            )
        except Exception as exc:
            latency = time.perf_counter() - start_time
            result = PairwiseResult(
                judge_id=llm.model_id,
                model=llm.model_id,
                winner="tie",
                confidence=0.0,
                reason=str(exc),
                latency_s=round(latency, 4),
                success=False,
                error=str(exc),
            )

        return {
            "success": True,
            "result": result.model_dump(),
            "modality": "vision-language",
            "image_count": count_image_blocks(normalized_messages),
        }

    async def evaluate_vlm_batch(
        self,
        test_data: List[Dict[str, Any]],
        judge_model: str = "gpt-4o",
        criteria: Optional[List[Dict[str, Any]]] = None,
        rubric: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a batch of multimodal rows using pointwise judging."""
        results: List[Dict[str, Any]] = []
        for row in test_data:
            messages = row.get("messages")
            generated = str(row.get("generated", "")).strip()
            if not generated:
                results.append(
                    {
                        "success": False,
                        "error": "generated is required for VLM batch evaluation",
                        "messages": messages,
                    }
                )
                continue

            reference = row.get("reference")
            if not isinstance(reference, str) or not reference.strip():
                reference = self._extract_assistant_text(messages)

            result = await self.evaluate_vlm_single(
                messages=messages,
                generated=generated,
                reference=reference,
                judge_model=judge_model,
                criteria=criteria,
                rubric=rubric,
            )
            result["generated"] = generated
            result["reference"] = reference
            result["messages"] = messages
            results.append(result)

        return {
            "success": True,
            "results": results,
            "summary": self._compute_vlm_batch_summary(results),
            "count": len(results),
            "modality": "vision-language",
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

    def _normalize_vlm_messages(
        self, messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        normalized: List[Dict[str, Any]] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).strip().lower()
            if not role:
                continue
            content = normalize_content_blocks(message.get("content"))
            if not content:
                continue
            normalized.append({"role": role, "content": content})

        if not normalized:
            raise ValueError("messages must contain at least one valid content block")
        if count_image_blocks(normalized) == 0:
            raise ValueError("messages must include at least one image block for VLM evaluation")
        return normalized

    def _parse_json_response(self, raw: str) -> Optional[Dict[str, Any]]:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

    def _clamp_score(self, value: Any, min_score: int, max_score: int) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(min_score)
        return max(float(min_score), min(float(max_score), numeric))

    def _build_criteria_prompt_section(
        self, criteria: List[JudgeCriterion],
    ) -> str:
        lines = []
        for criterion in criteria:
            lines.append(
                f"- {criterion.name} ({criterion.min_score}-{criterion.max_score}): {criterion.description}"
            )
            if criterion.score_levels:
                for level in criterion.score_levels:
                    lines.append(
                        f"    {level.min_score}-{level.max_score}: {level.description}"
                    )
        return "\n".join(lines)

    def _build_json_format_section(
        self, criteria: List[JudgeCriterion],
    ) -> str:
        entries = []
        for criterion in criteria:
            entries.append(
                f'  "{criterion.name}": {{"score": <{criterion.min_score}-{criterion.max_score}>, '
                f'"reason": "<brief explanation>"}}'
            )
        return "{\n" + ",\n".join(entries) + "\n}"

    def _build_vlm_pointwise_system_prompt(
        self, criteria: List[JudgeCriterion],
    ) -> str:
        return (
            "You are an expert multimodal evaluator. Assess the candidate answer "
            "using both the provided text and the attached images.\n\n"
            "Criteria:\n"
            f"{self._build_criteria_prompt_section(criteria)}\n\n"
            "Return ONLY a JSON object with this exact format:\n"
            f"{self._build_json_format_section(criteria)}"
        )

    def _build_vlm_pairwise_system_prompt(
        self, criteria: List[JudgeCriterion],
    ) -> str:
        criteria_section = (
            self._build_criteria_prompt_section(criteria)
            if criteria
            else "Use your best judgment on overall quality, grounding, and usefulness."
        )
        return (
            "You are an expert multimodal evaluator. Compare two candidate answers "
            "against the same multimodal input, using both text and images.\n\n"
            "Evaluation criteria:\n"
            f"{criteria_section}\n\n"
            'Return ONLY a JSON object: {"winner":"A or B or tie","confidence":<0.0-1.0>,"reason":"<brief explanation>"}'
        )

    def _build_vlm_pointwise_user_content(
        self,
        messages: List[Dict[str, Any]],
        generated: str,
        reference: Optional[str],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Review the multimodal conversation below. The images are embedded in the "
                    "conversation blocks that follow."
                ),
            }
        ]
        content.extend(self._conversation_blocks(messages))
        content.append({"type": "text", "text": f"Candidate answer:\n{generated}"})
        if reference:
            content.append({"type": "text", "text": f"Reference answer:\n{reference}"})
        return content

    def _build_vlm_pairwise_user_content(
        self,
        messages: List[Dict[str, Any]],
        generated_a: str,
        generated_b: str,
        reference: Optional[str],
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Review the multimodal conversation below. Compare Response A and "
                    "Response B using the full text and image context."
                ),
            }
        ]
        content.extend(self._conversation_blocks(messages))
        content.append({"type": "text", "text": f"Response A:\n{generated_a}"})
        content.append({"type": "text", "text": f"Response B:\n{generated_b}"})
        if reference:
            content.append({"type": "text", "text": f"Reference answer:\n{reference}"})
        return content

    def _conversation_blocks(
        self, messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        for index, message in enumerate(messages, start=1):
            role = str(message.get("role", "user")).strip().lower()
            blocks.append(
                {
                    "type": "text",
                    "text": f"Conversation message {index} ({role}):",
                }
            )
            blocks.extend(normalize_content_blocks(message.get("content")))
        return blocks

    def _extract_pointwise_scores(
        self,
        parsed: Dict[str, Any],
        criteria: List[JudgeCriterion],
    ) -> List[CriterionScore]:
        scores: List[CriterionScore] = []
        for criterion in criteria:
            entry = parsed.get(criterion.name, {})
            if isinstance(entry, dict):
                scores.append(
                    CriterionScore(
                        criterion=criterion.name,
                        score=self._clamp_score(
                            entry.get("score", criterion.min_score),
                            criterion.min_score,
                            criterion.max_score,
                        ),
                        reason=str(entry.get("reason", "")),
                        min_score=criterion.min_score,
                        max_score=criterion.max_score,
                    )
                )
            else:
                scores.append(
                    CriterionScore(
                        criterion=criterion.name,
                        score=float(criterion.min_score),
                        reason="Missing in judge response",
                        min_score=criterion.min_score,
                        max_score=criterion.max_score,
                    )
                )
        return scores

    def _compute_pointwise_overall(
        self,
        scores: List[Any],
        criteria: List[JudgeCriterion],
    ) -> float:
        total_weight = sum(criterion.weight for criterion in criteria)
        if total_weight == 0:
            return 0.0

        weighted_sum = 0.0
        for score, criterion in zip(scores, criteria):
            weighted_sum += score.normalized_score * criterion.weight
        return round(weighted_sum / total_weight, 4)

    def _build_vlm_pointwise_error_result(
        self,
        llm: BaseLLM,
        criteria: List[JudgeCriterion],
        latency: float,
        error: str,
        raw: str = "",
    ) -> SingleJudgeResult:
        return SingleJudgeResult(
            judge_id=llm.model_id,
            judge_type=JudgeType.POINTWISE,
            model=llm.model_id,
            criteria_scores=[
                CriterionScore(
                    criterion=criterion.name,
                    score=float(criterion.min_score),
                    reason=error,
                    min_score=criterion.min_score,
                    max_score=criterion.max_score,
                )
                for criterion in criteria
            ],
            overall_score=0.0,
            raw_response=raw,
            latency_s=round(latency, 4),
            success=False,
            error=error,
        )

    def _parse_pairwise_result(
        self,
        parsed: Optional[Dict[str, Any]],
        llm: BaseLLM,
        latency: float,
        raw: str,
    ) -> PairwiseResult:
        if parsed is None:
            return PairwiseResult(
                judge_id=llm.model_id,
                model=llm.model_id,
                winner="tie",
                confidence=0.0,
                reason="JSON parse failure",
                latency_s=round(latency, 4),
                success=False,
                error="JSON parse failure",
            )

        winner = str(parsed.get("winner", "tie")).strip().upper()
        if winner == "TIE":
            winner = "tie"
        elif winner not in {"A", "B"}:
            winner = "tie"

        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5

        return PairwiseResult(
            judge_id=llm.model_id,
            model=llm.model_id,
            winner=winner,
            confidence=confidence,
            reason=str(parsed.get("reason", raw if raw else "")),
            latency_s=round(latency, 4),
            success=winner in {"A", "B", "tie"},
            error=None if winner in {"A", "B", "tie"} else "Invalid winner",
        )

    def _extract_assistant_text(self, messages: Any) -> Optional[str]:
        if not isinstance(messages, list):
            return None
        texts: List[str] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            if str(message.get("role", "")).strip().lower() != "assistant":
                continue
            for block in normalize_content_blocks(message.get("content")):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text = block["text"].strip()
                    if text:
                        texts.append(text)
        return "\n".join(texts) if texts else None

    def _compute_vlm_batch_summary(
        self, results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        successful = [
            row for row in results
            if row.get("success") is True
            and isinstance(row.get("result"), dict)
            and row["result"].get("success", True) is not False
        ]
        failed = len(results) - len(successful)

        overall_scores: List[float] = []
        image_counts: List[int] = []
        criterion_scores: Dict[str, List[float]] = {}

        for row in successful:
            result = row.get("result", {})
            overall = result.get("overall_score")
            if isinstance(overall, (int, float)):
                overall_scores.append(float(overall))
            image_count = row.get("image_count")
            if isinstance(image_count, int):
                image_counts.append(image_count)
            for item in result.get("criteria_scores", []):
                if not isinstance(item, dict):
                    continue
                criterion = item.get("criterion")
                score = item.get("score")
                if isinstance(criterion, str) and isinstance(score, (int, float)):
                    criterion_scores.setdefault(criterion, []).append(float(score))

        def _stats(values: List[float]) -> Dict[str, float]:
            if not values:
                return {}
            return {
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "mean": round(stats_mod.mean(values), 4),
                "stdev": round(stats_mod.stdev(values), 4) if len(values) > 1 else 0.0,
            }

        return {
            "successful_rows": len(successful),
            "failed_rows": failed,
            "overall": _stats(overall_scores),
            "per_criterion": {
                criterion: _stats(values)
                for criterion, values in criterion_scores.items()
            },
            "images_per_sample": _stats([float(value) for value in image_counts]),
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
