"""Model evaluation service — post-training output quality scoring, export, and statistics."""
from __future__ import annotations

import json
import statistics as stats_mod
from typing import Any, Dict, List, Optional

from shared.config import ModelEvaluationConfig
from shared.provider_factory import create_llm
from shared.providers import BaseLLM

from ..metrics.rouge import compute_rouge
from ..metrics.bertscore import compute_bertscore
from ..metrics.llm_judge import llm_judge
from ..metrics.perplexity import compute_perplexity


class ModelEvaluationService:
    """MCP-ready service for evaluating fine-tuned model outputs against references."""

    def __init__(
        self,
        config: Optional[ModelEvaluationConfig] = None,
        llm: Optional[BaseLLM] = None,
    ):
        self.config = config or ModelEvaluationConfig()
        # LLM used for judge calls — lazily created if not provided
        self._llm = llm

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = create_llm(self.config)
        return self._llm

    # ------------------------------------------------------------------ #
    # Single evaluation
    # ------------------------------------------------------------------ #
    async def evaluate_single(
        self,
        question: str,
        generated: str,
        reference: str,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Score a single generated output against a reference.

        Args:
            question: The original instruction / question.
            generated: Model-generated answer.
            reference: Ground truth reference answer.
            metrics: Which metrics to run. Defaults to config.metrics.

        Returns:
            Dict with success, scores (keyed by metric name), and metadata.
        """
        metrics = metrics or self.config.metrics
        scores: Dict[str, Any] = {}

        try:
            if "rouge" in metrics:
                scores["rouge"] = await compute_rouge(generated, reference)

            if "bertscore" in metrics:
                scores["bertscore"] = await compute_bertscore(
                    generated, reference, model_type=self.config.bertscore_model,
                )

            if "llm_judge" in metrics:
                scores["llm_judge"] = await llm_judge(
                    question=question,
                    generated=generated,
                    reference=reference,
                    llm=self.llm,
                )

            if "perplexity" in metrics:
                scores["perplexity"] = await compute_perplexity(
                    question=question,
                    generated=generated,
                    reference=reference,
                    model=self.config.model,
                    api_key=self.config.api_key,
                    api_base=self.config.api_base,
                )

            return {
                "success": True,
                "question": question,
                "generated": generated,
                "reference": reference,
                "scores": scores,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------ #
    # Batch evaluation
    # ------------------------------------------------------------------ #
    async def evaluate_batch(
        self,
        test_data: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None,
        model_path: Optional[str] = None,
        adapter_path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        flatten: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a batch of test examples.

        Each item in test_data should have:
            - instruction: the question / prompt
            - output: ground truth reference
            - generated (optional): pre-generated model output

        If 'generated' is not present and model_path is provided,
        inference is run via the finetuning pipeline's InferenceService.

        Returns:
            Dict with success, results (per-row), and summary statistics.
        """
        metrics = metrics or self.config.metrics
        max_new_tokens = max_new_tokens or self.config.max_new_tokens

        # Run inference if generated outputs are missing
        if test_data and "generated" not in test_data[0] and model_path:
            test_data = await self._run_inference(
                test_data, model_path, adapter_path, max_new_tokens,
            )

        results: List[Dict[str, Any]] = []
        for row in test_data:
            instruction = row.get("instruction", "")
            reference = row.get("output", "")
            generated = row.get("generated", "")

            eval_result = await self.evaluate_single(
                question=instruction,
                generated=generated,
                reference=reference,
                metrics=metrics,
            )

            row_result: Dict[str, Any] = {
                "instruction": instruction,
                "reference": reference,
                "generated": generated,
                "scores": eval_result.get("scores", {}),
            }
            results.append(row_result)

        if flatten:
            results = [self.flatten_result(r) for r in results]

        summary = self.compute_summary(results) if not flatten else {}

        return {
            "success": True,
            "results": results,
            "summary": summary,
            "count": len(results),
        }

    # ------------------------------------------------------------------ #
    # Inference bridge
    # ------------------------------------------------------------------ #
    async def _run_inference(
        self,
        test_data: List[Dict[str, Any]],
        model_path: str,
        adapter_path: Optional[str],
        max_new_tokens: int,
    ) -> List[Dict[str, Any]]:
        """Run inference via the finetuning pipeline's InferenceService."""
        from finetuning_pipeline.services.inference_service import InferenceService

        inference_svc = InferenceService()
        prompts = [row.get("instruction", "") for row in test_data]

        result = await inference_svc.run_inference(
            prompts=prompts,
            model_path=model_path,
            adapter_path=adapter_path,
            max_new_tokens=max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0,
        )

        if result.get("success"):
            for row, inf in zip(test_data, result["results"]):
                row["generated"] = inf["response"]

        return test_data

    # ------------------------------------------------------------------ #
    # Flatten result
    # ------------------------------------------------------------------ #
    def flatten_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Transform nested scores into a flat dict.

        ``{scores: {rouge: {rouge1: 0.9}}}`` → ``{rouge1: 0.9}``
        ``{scores: {llm_judge: {correctness: {score: 8, reason: ...}}}}``
            → ``{llm_judge_correctness: 8}``
        """
        flat: Dict[str, Any] = {
            k: v for k, v in result.items() if k != "scores"
        }
        scores = result.get("scores", {})
        for metric_name, metric_scores in scores.items():
            if isinstance(metric_scores, dict):
                for sub_key, value in metric_scores.items():
                    if isinstance(value, dict) and "score" in value:
                        flat[f"{metric_name}_{sub_key}"] = value["score"]
                    elif isinstance(value, (int, float)):
                        flat[sub_key] = value
        return flat

    # ------------------------------------------------------------------ #
    # Summary statistics
    # ------------------------------------------------------------------ #
    def compute_summary(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute per-metric aggregate statistics from evaluation results.

        Returns nested dict: {metric_name: {sub_metric: {min, max, mean, stdev}}}.
        """
        if not results:
            return {}

        # Collect all score values grouped by metric → sub-metric
        collected: Dict[str, Dict[str, List[float]]] = {}

        for row in results:
            scores = row.get("scores", {})
            for metric_name, metric_scores in scores.items():
                if metric_name not in collected:
                    collected[metric_name] = {}

                if metric_name == "llm_judge":
                    # LLM judge has nested {criterion: {score, reason}} structure
                    for criterion, detail in metric_scores.items():
                        if isinstance(detail, dict) and "score" in detail:
                            collected[metric_name].setdefault(criterion, []).append(
                                float(detail["score"])
                            )
                elif isinstance(metric_scores, dict):
                    # ROUGE and BERTScore have flat {sub_metric: float} structure
                    for sub_key, value in metric_scores.items():
                        if isinstance(value, (int, float)):
                            collected[metric_name].setdefault(sub_key, []).append(
                                float(value)
                            )

        summary: Dict[str, Any] = {}
        for metric_name, sub_metrics in collected.items():
            summary[metric_name] = {}
            for sub_key, values in sub_metrics.items():
                summary[metric_name][sub_key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": stats_mod.mean(values),
                    "stdev": stats_mod.stdev(values) if len(values) > 1 else 0.0,
                }

        return summary

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    async def export_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        format: str = "jsonl",
    ) -> Dict[str, Any]:
        """Export evaluation results to a file.

        Args:
            results: List of evaluation result dicts.
            output_path: Path to write the output file.
            format: One of 'jsonl', 'json', 'xlsx'.

        Returns:
            Dict with success and metadata.
        """
        try:
            if format == "jsonl":
                await self._export_jsonl(results, output_path)
            elif format == "json":
                await self._export_json(results, output_path)
            elif format == "xlsx":
                await self._export_xlsx(results, output_path)
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

    async def _export_jsonl(
        self, results: List[Dict[str, Any]], path: str,
    ) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def _export_json(
        self, results: List[Dict[str, Any]], path: str,
    ) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    async def _export_xlsx(
        self, results: List[Dict[str, Any]], path: str,
    ) -> None:
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Evaluation Results"

        if not results:
            wb.save(path)
            return

        # Flatten scores into columns
        flat_rows = []
        for row in results:
            flat: Dict[str, Any] = {
                k: v for k, v in row.items() if k != "scores"
            }
            scores = row.get("scores", {})
            for metric_name, metric_scores in scores.items():
                if isinstance(metric_scores, dict):
                    for sub_key, value in metric_scores.items():
                        if isinstance(value, dict) and "score" in value:
                            flat[f"{metric_name}_{sub_key}"] = value["score"]
                        elif isinstance(value, (int, float)):
                            flat[f"{metric_name}_{sub_key}"] = value
            flat_rows.append(flat)

        # Write header
        headers = list(flat_rows[0].keys())
        ws.append(headers)

        # Write data
        for flat in flat_rows:
            ws.append([flat.get(h, "") for h in headers])

        wb.save(path)
