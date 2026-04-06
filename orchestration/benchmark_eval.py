from __future__ import annotations

from typing import Any, Dict, List, Optional

from .benchmarking import score_benchmark_case, summarize_pack_scores


def _average_metric(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float))
    ]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


async def evaluate_candidate_with_finetuner(
    *,
    finetuner: Any,
    model_path: str,
    adapter_path: Optional[str],
    evaluation_packs: Dict[str, List[Dict[str, Any]]],
    eval_system_prompt: Optional[str],
    eval_quantization: Optional[str],
    eval_max_new_tokens: int,
) -> Dict[str, Any]:
    evaluations: Dict[str, Any] = {}
    for pack_name, cases in evaluation_packs.items():
        grouped_cases: Dict[Optional[str], List[Dict[str, Any]]] = {}
        for case in cases:
            grouped_cases.setdefault(
                case.get("system_prompt") or eval_system_prompt,
                [],
            ).append(case)

        pack_scores: List[Dict[str, Any]] = []
        generation_metrics: List[Dict[str, Any]] = []
        for system_prompt, grouped in grouped_cases.items():
            prompts = [case["prompt"] for case in grouped]
            inference_result = await finetuner.run_inference(
                prompts=prompts,
                model_path=model_path,
                adapter_path=adapter_path,
                max_new_tokens=eval_max_new_tokens,
                do_sample=False,
                system_prompt=system_prompt,
                quantization=eval_quantization,
            )
            if not inference_result.get("success"):
                return {
                    "success": False,
                    "error": inference_result.get("error") or "Inference failed",
                    "failed_pack": pack_name,
                }

            for case, raw_result in zip(grouped, inference_result.get("results") or []):
                pack_scores.append(
                    score_benchmark_case(
                        case,
                        str(raw_result.get("response") or ""),
                    )
                )
                generation_metrics.append(
                    {
                        "generation_time_seconds": raw_result.get("generation_time_seconds"),
                        "tokens_generated": raw_result.get("tokens_generated"),
                        "tokens_per_second": raw_result.get("tokens_per_second"),
                    }
                )

        pack_summary = summarize_pack_scores(pack_scores)
        pack_summary["avg_generation_time_seconds"] = _average_metric(
            generation_metrics,
            "generation_time_seconds",
        )
        pack_summary["avg_tokens_per_second"] = _average_metric(
            generation_metrics,
            "tokens_per_second",
        )
        pack_summary["avg_tokens_generated"] = _average_metric(
            generation_metrics,
            "tokens_generated",
        )
        evaluations[pack_name] = {
            "cases": pack_scores,
            "summary": pack_summary,
        }

    return {"success": True, "packs": evaluations}
