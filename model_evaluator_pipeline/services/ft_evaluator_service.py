"""Domain knowledge evaluator service — PASS/FAIL verdicts with K-Type taxonomy.

Uses direct AsyncOpenAI client (not agentsoul's BaseLLM) for judge calls
with JSON mode and temperature=0.
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from shared.config import FTEvaluatorConfig

from ..models.ft_evaluator import (
    FailureType,
    FTEvalResult,
    FTEvalSummary,
    FTEvalVerdict,
    KSMILabel,
    Severity,
    SuggestedAction,
    Verdict,
)

_BUNDLED_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "ft_evaluator_prompt.md"


class FTEvaluatorService:
    """MCP-ready service for domain knowledge evaluation of fine-tuned model outputs."""

    def __init__(self, config: Optional[FTEvaluatorConfig] = None):
        self.config = config or FTEvaluatorConfig()

    # ------------------------------------------------------------------ #
    # Prompt loading
    # ------------------------------------------------------------------ #
    def _load_system_prompt(self) -> str:
        """Load the system prompt from custom path or bundled default."""
        path = self.config.system_prompt_path
        if path and Path(path).exists():
            return Path(path).read_text(encoding="utf-8")
        return _BUNDLED_PROMPT_PATH.read_text(encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Client creation
    # ------------------------------------------------------------------ #
    def _create_client(self) -> AsyncOpenAI:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        api_base = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_API_BASE_URL")
        )
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        return AsyncOpenAI(**kwargs)

    # ------------------------------------------------------------------ #
    # Single evaluation
    # ------------------------------------------------------------------ #
    async def evaluate_single(
        self,
        instruction: str,
        generated: str,
        reference: str,
        judge_model: Optional[str] = None,
        ksmi_label: Optional[str] = None,
    ) -> FTEvalVerdict:
        """Run a single judge on one sample. Returns one FTEvalVerdict."""
        judge_model = judge_model or self.config.judge_models[0]
        system_prompt = self._load_system_prompt()

        user_content = (
            f"## Instruction\n{instruction}\n\n"
            f"## Generated Answer\n{generated}\n\n"
            f"## Reference Answer\n{reference}"
        )
        if ksmi_label:
            user_content += f"\n\n## KSMI Label\n{ksmi_label}"

        client = self._create_client()

        try:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)

            return FTEvalVerdict(
                verdict=Verdict(parsed["verdict"]),
                failure_type=FailureType(parsed["failure_type"]) if parsed.get("failure_type") else None,
                severity=Severity(parsed["severity"]) if parsed.get("severity") else None,
                suggested_action=SuggestedAction(parsed["suggested_action"]) if parsed.get("suggested_action") else None,
                reasoning=parsed.get("reasoning", ""),
                judge_model=judge_model,
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return FTEvalVerdict(
                verdict=Verdict.FAIL,
                reasoning=f"Failed to parse judge JSON response for model {judge_model}",
                judge_model=judge_model,
            )

    # ------------------------------------------------------------------ #
    # Multi-judge evaluation
    # ------------------------------------------------------------------ #
    async def evaluate_multi_judge(
        self,
        instruction: str,
        generated: str,
        reference: str,
        ksmi_label: Optional[str] = None,
        judge_models: Optional[List[str]] = None,
    ) -> FTEvalResult:
        """Run N judges in parallel on one sample, compute majority-vote consensus."""
        judge_models = judge_models or self.config.judge_models

        tasks = [
            self.evaluate_single(
                instruction=instruction,
                generated=generated,
                reference=reference,
                judge_model=model,
                ksmi_label=ksmi_label,
            )
            for model in judge_models
        ]

        verdicts = await asyncio.gather(*tasks)

        # Majority vote consensus
        pass_count = sum(1 for v in verdicts if v.verdict == Verdict.PASS)
        consensus = Verdict.PASS if pass_count > len(verdicts) / 2 else Verdict.FAIL

        return FTEvalResult(
            instruction=instruction,
            generated=generated,
            reference=reference,
            ksmi_label=KSMILabel(ksmi_label) if ksmi_label else None,
            verdicts=list(verdicts),
            consensus_verdict=consensus,
        )

    # ------------------------------------------------------------------ #
    # Batch evaluation
    # ------------------------------------------------------------------ #
    async def evaluate_batch(
        self,
        test_data: List[Dict[str, Any]],
        judge_models: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a batch of test samples with multi-judge support."""
        judge_models = judge_models or self.config.judge_models

        results: List[FTEvalResult] = []
        for row in test_data:
            result = await self.evaluate_multi_judge(
                instruction=row.get("instruction", ""),
                generated=row.get("generated", ""),
                reference=row.get("reference", row.get("output", "")),
                ksmi_label=row.get("ksmi_label"),
                judge_models=judge_models,
            )
            results.append(result)

        summary = self.compute_summary(results)

        return {
            "success": True,
            "results": [r.model_dump() for r in results],
            "summary": summary.model_dump(),
            "count": len(results),
        }

    # ------------------------------------------------------------------ #
    # Summary computation
    # ------------------------------------------------------------------ #
    def compute_summary(self, results: List[FTEvalResult]) -> FTEvalSummary:
        """Compute aggregate statistics from evaluation results."""
        total = len(results)
        pass_count = sum(1 for r in results if r.consensus_verdict == Verdict.PASS)
        fail_count = total - pass_count

        # Failure type distribution (from all verdicts)
        failure_types: Counter[str] = Counter()
        severity_counts: Counter[str] = Counter()
        action_counts: Counter[str] = Counter()
        judge_pass: Counter[str] = Counter()
        judge_total: Counter[str] = Counter()

        for result in results:
            for v in result.verdicts:
                judge_total[v.judge_model] += 1
                if v.verdict == Verdict.PASS:
                    judge_pass[v.judge_model] += 1
                if v.failure_type:
                    failure_types[v.failure_type.value] += 1
                if v.severity:
                    severity_counts[v.severity.value] += 1
                if v.suggested_action:
                    action_counts[v.suggested_action.value] += 1

        # Per-KSMI pass rate
        ksmi_pass: Counter[str] = Counter()
        ksmi_total: Counter[str] = Counter()
        for result in results:
            if result.ksmi_label:
                label = result.ksmi_label.value
                ksmi_total[label] += 1
                if result.consensus_verdict == Verdict.PASS:
                    ksmi_pass[label] += 1

        pass_rate_by_ksmi = {
            label: ksmi_pass[label] / count
            for label, count in ksmi_total.items()
        }

        per_judge_pass_rate = {
            model: judge_pass[model] / count
            for model, count in judge_total.items()
        }

        return FTEvalSummary(
            total_samples=total,
            pass_count=pass_count,
            fail_count=fail_count,
            pass_rate=pass_count / total if total > 0 else 0.0,
            failure_type_distribution=dict(failure_types),
            severity_distribution=dict(severity_counts),
            action_distribution=dict(action_counts),
            pass_rate_by_ksmi_label=pass_rate_by_ksmi,
            per_judge_pass_rate=per_judge_pass_rate,
        )

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    async def export_results(
        self,
        results: List[FTEvalResult],
        output_path: str,
        format: str = "jsonl",
    ) -> Dict[str, Any]:
        """Export evaluation results to a file."""
        try:
            data = [r.model_dump() for r in results]

            if format == "jsonl":
                with open(output_path, "w", encoding="utf-8") as f:
                    for row in data:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            elif format == "json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
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
