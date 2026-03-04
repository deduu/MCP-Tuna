"""MCP server for model evaluation, LLM-as-a-judge, and domain knowledge evaluation."""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from agentsoul.server import MCPServer
from shared.config import AdvancedJudgeConfig, FTEvaluatorConfig, ModelEvaluationConfig


class ModelEvalMCPServer:
    """Exposes evaluate_model, judge, and ft_eval tools as a single MCP server."""

    def __init__(
        self,
        eval_config: Optional[ModelEvaluationConfig] = None,
        judge_config: Optional[AdvancedJudgeConfig] = None,
        ft_config: Optional[FTEvaluatorConfig] = None,
    ):
        self._eval_svc = None
        self._judge_svc = None
        self._ft_svc = None
        self._eval_config = eval_config
        self._judge_config = judge_config
        self._ft_config = ft_config
        self.mcp = MCPServer("mcp-tuna-model-eval", "1.0.0")
        self._register_tools()

    # -- lazy service accessors --
    @property
    def eval_svc(self):
        if self._eval_svc is None:
            from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService
            self._eval_svc = ModelEvaluationService(self._eval_config)
        return self._eval_svc

    @property
    def judge_svc(self):
        if self._judge_svc is None:
            from model_evaluator_pipeline.services.judge_service import AdvancedJudgeService
            self._judge_svc = AdvancedJudgeService(self._judge_config)
        return self._judge_svc

    @property
    def ft_svc(self):
        if self._ft_svc is None:
            from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService
            self._ft_svc = FTEvaluatorService(self._ft_config)
        return self._ft_svc

    def _register_tools(self):
        self._register_evaluate_model_tools()
        self._register_judge_tools()
        self._register_ft_eval_tools()

    # ------------------------------------------------------------------ #
    # evaluate_model.*
    # ------------------------------------------------------------------ #
    def _register_evaluate_model_tools(self):
        @self.mcp.tool(
            name="evaluate_model.single",
            description="Score a single generated output against reference with ROUGE, BERTScore, LLM-as-Judge.",
        )
        async def eval_single(
            question: str, generated: str, reference: str,
            metrics: Optional[List[str]] = None,
        ) -> str:
            return json.dumps(
                await self.eval_svc.evaluate_single(question, generated, reference, metrics),
                indent=2,
            )

        @self.mcp.tool(
            name="evaluate_model.batch",
            description="Run evaluation on a test set (optionally with model inference).",
        )
        async def eval_batch(
            test_data: List[Dict], metrics: Optional[List[str]] = None,
            model_path: Optional[str] = None, adapter_path: Optional[str] = None,
            max_new_tokens: int = 1024, flatten: bool = False,
        ) -> str:
            return json.dumps(
                await self.eval_svc.evaluate_batch(
                    test_data, metrics, model_path, adapter_path, max_new_tokens, flatten,
                ),
                indent=2,
            )

        @self.mcp.tool(
            name="evaluate_model.export",
            description="Export evaluation results as JSONL, JSON, or Excel.",
        )
        async def eval_export(
            results: List[Dict], output_path: str, format: str = "jsonl",
        ) -> str:
            return json.dumps(
                await self.eval_svc.export_results(results, output_path, format),
                indent=2,
            )

        @self.mcp.tool(
            name="evaluate_model.summary",
            description="Compute aggregate statistics for evaluation results.",
        )
        async def eval_summary(results: List[Dict]) -> str:
            return json.dumps(self.eval_svc.compute_summary(results), indent=2)

    # ------------------------------------------------------------------ #
    # judge.*
    # ------------------------------------------------------------------ #
    def _register_judge_tools(self):
        @self.mcp.tool(
            name="judge.evaluate",
            description="Run single LLM-as-a-judge evaluation with custom criteria/rubric.",
        )
        async def judge_evaluate(
            question: str, generated: str,
            reference: Optional[str] = None, generated_b: Optional[str] = None,
            judge_type: str = "pointwise", judge_model: str = "gpt-4o",
            criteria: Optional[List[Dict]] = None, rubric: Optional[Dict] = None,
        ) -> str:
            return json.dumps(
                await self.judge_svc.evaluate_single(
                    question, generated, reference, generated_b,
                    judge_type, judge_model, criteria, rubric,
                ),
                indent=2,
            )

        @self.mcp.tool(
            name="judge.evaluate_multi",
            description="Run multiple LLM judges in parallel and aggregate scores.",
        )
        async def judge_multi(
            question: str, generated: str,
            reference: Optional[str] = None, generated_b: Optional[str] = None,
            judge_type: str = "pointwise",
            judges: Optional[List[Dict]] = None,
            criteria: Optional[List[Dict]] = None, rubric: Optional[Dict] = None,
            aggregation: str = "mean",
        ) -> str:
            return json.dumps(
                await self.judge_svc.evaluate_multi_judge(
                    question, generated, reference, generated_b,
                    judge_type, judges, criteria, rubric, aggregation,
                ),
                indent=2,
            )

        @self.mcp.tool(
            name="judge.evaluate_batch",
            description="Batch evaluation with multi-judge support and custom criteria.",
        )
        async def judge_batch(
            test_data: List[Dict], judge_type: str = "pointwise",
            judges: Optional[List[Dict]] = None,
            criteria: Optional[List[Dict]] = None, rubric: Optional[Dict] = None,
            aggregation: str = "mean",
        ) -> str:
            return json.dumps(
                await self.judge_svc.evaluate_batch(
                    test_data, judge_type, judges, criteria, rubric, aggregation,
                ),
                indent=2,
            )

        @self.mcp.tool(
            name="judge.compare_pair",
            description="Pairwise comparison: which of two outputs is better?",
        )
        async def judge_pair(
            question: str, generated_a: str, generated_b: str,
            reference: Optional[str] = None,
            judges: Optional[List[Dict]] = None,
            criteria: Optional[List[Dict]] = None,
        ) -> str:
            return json.dumps(
                await self.judge_svc.evaluate_multi_judge(
                    question=question, generated=generated_a,
                    reference=reference, generated_b=generated_b,
                    judge_type="pairwise", judges=judges, criteria=criteria,
                ),
                indent=2,
            )

        @self.mcp.tool(name="judge.list_types", description="List available judge types.")
        async def judge_list_types() -> str:
            return json.dumps(self.judge_svc.list_judge_types(), indent=2)

        @self.mcp.tool(name="judge.export", description="Export judge results as JSONL or JSON.")
        async def judge_export(
            results: List[Dict], output_path: str, format: str = "jsonl",
        ) -> str:
            return json.dumps(
                await self.judge_svc.export_results(results, output_path, format),
                indent=2,
            )

        @self.mcp.tool(
            name="judge.create_rubric",
            description="Validate a rubric definition and return parsed result.",
        )
        async def judge_rubric(
            name: str, criteria: List[Dict], description: str = "",
        ) -> str:
            try:
                from model_evaluator_pipeline.models.judge_models import JudgeRubric, JudgeCriterion
                rubric = JudgeRubric(
                    name=name, description=description,
                    criteria=[JudgeCriterion(**c) for c in criteria],
                )
                return json.dumps({"success": True, "rubric": rubric.model_dump()}, indent=2)
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)}, indent=2)

    # ------------------------------------------------------------------ #
    # ft_eval.*
    # ------------------------------------------------------------------ #
    def _register_ft_eval_tools(self):
        @self.mcp.tool(
            name="ft_eval.single",
            description="Domain knowledge PASS/FAIL evaluation on one sample.",
        )
        async def ft_single(
            instruction: str, generated: str, reference: str,
            judge_model: Optional[str] = None, ksmi_label: Optional[str] = None,
        ) -> str:
            result = await self.ft_svc.evaluate_single(
                instruction, generated, reference, judge_model, ksmi_label,
            )
            return json.dumps(result.model_dump() if hasattr(result, "model_dump") else result, indent=2)

        @self.mcp.tool(
            name="ft_eval.batch",
            description="Batch domain knowledge evaluation with multi-judge + KSMI labels.",
        )
        async def ft_batch(
            test_data: List[Dict], judge_models: Optional[List[str]] = None,
        ) -> str:
            return json.dumps(
                await self.ft_svc.evaluate_batch(test_data, judge_models), indent=2,
            )

        @self.mcp.tool(
            name="ft_eval.summary",
            description="Compute stakeholder summary (pass rate, failure types, severity).",
        )
        async def ft_summary(results: List[Dict]) -> str:
            parsed = []
            for r in results:
                from model_evaluator_pipeline.models.ft_eval_models import FTEvalResult
                parsed.append(FTEvalResult(**r) if isinstance(r, dict) else r)
            summary = self.ft_svc.compute_summary(parsed)
            return json.dumps(summary.model_dump() if hasattr(summary, "model_dump") else summary, indent=2)

        @self.mcp.tool(
            name="ft_eval.export",
            description="Export domain knowledge evaluation results as JSONL or JSON.",
        )
        async def ft_export(
            results: List[Dict], output_path: str, format: str = "jsonl",
        ) -> str:
            parsed = []
            for r in results:
                from model_evaluator_pipeline.models.ft_eval_models import FTEvalResult
                parsed.append(FTEvalResult(**r) if isinstance(r, dict) else r)
            return json.dumps(
                await self.ft_svc.export_results(parsed, output_path, format),
                indent=2,
            )

    def run(self, transport=None):
        self.mcp.run(transport)
