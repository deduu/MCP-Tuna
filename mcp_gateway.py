"""
Transcendence Unified MCP Gateway
====================================

Single entry point exposing all pipeline operations as MCP tools.
Uses agentsoul's MCPServer (production-grade HTTP+stdio transport).

84 tools across 16 namespaces:
  system, extract, generate, clean, normalize, evaluate, evaluate_model,
  dataset, finetune, test, validate, host, workflow, orchestration,
  judge, ft_eval
"""
from __future__ import annotations

import inspect
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from agentsoul.server import MCPServer
from dotenv import load_dotenv
from shared.config import (
    AdvancedJudgeConfig,
    ChatConfig,
    FTEvaluatorConfig,
    GeneratorConfig,
    CleaningConfig,
    NormalizationConfig,
    EvaluatorConfig,
    HostingConfig,
    OrchestrationConfig,
    ModelEvaluationConfig,
)


class TranscendenceGateway:
    """Unified MCP gateway that composes all Transcendence pipeline services."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        load_dotenv(override=False)
        config = config or {}
        self.mcp = MCPServer("transcendence-gateway", "1.0.0")

        # Lazily-initialized services (avoid heavy imports at gateway startup)
        self._generator_svc = None
        self._cleaning_svc = None
        self._normalization_svc = None
        self._evaluator_svc = None
        self._model_evaluator_svc = None
        self._finetuning_svc = None
        self._hosting_svc = None
        self._orchestrator = None
        self._orchestration_data_svc = None
        self._advanced_judge_svc = None
        self._ft_evaluator_svc = None
        self._job_manager_instance = None
        self._dataset_svc = None
        self._chat_sessions: Dict[str, Any] = {}

        self._config = config
        self._register_all_tools()
        self._wrap_tools_with_diagnostics()

    # ------------------------------------------------------------------ #
    # Lazy service accessors
    # ------------------------------------------------------------------ #
    @property
    def generator(self):
        if self._generator_svc is None:
            try:
                from data_generator_pipeline.services.pipeline_service import PipelineService
            except ImportError:
                raise ImportError(
                    "Data generation tools require: pip install transcendence[data]"
                ) from None
            from shared.provider_factory import create_llm
            gen_config = GeneratorConfig(**{
                k: v for k, v in self._config.get("generator", {}).items()
                if k in GeneratorConfig.model_fields
            })
            llm = create_llm(gen_config)
            self._generator_svc = PipelineService(llm, gen_config)
        return self._generator_svc

    @property
    def cleaner(self):
        if self._cleaning_svc is None:
            from data_cleaning_pipeline.services.cleaning_service import DataCleaningService
            self._cleaning_svc = DataCleaningService()
        return self._cleaning_svc

    @property
    def normalizer(self):
        if self._normalization_svc is None:
            from data_normalization_pipeline.services.normalization_service import DataNormalizationService
            self._normalization_svc = DataNormalizationService()
        return self._normalization_svc

    @property
    def evaluator(self):
        if self._evaluator_svc is None:
            try:
                from data_evaluator_pipeline.services.pipeline_service import EvaluatorService
            except ImportError:
                raise ImportError(
                    "Evaluation tools require: pip install transcendence[eval]"
                ) from None
            eval_config = EvaluatorConfig(**self._config.get("evaluator", {}))
            self._evaluator_svc = EvaluatorService(eval_config)
        return self._evaluator_svc

    @property
    def model_evaluator(self):
        if self._model_evaluator_svc is None:
            try:
                from model_evaluator_pipeline.services.evaluation_service import ModelEvaluationService
            except ImportError:
                raise ImportError(
                    "Model evaluation tools require: pip install transcendence[model-eval]"
                ) from None
            eval_config = ModelEvaluationConfig(**{
                k: v for k, v in self._config.get("model_evaluator", {}).items()
                if k in ModelEvaluationConfig.model_fields
            })
            self._model_evaluator_svc = ModelEvaluationService(eval_config)
        return self._model_evaluator_svc

    @property
    def finetuner(self):
        if self._finetuning_svc is None:
            try:
                from finetuning_pipeline.services.pipeline_service import FineTuningService
            except ImportError:
                raise ImportError(
                    "Training tools require: pip install transcendence[training]"
                ) from None
            self._finetuning_svc = FineTuningService(
                default_base_model=self._config.get("finetuning", {}).get(
                    "base_model", "meta-llama/Llama-3.2-3B-Instruct"
                )
            )
        return self._finetuning_svc

    @property
    def hoster(self):
        if self._hosting_svc is None:
            try:
                from hosting_pipeline.services.hosting_service import HostingService
            except ImportError:
                raise ImportError(
                    "Hosting tools require: pip install transcendence[hosting]"
                ) from None
            self._hosting_svc = HostingService()
        return self._hosting_svc

    @property
    def orchestration_data_service(self):
        if self._orchestration_data_svc is None:
            try:
                from orchestration.orchestration_trainer import OrchestrationDataService
            except ImportError:
                raise ImportError(
                    "Orchestration tools require: pip install transcendence[orchestration]"
                ) from None
            from orchestration.rewards import OrchestrationRewardFunction
            from shared.provider_factory import create_llm
            orch_config = OrchestrationConfig(**{
                k: v for k, v in self._config.get("orchestration", {}).items()
                if k in OrchestrationConfig.model_fields
            })
            llm = create_llm(orch_config)
            reward_fn = OrchestrationRewardFunction(
                llm, weights=orch_config.reward_weights,
            )
            self._orchestration_data_svc = OrchestrationDataService(llm, reward_fn)
        return self._orchestration_data_svc

    @property
    def orchestrator(self):
        if self._orchestrator is None:
            from orchestration.workflow import PipelineOrchestrator
            self._orchestrator = PipelineOrchestrator(
                generator=self.generator,
                cleaner=self.cleaner,
                normalizer=self.normalizer,
                evaluator=self.evaluator,
                finetuner=self.finetuner,
                hoster=self.hoster,
                orchestration_data_service=self.orchestration_data_service,
            )
        return self._orchestrator

    @property
    def advanced_judge(self):
        if self._advanced_judge_svc is None:
            try:
                from model_evaluator_pipeline.services.judge_service import AdvancedJudgeService
            except ImportError:
                raise ImportError(
                    "Judge tools require: pip install transcendence[model-eval]"
                ) from None
            judge_config = AdvancedJudgeConfig(**{
                k: v for k, v in self._config.get("advanced_judge", {}).items()
                if k in AdvancedJudgeConfig.model_fields
            })
            self._advanced_judge_svc = AdvancedJudgeService(judge_config)
        return self._advanced_judge_svc

    @property
    def ft_evaluator(self):
        if self._ft_evaluator_svc is None:
            try:
                from model_evaluator_pipeline.services.ft_evaluator_service import FTEvaluatorService
            except ImportError:
                raise ImportError(
                    "Fine-tune evaluation tools require: pip install transcendence[model-eval]"
                ) from None
            ft_config = FTEvaluatorConfig(**{
                k: v for k, v in self._config.get("ft_evaluator", {}).items()
                if k in FTEvaluatorConfig.model_fields
            })
            self._ft_evaluator_svc = FTEvaluatorService(ft_config)
        return self._ft_evaluator_svc

    @property
    def job_manager(self):
        if self._job_manager_instance is None:
            from shared.training_jobs import TrainingJobManager
            self._job_manager_instance = TrainingJobManager(max_concurrent=1)
        return self._job_manager_instance

    @property
    def dataset_service(self):
        if self._dataset_svc is None:
            from shared.dataset_service import DatasetService
            self._dataset_svc = DatasetService()
        return self._dataset_svc

    # ------------------------------------------------------------------ #
    # Auto-deploy helper
    # ------------------------------------------------------------------ #
    async def _auto_deploy_if_requested(
        self,
        train_result: Dict[str, Any],
        deploy: bool,
        deploy_port: int,
        base_model: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Conditionally deploy a trained model after successful training.

        Returns the deployment result dict if deployed, None otherwise.
        """
        if not deploy or not train_result.get("success"):
            return None

        model_path = train_result.get("model_path") or train_result.get(
            "final_model_path"
        )
        if not model_path:
            return None

        resolved_base = base_model or self.finetuner.config.base_model
        config = HostingConfig(
            model_path=resolved_base,
            adapter_path=model_path,
            port=deploy_port,
        )
        result = await self.hoster.deploy_as_mcp(config)
        if result.get("success"):
            endpoint = result.get("endpoint", "")
            result["chat_command"] = (
                f"python scripts/chat_cli.py --endpoint {endpoint}"
            )
        return result

    # ------------------------------------------------------------------ #
    # Tool registration
    # ------------------------------------------------------------------ #
    def _register_all_tools(self):
        self._register_system_tools()
        self._register_extract_tools()
        self._register_generate_tools()
        self._register_clean_tools()
        self._register_normalize_tools()
        self._register_evaluate_tools()
        self._register_model_eval_tools()
        self._register_finetune_tools()
        self._register_training_monitor_tools()
        self._register_test_tools()
        self._register_validate_tools()
        self._register_host_tools()
        self._register_workflow_tools()
        self._register_orchestration_tools()
        self._register_judge_tools()
        self._register_ft_evaluator_tools()
        self._register_dataset_tools()

    # -- System (Resource Checking) --
    def _register_system_tools(self):
        @self.mcp.tool(
            name="system.check_resources",
            description=(
                "Check GPU/RAM/disk status before training. Call this before any "
                "finetune.train* or workflow.* tool to verify hardware is sufficient. "
                "Returns gpu (name, vram_total_gb, vram_free_gb, compute_capability), "
                "ram (total_gb, free_gb, percent_used), and disk (free_gb) info."
            ),
        )
        async def check_resources() -> str:
            result = self.finetuner.check_resources()
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.preflight_check",
            description=(
                "Estimate VRAM requirements for a training configuration and check if "
                "it will fit on the available GPU. Returns can_run (bool), "
                "estimated_vram_gb, available_vram_gb, headroom_gb, a detailed "
                "breakdown (model_gb, lora_gb, optimizer_gb, activations_gb, "
                "overhead_gb), recommendations for optimal settings, and warnings. "
                "Call this before finetune.train* to avoid OOM errors."
            ),
        )
        async def preflight_check(
            model_name: str,
            quantization: str = "4bit",
            batch_size: int = 1,
            max_seq_length: int = 512,
            technique: str = "sft",
            use_lora: bool = True,
            lora_r: int = 8,
            gradient_checkpointing: bool = False,
        ) -> str:
            result = self.finetuner.preflight_check(
                model_name=model_name,
                quantization=quantization,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                technique=technique,
                use_lora=use_lora,
                lora_r=lora_r,
                gradient_checkpointing=gradient_checkpointing,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="system.setup_check",
            description=(
                "Validate all prerequisites for using Transcendence: API keys, GPU, "
                "HuggingFace token, disk space, required Python packages. "
                "Run this first before any pipeline operation."
            ),
        )
        async def setup_check() -> str:
            import os
            import shutil
            checks = []

            # OpenAI API key
            has_key = bool(os.getenv("OPENAI_API_KEY"))
            checks.append({
                "name": "OPENAI_API_KEY", "status": "pass" if has_key else "warn",
                "detail": "Set" if has_key else "Not set — generation/evaluation tools require this",
            })

            # HuggingFace token
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            checks.append({
                "name": "HF_TOKEN", "status": "pass" if hf_token else "warn",
                "detail": "Set" if hf_token else "Not set — required for gated models and push_to_hub",
            })

            # GPU
            try:
                import torch
                gpu_ok = torch.cuda.is_available()
                gpu_name = torch.cuda.get_device_name(0) if gpu_ok else None
                checks.append({
                    "name": "GPU", "status": "pass" if gpu_ok else "warn",
                    "detail": gpu_name or "No GPU detected — training will be slow",
                })
            except ImportError:
                checks.append({"name": "GPU", "status": "fail", "detail": "torch not installed"})

            # Disk space
            disk = shutil.disk_usage(".")
            free_gb = round(disk.free / (1024 ** 3), 1)
            checks.append({
                "name": "Disk Space", "status": "pass" if free_gb > 10 else "warn",
                "detail": f"{free_gb} GB free",
            })

            # Key packages
            for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
                try:
                    __import__(pkg)
                    checks.append({"name": f"Package: {pkg}", "status": "pass", "detail": "Installed"})
                except ImportError:
                    checks.append({"name": f"Package: {pkg}", "status": "fail", "detail": "Not installed"})

            all_passed = all(c["status"] != "fail" for c in checks)
            return json.dumps({"success": True, "checks": checks, "all_passed": all_passed}, indent=2)

        @self.mcp.tool(
            name="system.config",
            description="Show current gateway configuration (model defaults, output dirs, env vars).",
        )
        async def show_config() -> str:
            import os
            return json.dumps({
                "success": True,
                "config": self._config,
                "env": {
                    "OPENAI_API_KEY": "***" if os.getenv("OPENAI_API_KEY") else None,
                    "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
                    "HF_TOKEN": "***" if os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") else None,
                },
            }, indent=2)

    # -- Extract --
    def _register_extract_tools(self):
        @self.mcp.tool(name="extract.load_document",
                       description="Load and parse a document file (PDF, Markdown, etc.)")
        async def load_document(file_path: str) -> str:
            # Avoid initializing the generator LLM provider for pure document loading.
            from data_generator_pipeline.loaders import get_loader

            try:
                loader = get_loader(file_path)
                file_name, pages = loader.load(file_path)
                result = {
                    "success": True,
                    "file_name": file_name,
                    "file_path": file_path,
                    "pages": pages,
                    "total_pages": len(pages),
                }
            except Exception as e:
                result = {"success": False, "error": str(e), "file_path": file_path}

            return json.dumps(result, indent=2)

    # -- Generate --
    def _register_generate_tools(self):
        @self.mcp.tool(name="generate.from_document",
                       description="Generate fine-tuning data from an entire document")
        async def gen_from_doc(
            technique: str, file_path: str,
            custom_template: Optional[str] = None,
            start_page: Optional[int] = None, end_page: Optional[int] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_from_document(
                technique=technique, file_path=file_path,
                custom_template=custom_template,
                start_page=start_page, end_page=end_page,
            ), indent=2)

        @self.mcp.tool(name="generate.from_page",
                       description="Generate fine-tuning data from a single page")
        async def gen_from_page(
            technique: str, page_text: str, page_index: int, file_name: str,
            custom_template: Optional[str] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_from_page(
                technique=technique, page_text=page_text,
                page_index=page_index, file_name=file_name,
                custom_template=custom_template,
            ), indent=2)

        @self.mcp.tool(name="generate.batch",
                       description="Generate fine-tuning data from multiple documents")
        async def gen_batch(
            technique: str, file_paths: List[str],
            custom_template: Optional[str] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_batch(
                technique=technique, file_paths=file_paths,
                custom_template=custom_template,
            ), indent=2)

        @self.mcp.tool(name="generate.list_techniques",
                       description="List available fine-tuning techniques")
        async def list_techniques() -> str:
            from data_generator_pipeline.generators.registry import GENERATOR_REGISTRY
            return json.dumps({
                "success": True,
                "techniques": list(GENERATOR_REGISTRY.keys()),
            }, indent=2)

        @self.mcp.tool(name="generate.get_schema",
                       description="Get the data schema for a fine-tuning technique")
        async def get_schema(technique: str) -> str:
            from data_generator_pipeline.generators.registry import GENERATOR_REGISTRY
            import dataclasses
            if technique not in GENERATOR_REGISTRY:
                return json.dumps({"success": False, "error": f"Unknown technique: {technique}"}, indent=2)
            _, datapoint_class = GENERATOR_REGISTRY[technique]
            schema = {
                field.name: {
                    "type": str(field.type),
                    "required": field.default == dataclasses.MISSING,
                }
                for field in dataclasses.fields(datapoint_class)
            }
            return json.dumps({"success": True, "technique": technique, "schema": schema}, indent=2)

        @self.mcp.tool(
            name="generate.from_text",
            description=(
                "Generate fine-tuning data from raw text (no file required). "
                "Useful when text is already in memory or pasted by the user. "
                "Supports all techniques: sft, dpo, grpo, kto."
            ),
        )
        async def gen_from_text(
            technique: str,
            text: str,
            source_name: str = "raw_text",
            custom_template: Optional[str] = None,
        ) -> str:
            return json.dumps(await self.generator.generate_from_page(
                technique=technique, page_text=text,
                page_index=0, file_name=source_name,
                custom_template=custom_template,
            ), indent=2)

        @self.mcp.tool(
            name="generate.from_hf_dataset",
            description=(
                "Load a dataset from the HuggingFace Hub and return as Transcendence "
                "data_points. Automatically maps columns if the dataset uses "
                "standard naming (instruction/input/output or prompt/chosen/rejected). "
                "Use column_mapping JSON to override: e.g., "
                '\'{"question": "instruction", "answer": "output"}\'. '
                "Returns data_points ready for clean.dataset, evaluate.dataset, "
                "or dataset.save."
            ),
        )
        async def gen_from_hf_dataset(
            dataset_name: str,
            split: str = "train",
            subset: Optional[str] = None,
            max_rows: Optional[int] = None,
            column_mapping: Optional[str] = None,
        ) -> str:
            try:
                from datasets import load_dataset as hf_load_dataset
                ds = hf_load_dataset(dataset_name, subset, split=split)
                if max_rows:
                    ds = ds.select(range(min(max_rows, len(ds))))

                mapping = json.loads(column_mapping) if column_mapping else {}
                data_points = []
                for row in ds:
                    dp = {}
                    for src_col, val in row.items():
                        target_col = mapping.get(src_col, src_col)
                        if isinstance(val, (str, int, float, bool, list)):
                            dp[target_col] = val
                        else:
                            dp[target_col] = str(val)
                    data_points.append(dp)

                return json.dumps({
                    "success": True,
                    "dataset_name": dataset_name,
                    "split": split,
                    "data_points": data_points,
                    "count": len(data_points),
                    "original_columns": list(ds.column_names),
                }, indent=2)
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)}, indent=2)

    # -- Clean --
    def _register_clean_tools(self):
        @self.mcp.tool(name="clean.dataset",
                       description="Run all cleaning steps on a dataset")
        async def clean_dataset(
            data_points: List[Dict],
            remove_duplicates: bool = True,
            min_instruction_length: int = 10,
            min_output_length: int = 20,
        ) -> str:
            config = CleaningConfig(
                remove_duplicates=remove_duplicates,
                min_instruction_length=min_instruction_length,
                min_output_length=min_output_length,
            )
            return json.dumps(await self.cleaner.clean_dataset(data_points, config), indent=2)

        @self.mcp.tool(name="clean.deduplicate",
                       description="Remove duplicate entries by key")
        async def deduplicate(data_points: List[Dict], key: str = "instruction") -> str:
            return json.dumps(await self.cleaner.deduplicate(data_points, key), indent=2)

        @self.mcp.tool(name="clean.validate_schema",
                       description="Validate entries have required fields for a technique")
        async def validate_schema(data_points: List[Dict], technique: str = "sft") -> str:
            return json.dumps(await self.cleaner.validate_schema(data_points, technique), indent=2)

        @self.mcp.tool(name="clean.remove_short",
                       description="Filter entries below length thresholds")
        async def remove_short(
            data_points: List[Dict], min_instruction: int = 10, min_output: int = 20,
        ) -> str:
            return json.dumps(await self.cleaner.remove_short_entries(
                data_points, min_instruction, min_output,
            ), indent=2)

    # -- Normalize --
    def _register_normalize_tools(self):
        @self.mcp.tool(name="normalize.dataset",
                       description="Apply all normalization steps to a dataset")
        async def normalize_dataset(
            data_points: List[Dict],
            target_format: str = "sft",
            merge_instruction_input: bool = True,
        ) -> str:
            config = NormalizationConfig(
                target_format=target_format,
                merge_instruction_input=merge_instruction_input,
            )
            return json.dumps(await self.normalizer.normalize_dataset(data_points, config), indent=2)

        @self.mcp.tool(name="normalize.merge_fields",
                       description="Merge instruction + input into a single field")
        async def merge_fields(data_points: List[Dict]) -> str:
            return json.dumps(await self.normalizer.merge_instruction_input(data_points), indent=2)

        @self.mcp.tool(name="normalize.standardize_keys",
                       description="Rename keys to match target format")
        async def standardize_keys(data_points: List[Dict], target_format: str = "sft") -> str:
            return json.dumps(await self.normalizer.standardize_keys(data_points, target_format), indent=2)

        @self.mcp.tool(name="normalize.strip_text",
                       description="Strip whitespace and normalize unicode")
        async def strip_text(data_points: List[Dict]) -> str:
            return json.dumps(await self.normalizer.strip_and_clean_text(data_points), indent=2)

    # -- Evaluate --
    def _register_evaluate_tools(self):
        @self.mcp.tool(name="evaluate.dataset",
                       description="Score dataset with complexity, IFD, and quality metrics")
        async def evaluate_dataset(data_points: List[Dict]) -> str:
            return json.dumps(await self.evaluator.evaluate_dataset(data_points), indent=2)

        @self.mcp.tool(name="evaluate.filter_by_quality",
                       description="Return entries above quality threshold")
        async def filter_quality(data_points: List[Dict], threshold: float = 0.7) -> str:
            return json.dumps(await self.evaluator.filter_by_quality(data_points, threshold), indent=2)

        @self.mcp.tool(name="evaluate.statistics",
                       description="Return per-metric statistics (min/max/mean/stdev)")
        async def statistics(data_points: List[Dict]) -> str:
            return json.dumps(await self.evaluator.analyze_statistics(data_points), indent=2)

        @self.mcp.tool(name="evaluate.list_metrics",
                       description="List all registered evaluation metrics")
        async def list_metrics() -> str:
            return json.dumps(self.evaluator.list_metrics(), indent=2)

    # -- Model Evaluate --
    def _register_model_eval_tools(self):
        @self.mcp.tool(
            name="evaluate_model.single",
            description="Score a single generated output against a reference with ROUGE, BERTScore, and LLM-as-Judge",
        )
        async def eval_model_single(
            question: str,
            generated: str,
            reference: str,
            metrics: Optional[List[str]] = None,
        ) -> str:
            result = await self.model_evaluator.evaluate_single(
                question=question,
                generated=generated,
                reference=reference,
                metrics=metrics,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="evaluate_model.batch",
            description="Run evaluation on a test set -- optionally runs inference first if model_path is provided",
        )
        async def eval_model_batch(
            test_data: List[Dict],
            metrics: Optional[List[str]] = None,
            model_path: Optional[str] = None,
            adapter_path: Optional[str] = None,
            max_new_tokens: int = 1024,
            flatten: bool = False,
        ) -> str:
            result = await self.model_evaluator.evaluate_batch(
                test_data=test_data,
                metrics=metrics,
                model_path=model_path,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
                flatten=flatten,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="evaluate_model.export",
            description="Export evaluation results as JSONL, JSON, or Excel",
        )
        async def eval_model_export(
            results: List[Dict],
            output_path: str,
            format: str = "jsonl",
        ) -> str:
            result = await self.model_evaluator.export_results(
                results=results,
                output_path=output_path,
                format=format,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="evaluate_model.summary",
            description="Compute aggregate statistics (min/max/mean/stdev) for evaluation results",
        )
        async def eval_model_summary(results: List[Dict]) -> str:
            summary = self.model_evaluator.compute_summary(results)
            return json.dumps({"success": True, "summary": summary}, indent=2)

    # -- Finetune --
    def _register_finetune_tools(self):
        @self.mcp.tool(
            name="finetune.train",
            description=(
                "Fine-tune a model using a dataset file (SFT with QLoRA). "
                "Returns model_path usable as base_model in finetune.train_dpo, "
                "finetune.train_grpo, finetune.train_kto, or finetune.train_curriculum. "
                "Set deploy=True to auto-deploy after training. "
                "After deployment, use host.chat to let the user chat with the model, "
                "or share the chat_command from the deployment result for CLI access."
            ),
        )
        async def train(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
            completion_only_loss: bool = True,
            early_stopping_patience: Optional[int] = None,
            eval_file_path: Optional[str] = None,
            push_to_hub: Optional[str] = None,
            lr_scheduler_type: str = "linear",
            warmup_ratio: float = 0.0,
            weight_decay: float = 0.0,
            max_grad_norm: float = 1.0,
            learning_rate: float = 2e-4,
            report_to: Optional[str] = None,
            max_seq_length: int = 2048,
            per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 4,
            gradient_checkpointing: bool = False,
            optim: str = "adamw_torch",
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_model(
                dataset=load_result["dataset_object"],
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                completion_only_loss=completion_only_loss,
                early_stopping_patience=early_stopping_patience,
                eval_file_path=eval_file_path,
                push_to_hub=push_to_hub,
                lr_scheduler_type=lr_scheduler_type,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                max_grad_norm=max_grad_norm,
                learning_rate=learning_rate,
                report_to=report_to or [],
                max_seq_length=max_seq_length,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                optim=optim,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.train_dpo",
            description=(
                "Fine-tune a model with DPO (Direct Preference Optimization) -- "
                "dataset needs prompt/chosen/rejected columns. "
                "Accepts base_model from a previous finetune.train result (model_path). "
                "Set deploy=True to auto-deploy; then use host.chat for interactive chat."
            ),
        )
        async def train_dpo(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            beta: float = 0.1,
            use_lora: bool = True,
            lora_r: int = 8,
            resume_from_checkpoint: Optional[str] = None,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_dpo_model(
                dataset=load_result["dataset_object"],
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                beta=beta,
                use_lora=use_lora,
                lora_r=lora_r,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.train_grpo",
            description=(
                "Fine-tune a model with GRPO (Group Relative Policy Optimization) -- "
                "dataset needs prompt/responses/rewards columns. "
                "Accepts base_model from a previous finetune.train result (model_path). "
                "Set deploy=True to auto-deploy; then use host.chat for interactive chat."
            ),
        )
        async def train_grpo(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            num_generations: int = 4,
            resume_from_checkpoint: Optional[str] = None,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_grpo_model(
                dataset=load_result["dataset_object"],
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                num_generations=num_generations,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.train_kto",
            description=(
                "Fine-tune a model with KTO (Kahneman-Tversky Optimization) -- "
                "dataset needs prompt/completion/label columns. "
                "Accepts base_model from a previous finetune.train result (model_path). "
                "Set deploy=True to auto-deploy; then use host.chat for interactive chat."
            ),
        )
        async def train_kto(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            beta: float = 0.1,
            use_lora: bool = True,
            lora_r: int = 8,
            resume_from_checkpoint: Optional[str] = None,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_kto_model(
                dataset=load_result["dataset_object"],
                output_dir=output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
                beta=beta,
                use_lora=use_lora,
                lora_r=lora_r,
                resume_from_checkpoint=resume_from_checkpoint,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.train_curriculum",
            description=(
                "Curriculum fine-tune: auto-scores dataset by difficulty, trains easy-to-hard "
                "in stages. Each stage's output model feeds into the next stage automatically. "
                "Set deploy=True to auto-deploy; then use host.chat for interactive chat."
            ),
        )
        async def train_curriculum(
            dataset_path: str,
            output_dir: str,
            base_model: Optional[str] = None,
            num_stages: int = 3,
            num_epochs_per_stage: int = 1,
            difficulty_order: str = "easy_first",
            score_column: str = "weighted_score",
            use_lora: bool = True,
            lora_r: int = 8,
            max_seq_length: int = 2048,
            per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 4,
            gradient_checkpointing: bool = False,
            optim: str = "adamw_torch",
            learning_rate: float = 2e-4,
            lr_scheduler_type: str = "linear",
            warmup_ratio: float = 0.0,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_curriculum_model(
                dataset=load_result["dataset_object"],
                output_dir=output_dir,
                base_model=base_model,
                num_stages=num_stages,
                num_epochs_per_stage=num_epochs_per_stage,
                difficulty_order=difficulty_order,
                score_column=score_column,
                use_lora=use_lora,
                lora_r=lora_r,
                max_seq_length=max_seq_length,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                gradient_checkpointing=gradient_checkpointing,
                optim=optim,
                learning_rate=learning_rate,
                lr_scheduler_type=lr_scheduler_type,
                warmup_ratio=warmup_ratio,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.sequential_train",
            description=(
                "Chain multiple training techniques sequentially (e.g., SFT -> DPO -> GRPO). "
                "Each stage's output model_path automatically becomes the next stage's base_model. "
                "Accepts a JSON list of stages, each with technique, dataset_path, and optional "
                "hyperparameters. Returns per-stage results and final_model_path for deployment "
                "via host.deploy_mcp or evaluation via evaluate_model.batch."
            ),
        )
        async def sequential_train(
            stages: str,
            output_dir: str,
            base_model: Optional[str] = None,
            merge_between_stages: bool = True,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            parsed_stages = json.loads(stages) if isinstance(stages, str) else stages
            result = await self.finetuner.train_sequential(
                stages=parsed_stages,
                output_dir=output_dir,
                base_model=base_model,
                merge_between_stages=merge_between_stages,
            )
            deploy_result = await self._auto_deploy_if_requested(
                result, deploy, deploy_port, base_model
            )
            if deploy_result is not None:
                result["deployment"] = deploy_result
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="finetune.load_dataset",
                       description="Load a dataset from file for fine-tuning")
        async def load_dataset(file_path: str, format: str = "jsonl") -> str:
            result = await self.finetuner.load_dataset_from_file(file_path, format)
            if "dataset_object" in result:
                del result["dataset_object"]
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="finetune.prepare_dataset",
                       description="Prepare inline data for fine-tuning")
        async def prepare_dataset(data: str) -> str:
            data_list = json.loads(data) if isinstance(data, str) else data
            result = await self.finetuner.prepare_dataset(data_list)
            if "dataset_object" in result:
                del result["dataset_object"]
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.merge_adapter",
            description=(
                "Merge a LoRA adapter into the base model to produce a standalone "
                "model directory. Optionally push the merged model to HuggingFace Hub."
            ),
        )
        async def merge_adapter(
            base_model: str,
            adapter_path: str,
            output_path: str,
            push_to_hub: Optional[str] = None,
        ) -> str:
            return json.dumps(
                await self.finetuner.merge_adapter(
                    base_model=base_model, adapter_path=adapter_path,
                    output_path=output_path, push_to_hub=push_to_hub,
                ),
                indent=2,
            )

        @self.mcp.tool(
            name="finetune.export_gguf",
            description=(
                "Export a model to GGUF format for use with llama.cpp or Ollama. "
                "Requires the llama-cpp-python package (pip install transcendence[export]). "
                "Supported quantizations: q4_0, q4_k_m, q5_k_m, q8_0, f16."
            ),
        )
        async def export_gguf(
            model_path: str,
            output_path: str,
            quantization: str = "q4_k_m",
        ) -> str:
            return json.dumps(
                await self.finetuner.export_gguf(
                    model_path=model_path, output_path=output_path,
                    quantization=quantization,
                ),
                indent=2,
            )

    # -- Training Monitor --
    def _register_training_monitor_tools(self):
        @self.mcp.tool(
            name="finetune.train_async",
            description=(
                "Start SFT fine-tuning in the background and return a job_id immediately. "
                "Use finetune.job_status(job_id) to poll progress (step, loss, ETA, GPU). "
                "Use finetune.cancel_job(job_id) to stop training early. "
                "Same parameters as finetune.train but non-blocking."
            ),
        )
        async def train_async(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
            completion_only_loss: bool = True,
            early_stopping_patience: Optional[int] = None,
            eval_file_path: Optional[str] = None,
            push_to_hub: Optional[str] = None,
            lr_scheduler_type: str = "linear",
            warmup_ratio: float = 0.0,
            weight_decay: float = 0.0,
            max_grad_norm: float = 1.0,
            learning_rate: float = 2e-4,
            report_to: Optional[str] = None,
            max_seq_length: int = 2048,
            per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 4,
            gradient_checkpointing: bool = False,
            optim: str = "adamw_torch",
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="sft",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={
                    "num_epochs": num_epochs, "lora_r": lora_r,
                    "batch_size": per_device_train_batch_size,
                    "learning_rate": learning_rate,
                },
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_model(
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    completion_only_loss=completion_only_loss,
                    early_stopping_patience=early_stopping_patience,
                    eval_file_path=eval_file_path,
                    push_to_hub=push_to_hub,
                    lr_scheduler_type=lr_scheduler_type,
                    warmup_ratio=warmup_ratio,
                    weight_decay=weight_decay,
                    max_grad_norm=max_grad_norm,
                    learning_rate=learning_rate,
                    report_to=report_to or [],
                    max_seq_length=max_seq_length,
                    per_device_train_batch_size=per_device_train_batch_size,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                    gradient_checkpointing=gradient_checkpointing,
                    optim=optim,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True,
                "job_id": job.job_id,
                "status": "running",
                "message": "Training started. Use finetune.job_status to monitor progress.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.train_dpo_async",
            description=(
                "Start DPO fine-tuning in the background. Returns job_id. "
                "Dataset needs prompt/chosen/rejected columns. "
                "Use finetune.job_status(job_id) to monitor."
            ),
        )
        async def train_dpo_async(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            beta: float = 0.1,
            use_lora: bool = True,
            lora_r: int = 8,
            resume_from_checkpoint: Optional[str] = None,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="dpo",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={"num_epochs": num_epochs, "beta": beta},
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_dpo_model(
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    beta=beta,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    resume_from_checkpoint=resume_from_checkpoint,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True, "job_id": job.job_id,
                "status": "running",
                "message": "DPO training started. Use finetune.job_status to monitor.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.train_grpo_async",
            description=(
                "Start GRPO fine-tuning in the background. Returns job_id. "
                "Dataset needs prompt/responses/rewards columns. "
                "Use finetune.job_status(job_id) to monitor."
            ),
        )
        async def train_grpo_async(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            num_generations: int = 4,
            max_prompt_length: int = 512,
            max_completion_length: int = 256,
            resume_from_checkpoint: Optional[str] = None,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="grpo",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={"num_epochs": num_epochs, "num_generations": num_generations},
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_grpo_model(
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    num_generations=num_generations,
                    max_prompt_length=max_prompt_length,
                    max_completion_length=max_completion_length,
                    resume_from_checkpoint=resume_from_checkpoint,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True, "job_id": job.job_id,
                "status": "running",
                "message": "GRPO training started. Use finetune.job_status to monitor.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.train_kto_async",
            description=(
                "Start KTO fine-tuning in the background. Returns job_id. "
                "Dataset needs prompt/completion/label columns. "
                "Use finetune.job_status(job_id) to monitor."
            ),
        )
        async def train_kto_async(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            beta: float = 0.1,
            use_lora: bool = True,
            lora_r: int = 8,
            desirable_weight: float = 1.0,
            undesirable_weight: float = 1.0,
            resume_from_checkpoint: Optional[str] = None,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="kto",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={"num_epochs": num_epochs, "beta": beta},
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_kto_model(
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_epochs=num_epochs,
                    beta=beta,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    desirable_weight=desirable_weight,
                    undesirable_weight=undesirable_weight,
                    resume_from_checkpoint=resume_from_checkpoint,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True, "job_id": job.job_id,
                "status": "running",
                "message": "KTO training started. Use finetune.job_status to monitor.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.train_curriculum_async",
            description=(
                "Start curriculum learning in the background. Returns job_id. "
                "Scores dataset, buckets by difficulty, trains stage-by-stage. "
                "Use finetune.job_status(job_id) to monitor."
            ),
        )
        async def train_curriculum_async(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_stages: int = 3,
            num_epochs_per_stage: int = 1,
            score_column: str = "weighted_score",
            difficulty_order: str = "easy_first",
            use_lora: bool = True,
            lora_r: int = 8,
            lora_alpha: int = 16,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            job = self.job_manager.create_job(
                trainer_type="curriculum",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={
                    "num_stages": num_stages,
                    "epochs_per_stage": num_epochs_per_stage,
                },
            )
            dataset_obj = load_result["dataset_object"]

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_curriculum_model(
                    dataset=dataset_obj,
                    output_dir=output_dir,
                    base_model=base_model,
                    num_stages=num_stages,
                    num_epochs_per_stage=num_epochs_per_stage,
                    score_column=score_column,
                    difficulty_order=difficulty_order,
                    use_lora=use_lora,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True, "job_id": job.job_id,
                "status": "running",
                "message": "Curriculum training started. Use finetune.job_status to monitor.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.sequential_train_async",
            description=(
                "Start sequential multi-technique training in the background. Returns job_id. "
                "Chains SFT -> DPO -> GRPO -> KTO stages, auto-merging LoRA between them. "
                "Use finetune.job_status(job_id) to monitor."
            ),
        )
        async def sequential_train_async(
            stages: str, output_dir: str,
            base_model: Optional[str] = None,
            merge_between_stages: bool = True,
        ) -> str:
            try:
                stages_list = json.loads(stages)
            except json.JSONDecodeError as e:
                return json.dumps({"success": False, "error": f"Invalid stages JSON: {e}"}, indent=2)

            resolved_base = base_model or self.finetuner.config.base_model
            techniques = [s.get("technique", "?") for s in stages_list]
            job = self.job_manager.create_job(
                trainer_type="sequential",
                base_model=resolved_base,
                output_dir=output_dir,
                config_summary={"techniques": techniques, "num_stages": len(stages_list)},
            )

            async def _run_training(extra_callbacks=None):
                return await self.finetuner.train_sequential(
                    stages=stages_list,
                    output_dir=output_dir,
                    base_model=base_model,
                    merge_between_stages=merge_between_stages,
                    extra_callbacks=extra_callbacks,
                )

            await self.job_manager.start_job(job.job_id, _run_training)
            return json.dumps({
                "success": True, "job_id": job.job_id,
                "status": "running",
                "message": "Sequential training started. Use finetune.job_status to monitor.",
            }, indent=2)

        @self.mcp.tool(
            name="finetune.job_status",
            description=(
                "Get real-time status of a training job. Returns current step, max steps, "
                "epoch, loss, learning rate, eval_loss, ETA, GPU memory, percent complete, "
                "and full result when completed. Use the job_id from finetune.train_async."
            ),
        )
        async def job_status(job_id: str) -> str:
            job = self.job_manager.get_job(job_id)
            if job is None:
                return json.dumps({"success": False, "error": f"Job not found: {job_id}"}, indent=2)
            return json.dumps({"success": True, **job.model_dump()}, indent=2)

        @self.mcp.tool(
            name="finetune.list_jobs",
            description=(
                "List all training jobs (running, completed, failed, cancelled). "
                "Optionally filter by status. Returns summary of each job."
            ),
        )
        async def list_jobs(
            status: Optional[str] = None,
            limit: int = 20,
        ) -> str:
            from shared.training_jobs import JobStatus as JS
            status_enum = JS(status) if status else None
            jobs = self.job_manager.list_jobs(status=status_enum, limit=limit)
            return json.dumps({
                "success": True,
                "count": len(jobs),
                "jobs": [j.model_dump() for j in jobs],
            }, indent=2)

        @self.mcp.tool(
            name="finetune.cancel_job",
            description=(
                "Cancel a running training job. The model checkpoint at the current step "
                "will be saved. The job status changes to 'cancelled'."
            ),
        )
        async def cancel_job(job_id: str) -> str:
            success = self.job_manager.cancel_job(job_id)
            if not success:
                return json.dumps({
                    "success": False,
                    "error": f"Job not found or not running: {job_id}",
                }, indent=2)
            return json.dumps({
                "success": True,
                "job_id": job_id,
                "message": "Cancellation requested. Job will stop after current step.",
            }, indent=2)

    # -- Test --
    def _register_test_tools(self):
        @self.mcp.tool(name="test.inference",
                       description="Run inference on prompts using a model")
        async def run_inference(
            prompts: List[str], model_path: str,
            adapter_path: Optional[str] = None,
            max_new_tokens: int = 512,
        ) -> str:
            result = await self.finetuner.run_inference(
                prompts=prompts, model_path=model_path,
                adapter_path=adapter_path, max_new_tokens=max_new_tokens,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="test.compare_models",
                       description="Compare base model vs fine-tuned model")
        async def compare_models(
            prompts: List[str], base_model_path: str,
            finetuned_adapter_path: str,
            max_new_tokens: int = 512,
        ) -> str:
            result = await self.finetuner.compare_models(
                prompts=prompts,
                base_model_path=base_model_path,
                finetuned_adapter_path=finetuned_adapter_path,
                max_new_tokens=max_new_tokens,
            )
            return json.dumps(result, indent=2)

    # -- Validate --
    def _register_validate_tools(self):
        @self.mcp.tool(name="validate.model_info",
                       description="Get info about a local model or adapter")
        async def model_info(model_path: str) -> str:
            return json.dumps(self.finetuner.get_model_info(model_path), indent=2)

        @self.mcp.tool(name="validate.list_models",
                       description="List locally cached HuggingFace models")
        async def list_models(query: str = "") -> str:
            result = await self.finetuner.list_available_base_models(query=query)
            return json.dumps(result, indent=2)

    # -- Host --
    def _register_host_tools(self):
        @self.mcp.tool(
            name="host.deploy_mcp",
            description=(
                "Deploy a fine-tuned model as an MCP tool server. "
                "Use model_path from finetune.train* results as adapter_path. "
                "Returns deployment_id and endpoint URL."
            ),
        )
        async def deploy_mcp(
            model_path: str, adapter_path: Optional[str] = None,
            port: int = 8001,
        ) -> str:
            config = HostingConfig(model_path=model_path, adapter_path=adapter_path, port=port)
            return json.dumps(await self.hoster.deploy_as_mcp(config), indent=2)

        @self.mcp.tool(
            name="host.deploy_api",
            description=(
                "Deploy a fine-tuned model as a REST API with /generate endpoint. "
                "Use model_path from finetune.train* results as adapter_path. "
                "Returns deployment_id and endpoint URL."
            ),
        )
        async def deploy_api(
            model_path: str, adapter_path: Optional[str] = None,
            port: int = 8001,
        ) -> str:
            config = HostingConfig(model_path=model_path, adapter_path=adapter_path, port=port)
            return json.dumps(await self.hoster.deploy_as_api(config), indent=2)

        @self.mcp.tool(name="host.list_deployments",
                       description="List running model deployments")
        async def list_deployments() -> str:
            return json.dumps(await self.hoster.list_deployments(), indent=2)

        @self.mcp.tool(name="host.stop",
                       description="Stop a running deployment")
        async def stop_deployment(deployment_id: str) -> str:
            return json.dumps(await self.hoster.stop_deployment(deployment_id), indent=2)

        @self.mcp.tool(
            name="host.health",
            description="Health check on a running deployment. Returns status, uptime, endpoint.",
        )
        async def host_health(deployment_id: str) -> str:
            return json.dumps(await self.hoster.health_check(deployment_id), indent=2)

        @self.mcp.tool(
            name="host.chat",
            description=(
                "Chat with a deployed or local fine-tuned model. "
                "IMPORTANT: After any finetune.train* with deploy=True succeeds, "
                "offer to use this tool so the user can test the model interactively. "
                "Pass the endpoint from the deployment result, or a model_path for direct loading. "
                "Use conversation_id to maintain multi-turn context across calls. "
                "Also share the deployment's chat_command for standalone CLI access."
            ),
        )
        async def chat_with_model(
            message: str,
            endpoint: Optional[str] = None,
            model_path: Optional[str] = None,
            adapter_path: Optional[str] = None,
            conversation_id: Optional[str] = None,
            max_new_tokens: int = 512,
            system_prompt: Optional[str] = None,
        ) -> str:
            from hosting_pipeline.services.chat_service import ChatSession

            cid = conversation_id or str(uuid.uuid4())[:8]

            if cid in self._chat_sessions:
                session = self._chat_sessions[cid]
            else:
                config = ChatConfig(
                    endpoint=endpoint,
                    model_path=model_path,
                    adapter_path=adapter_path,
                    max_new_tokens=max_new_tokens,
                    system_prompt=system_prompt,
                )
                session = ChatSession(config)
                await session.initialize()
                self._chat_sessions[cid] = session

            response = await session.send_message(message)
            return json.dumps(
                {
                    "success": True,
                    "conversation_id": cid,
                    "response": response,
                    "turns": session.get_info()["turns"],
                },
                indent=2,
            )

    # -- Workflow --
    def _register_workflow_tools(self):
        @self.mcp.tool(name="workflow.full_pipeline",
                       description="End-to-end: Extract -> Generate -> Clean -> Normalize -> Evaluate -> Filter -> Train -> Test -> Host")
        async def full_pipeline(
            file_path: str,
            technique: str = "sft",
            output_dir: str = "./output",
            quality_threshold: float = 0.7,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            result = await self.orchestrator.full_pipeline(
                file_path=file_path,
                technique=technique,
                output_dir=output_dir,
                quality_threshold=quality_threshold,
                base_model=base_model,
                num_epochs=num_epochs,
                deploy=deploy,
                deploy_port=deploy_port,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="workflow.generate_and_evaluate",
                       description="Extract -> Generate -> Clean -> Normalize -> Evaluate -> Filter")
        async def generate_and_evaluate(
            file_path: str,
            technique: str = "sft",
            quality_threshold: float = 0.7,
        ) -> str:
            result = await self.orchestrator.generate_and_evaluate(
                file_path=file_path,
                technique=technique,
                quality_threshold=quality_threshold,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="workflow.curriculum_pipeline",
            description=(
                "Full curriculum learning pipeline from raw documents to a compared model. "
                "Runs: Extract -> Generate -> Clean -> Normalize -> Evaluate -> Filter -> "
                "Curriculum Train (staged easy->hard) -> Compare vs base model -> (optional) Deploy. "
                "Accepts one or more Markdown/PDF/text files. "
                "The evaluate step scores every row with weighted_score so curriculum "
                "training always receives a pre-scored dataset -- no double work."
            ),
        )
        async def curriculum_pipeline(
            file_paths: Union[str, List[str]],
            output_dir: str,
            technique: str = "sft",
            quality_threshold: float = 0.6,
            base_model: Optional[str] = None,
            num_stages: int = 3,
            num_epochs_per_stage: int = 1,
            difficulty_order: str = "easy_first",
            use_lora: bool = True,
            lora_r: int = 8,
            deploy: bool = False,
            deploy_port: int = 8001,
        ) -> str:
            # Accept a JSON-encoded list or a bare string (single file)
            if isinstance(file_paths, str):
                try:
                    parsed = json.loads(file_paths)
                    file_paths = parsed if isinstance(parsed, list) else [parsed]
                except (json.JSONDecodeError, ValueError):
                    file_paths = [file_paths]
            result = await self.orchestrator.curriculum_pipeline(
                file_paths=file_paths,
                output_dir=output_dir,
                technique=technique,
                quality_threshold=quality_threshold,
                base_model=base_model,
                num_stages=num_stages,
                num_epochs_per_stage=num_epochs_per_stage,
                difficulty_order=difficulty_order,
                use_lora=use_lora,
                lora_r=lora_r,
                deploy=deploy,
                deploy_port=deploy_port,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="workflow.compare_flat_vs_curriculum",
            description=(
                "Compare flat SFT vs curriculum SFT on the same dataset. "
                "Trains both approaches and provides side-by-side comparison "
                "using test.compare_models. Returns structured results with "
                "training metrics and qualitative output comparisons."
            ),
        )
        async def compare_flat_vs_curriculum(
            dataset_path: str,
            output_dir: str = "./output/comparison",
            base_model: Optional[str] = None,
            num_epochs_flat: int = 3,
            num_stages: int = 3,
            num_epochs_per_stage: int = 1,
            difficulty_order: str = "easy_first",
            use_lora: bool = True,
            lora_r: int = 8,
            test_data_path: Optional[str] = None,
            test_prompts: Optional[str] = None,
        ) -> str:
            parsed_prompts = None
            if test_prompts:
                parsed_prompts = (
                    json.loads(test_prompts)
                    if isinstance(test_prompts, str)
                    else test_prompts
                )
            result = await self.orchestrator.compare_flat_vs_curriculum(
                dataset_path=dataset_path,
                output_dir=output_dir,
                base_model=base_model,
                num_epochs_flat=num_epochs_flat,
                num_stages=num_stages,
                num_epochs_per_stage=num_epochs_per_stage,
                difficulty_order=difficulty_order,
                use_lora=use_lora,
                lora_r=lora_r,
                test_data_path=test_data_path,
                test_prompts=parsed_prompts,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="workflow.run_pipeline",
            description=(
                "Execute a sequence of MCP tools server-side in a single call. "
                "Avoids back-and-forth round-trips between client and server. "
                "Each step specifies a tool name and params. Use '$prev.key' in "
                "params to reference the previous step's output (e.g., "
                "'$prev.model_path' passes the model_path from the prior step). "
                "Stops on first failure and returns partial results. "
                "Set dry_run=True to validate the pipeline without executing. "
                "Example steps: "
                '[{"tool":"system.preflight_check","params":{"model_name":"meta-llama/Llama-3.2-3B-Instruct"}},'
                '{"tool":"finetune.train","params":{"dataset_path":"data.jsonl"}},'
                '{"tool":"test.inference","params":{"model_path":"$prev.model_path"}}]'
            ),
        )
        async def run_pipeline(steps: str, dry_run: bool = False) -> str:
            from shared.pipeline_executor import PipelineExecutor, PipelineStep

            parsed_steps = [PipelineStep(**s) for s in json.loads(steps)]
            executor = PipelineExecutor(self.mcp._tools)
            result = await executor.execute(parsed_steps, dry_run=dry_run)
            return result.model_dump_json(indent=2)

        @self.mcp.tool(
            name="workflow.guided_pipeline",
            description=(
                "Describe what you want to do in plain English. Returns the exact "
                "sequence of MCP tool calls with parameters to execute your goal. "
                "Use this FIRST when you don't know which tools to call. "
                "This is a pure planning tool — it does NOT execute anything. "
                "Take the returned steps and either pass them to workflow.run_pipeline "
                "or call the tools individually."
            ),
        )
        async def guided_pipeline(
            goal: str,
            file_path: Optional[str] = None,
            base_model: Optional[str] = None,
        ) -> str:
            from shared.workflow_planner import WorkflowPlanner

            planner = WorkflowPlanner()
            plan = planner.plan(goal, file_path=file_path, base_model=base_model)
            return json.dumps(plan, indent=2)

    # -- Orchestration --
    def _register_orchestration_tools(self):
        @self.mcp.tool(
            name="orchestration.generate_problems",
            description="Generate synthetic orchestration tasks for a domain",
        )
        async def generate_problems(
            domain_description: str,
            num_problems: int = 50,
            tool_descriptions: Optional[List[Dict]] = None,
        ) -> str:
            result = await self.orchestration_data_service.generate_problems(
                domain_description=domain_description,
                num_problems=num_problems,
                tool_descriptions=tool_descriptions,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="orchestration.collect_trajectories",
            description="Run problems through agent, record trajectories with cost/latency",
        )
        async def collect_trajectories(
            problems: List[Dict],
            n_per_problem: int = 4,
        ) -> str:
            # Uses the orchestrator's agent (the gateway itself uses an internal agent)
            from agentsoul.core.agent import AgentSoul
            from shared.provider_factory import create_llm
            from shared.config import PipelineConfig
            llm = create_llm(PipelineConfig())
            agent = AgentSoul(llm_provider=llm)
            result = await self.orchestration_data_service.collect_trajectories(
                problems=problems, agent=agent, n_per_problem=n_per_problem,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="orchestration.build_training_data",
            description="Score trajectories and convert to SFT/DPO/GRPO training format",
        )
        async def build_training_data(
            collected: List[Dict],
            format: str = "sft",
            tool_descriptions: Optional[List[Dict]] = None,
            cost_budget: float = 1.0,
            time_budget: float = 60.0,
        ) -> str:
            result = await self.orchestration_data_service.build_training_data(
                collected=collected,
                format=format,
                tool_descriptions=tool_descriptions,
                cost_budget=cost_budget,
                time_budget=time_budget,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="orchestration.train_orchestrator",
            description="Full pipeline: problems -> trajectories -> rewards -> train -> deploy",
        )
        async def train_orchestrator(
            domain_description: str,
            num_problems: int = 50,
            n_per_problem: int = 4,
            output_dir: str = "./output/orchestrator",
            output_format: str = "sft",
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            deploy: bool = False,
            deploy_port: int = 8002,
        ) -> str:
            from agentsoul.core.agent import AgentSoul
            from shared.provider_factory import create_llm
            from shared.config import PipelineConfig
            llm = create_llm(PipelineConfig())
            agent = AgentSoul(llm_provider=llm)
            result = await self.orchestrator.train_orchestrator(
                domain_description=domain_description,
                agent=agent,
                num_problems=num_problems,
                n_per_problem=n_per_problem,
                output_dir=output_dir,
                output_format=output_format,
                base_model=base_model,
                num_epochs=num_epochs,
                deploy=deploy,
                deploy_port=deploy_port,
            )
            return json.dumps(result, indent=2)

    # -- Judge (Advanced LLM-as-a-Judge) --
    def _register_judge_tools(self):
        @self.mcp.tool(
            name="judge.evaluate",
            description="Run a single LLM-as-a-judge evaluation on one sample with custom criteria/rubric",
        )
        async def judge_evaluate(
            question: str,
            generated: str,
            reference: Optional[str] = None,
            generated_b: Optional[str] = None,
            judge_type: str = "pointwise",
            judge_model: str = "gpt-4o",
            criteria: Optional[List[Dict]] = None,
            rubric: Optional[Dict] = None,
        ) -> str:
            result = await self.advanced_judge.evaluate_single(
                question=question, generated=generated,
                reference=reference, generated_b=generated_b,
                judge_type=judge_type, judge_model=judge_model,
                criteria=criteria, rubric=rubric,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.evaluate_multi",
            description="Run multiple LLM judges in parallel on one sample and aggregate scores",
        )
        async def judge_evaluate_multi(
            question: str,
            generated: str,
            reference: Optional[str] = None,
            generated_b: Optional[str] = None,
            judge_type: str = "pointwise",
            judges: Optional[List[Dict]] = None,
            criteria: Optional[List[Dict]] = None,
            rubric: Optional[Dict] = None,
            aggregation: str = "mean",
        ) -> str:
            result = await self.advanced_judge.evaluate_multi_judge(
                question=question, generated=generated,
                reference=reference, generated_b=generated_b,
                judge_type=judge_type, judges=judges,
                criteria=criteria, rubric=rubric,
                aggregation=aggregation,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.evaluate_batch",
            description="Run batch evaluation with multi-judge support and custom criteria",
        )
        async def judge_evaluate_batch(
            test_data: List[Dict],
            judge_type: str = "pointwise",
            judges: Optional[List[Dict]] = None,
            criteria: Optional[List[Dict]] = None,
            rubric: Optional[Dict] = None,
            aggregation: str = "mean",
        ) -> str:
            result = await self.advanced_judge.evaluate_batch(
                test_data=test_data, judge_type=judge_type,
                judges=judges, criteria=criteria,
                rubric=rubric, aggregation=aggregation,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.compare_pair",
            description="Pairwise comparison: which of two outputs is better?",
        )
        async def judge_compare_pair(
            question: str,
            generated_a: str,
            generated_b: str,
            reference: Optional[str] = None,
            judges: Optional[List[Dict]] = None,
            criteria: Optional[List[Dict]] = None,
        ) -> str:
            result = await self.advanced_judge.evaluate_multi_judge(
                question=question,
                generated=generated_a,
                generated_b=generated_b,
                reference=reference,
                judge_type="pairwise",
                judges=judges,
                criteria=criteria,
                aggregation="mean",
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.list_types",
            description="List available judge evaluation types (pointwise, pairwise, reference_free, rubric)",
        )
        async def judge_list_types() -> str:
            return json.dumps(self.advanced_judge.list_judge_types(), indent=2)

        @self.mcp.tool(
            name="judge.export",
            description="Export judge evaluation results as JSONL or JSON",
        )
        async def judge_export(
            results: List[Dict],
            output_path: str,
            format: str = "jsonl",
        ) -> str:
            result = await self.advanced_judge.export_results(
                results=results, output_path=output_path, format=format,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="judge.create_rubric",
            description="Validate a rubric definition and return the parsed result",
        )
        async def judge_create_rubric(
            name: str,
            criteria: List[Dict],
            description: str = "",
        ) -> str:
            from model_evaluator_pipeline.judges.models import JudgeCriterion, JudgeRubric
            try:
                rubric = JudgeRubric(
                    name=name, description=description,
                    criteria=[JudgeCriterion(**c) for c in criteria],
                )
                return json.dumps({
                    "success": True,
                    "rubric": rubric.model_dump(),
                }, indent=2)
            except Exception as e:
                return json.dumps({"success": False, "error": str(e)}, indent=2)

    # -- FT Evaluator (Domain Knowledge Judge) --
    def _register_ft_evaluator_tools(self):
        @self.mcp.tool(
            name="ft_eval.single",
            description="Domain knowledge PASS/FAIL evaluation on one sample (single judge)",
        )
        async def ft_eval_single(
            instruction: str,
            generated: str,
            reference: str,
            judge_model: Optional[str] = None,
            ksmi_label: Optional[str] = None,
        ) -> str:
            verdict = await self.ft_evaluator.evaluate_single(
                instruction=instruction,
                generated=generated,
                reference=reference,
                judge_model=judge_model,
                ksmi_label=ksmi_label,
            )
            return json.dumps(verdict.model_dump(), indent=2)

        @self.mcp.tool(
            name="ft_eval.batch",
            description="Batch domain knowledge evaluation with multi-judge + KSMI labels",
        )
        async def ft_eval_batch(
            test_data: List[Dict],
            judge_models: Optional[List[str]] = None,
        ) -> str:
            result = await self.ft_evaluator.evaluate_batch(
                test_data=test_data,
                judge_models=judge_models,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="ft_eval.summary",
            description="Compute stakeholder summary (pass rate, failure types, severity, KSMI breakdown)",
        )
        async def ft_eval_summary(results: List[Dict]) -> str:
            from model_evaluator_pipeline.models.ft_evaluator import FTEvalResult
            parsed = [FTEvalResult(**r) for r in results]
            summary = self.ft_evaluator.compute_summary(parsed)
            return json.dumps({"success": True, "summary": summary.model_dump()}, indent=2)

        @self.mcp.tool(
            name="ft_eval.export",
            description="Export domain knowledge evaluation results as JSONL or JSON",
        )
        async def ft_eval_export(
            results: List[Dict],
            output_path: str,
            format: str = "jsonl",
        ) -> str:
            from model_evaluator_pipeline.models.ft_evaluator import FTEvalResult
            parsed = [FTEvalResult(**r) for r in results]
            result = await self.ft_evaluator.export_results(
                results=parsed,
                output_path=output_path,
                format=format,
            )
            return json.dumps(result, indent=2)

    # ------------------------------------------------------------------ #
    # Dataset tools
    # ------------------------------------------------------------------ #
    def _register_dataset_tools(self):
        @self.mcp.tool(
            name="dataset.save",
            description=(
                "Save data_points to a file (JSONL, JSON, or Parquet). Use after "
                "generate.from_document, clean.dataset, normalize.dataset, or "
                "evaluate.filter_by_quality to persist results to disk. "
                "Returns dataset_id (derived from filename) and file_path."
            ),
        )
        async def dataset_save(
            data_points: List[Dict],
            output_path: str,
            format: str = "jsonl",
        ) -> str:
            return json.dumps(
                await self.dataset_service.save(data_points, output_path, format),
                indent=2,
            )

        @self.mcp.tool(
            name="dataset.load",
            description=(
                "Load a dataset from file and return data_points. "
                "Auto-detects format from extension (.jsonl, .json, .parquet, .csv). "
                "Pipe results into clean.dataset, evaluate.dataset, or finetune.train."
            ),
        )
        async def dataset_load(file_path: str) -> str:
            return json.dumps(
                await self.dataset_service.load(file_path), indent=2,
            )

        @self.mcp.tool(
            name="dataset.preview",
            description="Show first N rows from a dataset file without loading everything.",
        )
        async def dataset_preview(file_path: str, n: int = 5) -> str:
            return json.dumps(
                await self.dataset_service.preview(file_path, n), indent=2,
            )

        @self.mcp.tool(
            name="dataset.info",
            description=(
                "Get metadata about a dataset file: row count, columns, "
                "detected technique (sft/dpo/grpo/kto), file size."
            ),
        )
        async def dataset_info(file_path: str) -> str:
            return json.dumps(
                await self.dataset_service.info(file_path), indent=2,
            )

        @self.mcp.tool(
            name="dataset.split",
            description=(
                "Split a dataset file into train/val/test files with configurable "
                "ratios and random seed. Returns paths to the three split files."
            ),
        )
        async def dataset_split(
            file_path: str,
            output_dir: str,
            train_ratio: float = 0.8,
            val_ratio: float = 0.1,
            test_ratio: float = 0.1,
            seed: int = 42,
        ) -> str:
            return json.dumps(
                await self.dataset_service.split(
                    file_path, output_dir,
                    train_ratio=train_ratio, val_ratio=val_ratio,
                    test_ratio=test_ratio, seed=seed,
                ),
                indent=2,
            )

        @self.mcp.tool(
            name="dataset.merge",
            description=(
                "Merge multiple dataset files into one. "
                "Optionally deduplicate by a key (default: instruction)."
            ),
        )
        async def dataset_merge(
            file_paths: List[str],
            output_path: str,
            deduplicate: bool = False,
            dedup_key: str = "instruction",
        ) -> str:
            return json.dumps(
                await self.dataset_service.merge(
                    file_paths, output_path,
                    deduplicate=deduplicate, dedup_key=dedup_key,
                ),
                indent=2,
            )

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    def _wrap_tools_with_diagnostics(self) -> None:
        """Wrap all registered async tool functions with timing + emit_tool_call.

        Patches self.mcp._tools in-place after registration so no individual
        handler needs to be modified. No-op if DiagnosticWriter is not initialized.
        """
        from shared.diagnostics import emit_tool_call, sanitize

        def _make_wrapper(name: str, fn):
            async def _wrapper(**kwargs):
                # Generate a per-call trace_id (gateway is a separate process)
                from shared.diagnostics import trace_id_var
                trace_id_var.set(str(uuid.uuid4()))

                t0 = time.perf_counter()
                try:
                    result = await fn(**kwargs)
                    latency = round(time.perf_counter() - t0, 4)
                    preview = str(result)[:500] if result else ""
                    await emit_tool_call(
                        tool_name=name,
                        arguments=sanitize(kwargs),
                        result_preview=preview,
                        latency_s=latency,
                        success=True,
                    )
                    return result
                except Exception as exc:
                    latency = round(time.perf_counter() - t0, 4)
                    await emit_tool_call(
                        tool_name=name,
                        arguments=sanitize(kwargs),
                        result_preview="",
                        latency_s=latency,
                        success=False,
                        error=str(exc),
                    )
                    raise

            return _wrapper

        for tool_name in list(self.mcp._tools.keys()):
            tool_info = self.mcp._tools[tool_name]
            original_func = tool_info["func"]
            if inspect.iscoroutinefunction(original_func):
                tool_info["func"] = _make_wrapper(tool_name, original_func)

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def run(self, transport=None):
        # Set a gateway-scoped session_id before the event loop starts so
        # it propagates (via context copy) to all asyncio tasks.
        from shared.diagnostics import init_diagnostics, session_id_var
        gw_session = f"gw-{str(uuid.uuid4())[:8]}"
        session_id_var.set(gw_session)
        init_diagnostics(log_root="logs")
        self.mcp.run(transport)


# Backwards-compatible alias
AgentYGateway = TranscendenceGateway
