"""
AgentY Unified MCP Gateway
============================

Single entry point exposing all pipeline operations as MCP tools.
Uses agentsoul's MCPServer (production-grade HTTP+stdio transport).

28 tools across 10 namespaces:
  extract, generate, clean, normalize, evaluate,
  finetune, test, validate, host, workflow
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
    GeneratorConfig,
    CleaningConfig,
    NormalizationConfig,
    EvaluatorConfig,
    HostingConfig,
    OrchestrationConfig,
)


class AgentYGateway:
    """Unified MCP gateway that composes all AgentY pipeline services."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        load_dotenv(override=False)
        config = config or {}
        self.mcp = MCPServer("agenty-gateway", "1.0.0")

        # Lazily-initialized services (avoid heavy imports at gateway startup)
        self._generator_svc = None
        self._cleaning_svc = None
        self._normalization_svc = None
        self._evaluator_svc = None
        self._finetuning_svc = None
        self._hosting_svc = None
        self._orchestrator = None
        self._orchestration_data_svc = None

        self._config = config
        self._register_all_tools()
        self._wrap_tools_with_diagnostics()

    # ------------------------------------------------------------------ #
    # Lazy service accessors
    # ------------------------------------------------------------------ #
    @property
    def generator(self):
        if self._generator_svc is None:
            from data_generator_pipeline.services.pipeline_service import PipelineService
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
            from data_evaluator_pipeline.services.pipeline_service import EvaluatorService
            eval_config = EvaluatorConfig(**self._config.get("evaluator", {}))
            self._evaluator_svc = EvaluatorService(eval_config)
        return self._evaluator_svc

    @property
    def finetuner(self):
        if self._finetuning_svc is None:
            from finetuning_pipeline.services.pipeline_service import FineTuningService
            self._finetuning_svc = FineTuningService(
                default_base_model=self._config.get("finetuning", {}).get(
                    "base_model", "meta-llama/Llama-3.2-3B-Instruct"
                )
            )
        return self._finetuning_svc

    @property
    def hoster(self):
        if self._hosting_svc is None:
            from hosting_pipeline.services.hosting_service import HostingService
            self._hosting_svc = HostingService()
        return self._hosting_svc

    @property
    def orchestration_data_service(self):
        if self._orchestration_data_svc is None:
            from orchestration.orchestration_trainer import OrchestrationDataService
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

    # ------------------------------------------------------------------ #
    # Tool registration
    # ------------------------------------------------------------------ #
    def _register_all_tools(self):
        self._register_extract_tools()
        self._register_generate_tools()
        self._register_clean_tools()
        self._register_normalize_tools()
        self._register_evaluate_tools()
        self._register_finetune_tools()
        self._register_test_tools()
        self._register_validate_tools()
        self._register_host_tools()
        self._register_workflow_tools()
        self._register_orchestration_tools()

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

    # -- Finetune --
    def _register_finetune_tools(self):
        @self.mcp.tool(name="finetune.train",
                       description="Fine-tune a model using a dataset file")
        async def train(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            use_lora: bool = True,
            lora_r: int = 8,
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
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="finetune.train_dpo",
                       description="Fine-tune a model with DPO — dataset needs prompt/chosen/rejected columns")
        async def train_dpo(
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
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="finetune.train_grpo",
                       description="Fine-tune a model with GRPO — dataset needs prompt/responses/rewards columns")
        async def train_grpo(
            dataset_path: str, output_dir: str,
            base_model: Optional[str] = None,
            num_epochs: int = 3,
            num_generations: int = 4,
            resume_from_checkpoint: Optional[str] = None,
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
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="finetune.train_kto",
                       description="Fine-tune a model with KTO — dataset needs prompt/completion/label columns")
        async def train_kto(
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
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="finetune.train_curriculum",
            description="Curriculum fine-tune: auto-scores dataset, trains easy→hard in stages",
        )
        async def train_curriculum(
            dataset_path: str,
            output_dir: str,
            base_model: Optional[str] = None,
            num_stages: int = 3,
            num_epochs_per_stage: int = 1,
            difficulty_order: str = "easy_first",
            use_lora: bool = True,
            lora_r: int = 8,
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
                use_lora=use_lora,
                lora_r=lora_r,
            )
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
        @self.mcp.tool(name="host.deploy_mcp",
                       description="Deploy a fine-tuned model as an MCP tool server")
        async def deploy_mcp(
            model_path: str, adapter_path: Optional[str] = None,
            port: int = 8001,
        ) -> str:
            config = HostingConfig(model_path=model_path, adapter_path=adapter_path, port=port)
            return json.dumps(await self.hoster.deploy_as_mcp(config), indent=2)

        @self.mcp.tool(name="host.deploy_api",
                       description="Deploy a fine-tuned model as a REST API")
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
                "Runs: Extract → Generate → Clean → Normalize → Evaluate → Filter → "
                "Curriculum Train (staged easy→hard) → Compare vs base model → (optional) Deploy. "
                "Accepts one or more Markdown/PDF/text files. "
                "The evaluate step scores every row with weighted_score so curriculum "
                "training always receives a pre-scored dataset — no double work."
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
