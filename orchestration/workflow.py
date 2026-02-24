"""
End-to-end pipeline orchestrator.

Composes: Extract → Generate → Clean → Normalize → Evaluate → Filter → Train → Test → Host
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from AgentY.shared.config import (
    CleaningConfig,
    NormalizationConfig,
    EvaluatorConfig,
    FinetuningConfig,
    HostingConfig,
    OrchestrationConfig,
)


class PipelineOrchestrator:
    """Chains pipeline services into composable workflows."""

    def __init__(self, generator, cleaner, normalizer, evaluator, finetuner, hoster,
                 orchestration_data_service=None):
        self.generator = generator
        self.cleaner = cleaner
        self.normalizer = normalizer
        self.evaluator = evaluator
        self.finetuner = finetuner
        self.hoster = hoster
        self.orchestration_data_service = orchestration_data_service

    async def generate_and_evaluate(
        self,
        file_path: str,
        technique: str = "sft",
        quality_threshold: float = 0.7,
        custom_template: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract → Generate → Clean → Normalize → Evaluate → Filter"""

        # 1) Extract + Generate
        gen_result = await self.generator.generate_from_document(
            technique=technique,
            file_path=file_path,
            custom_template=custom_template,
        )
        if not gen_result.get("success"):
            return gen_result

        data_points = gen_result["data_points"]

        # 2) Clean
        clean_result = await self.cleaner.clean_dataset(data_points, CleaningConfig())
        data_points = clean_result["data_points"]

        # 3) Normalize
        norm_result = await self.normalizer.normalize_dataset(
            data_points, NormalizationConfig(target_format=technique),
        )
        data_points = norm_result["data_points"]

        # 4) Evaluate
        eval_result = await self.evaluator.evaluate_dataset(data_points)
        data_points = eval_result["data_points"]

        # 5) Filter
        filter_result = await self.evaluator.filter_by_quality(data_points, quality_threshold)
        data_points = filter_result["data_points"]

        # 6) Statistics
        stats = await self.evaluator.analyze_statistics(data_points)

        return {
            "success": True,
            "file_path": file_path,
            "technique": technique,
            "pipeline_stages": {
                "generated": gen_result.get("stats", {}).get("total_data_points", 0),
                "after_cleaning": clean_result["cleaned_count"],
                "after_normalization": norm_result["count"],
                "after_evaluation": eval_result["count"],
                "after_filtering": filter_result["filtered_count"],
            },
            "quality_threshold": quality_threshold,
            "statistics": stats.get("statistics", {}),
            "data_points": data_points,
        }

    async def full_pipeline(
        self,
        file_path: str,
        technique: str = "sft",
        output_dir: str = "./output",
        quality_threshold: float = 0.7,
        base_model: Optional[str] = None,
        num_epochs: int = 3,
        deploy: bool = False,
        deploy_port: int = 8001,
    ) -> Dict[str, Any]:
        """Extract → Generate → Clean → Normalize → Evaluate → Filter → Train → Test → Host"""

        # Stages 1-5: generate + evaluate
        ge_result = await self.generate_and_evaluate(
            file_path=file_path,
            technique=technique,
            quality_threshold=quality_threshold,
        )
        if not ge_result.get("success"):
            return ge_result

        data_points = ge_result["data_points"]

        # 6) Export dataset for finetuning
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_path = str(output_path / "dataset.jsonl")

        await self.generator.export_dataset(
            data_points=data_points,
            output_path=dataset_path,
            format="jsonl",
        )

        # 7) Train
        load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
        if not load_result.get("success"):
            return {**ge_result, "training": load_result}

        train_result = await self.finetuner.train_model(
            dataset=load_result["dataset_object"],
            output_dir=str(output_path / "model"),
            base_model=base_model,
            num_epochs=num_epochs,
        )

        # 8) Test inference
        test_result = None
        if train_result.get("success"):
            test_prompts = []
            for dp in data_points[:3]:
                test_prompts.append(dp.get("instruction", dp.get("prompt", "")))

            if test_prompts:
                test_result = await self.finetuner.run_inference(
                    prompts=test_prompts,
                    model_path=base_model or "meta-llama/Llama-3.2-3B-Instruct",
                    adapter_path=train_result.get("model_path"),
                )

        # 9) Deploy (optional)
        deploy_result = None
        if deploy and train_result.get("success"):
            config = HostingConfig(
                model_path=base_model or "meta-llama/Llama-3.2-3B-Instruct",
                adapter_path=train_result.get("model_path"),
                port=deploy_port,
            )
            deploy_result = await self.hoster.deploy_as_mcp(config)

        return {
            "success": True,
            "file_path": file_path,
            "technique": technique,
            "pipeline_stages": ge_result["pipeline_stages"],
            "quality_threshold": quality_threshold,
            "dataset_path": dataset_path,
            "training": train_result,
            "testing": test_result,
            "deployment": deploy_result,
        }

    async def train_orchestrator(
        self,
        domain_description: str,
        agent,
        num_problems: int = 50,
        n_per_problem: int = 4,
        output_dir: str = "./output/orchestrator",
        output_format: str = "sft",
        base_model: Optional[str] = None,
        num_epochs: int = 3,
        cost_budget: float = 1.0,
        time_budget: float = 60.0,
        tool_descriptions: Optional[list] = None,
        deploy: bool = False,
        deploy_port: int = 8002,
    ) -> Dict[str, Any]:
        """
        Full orchestration training pipeline:
        Problems -> Trajectories -> Rewards -> Training data -> Fine-tune -> Deploy
        """
        if self.orchestration_data_service is None:
            return {"success": False, "error": "OrchestrationDataService not configured"}

        ods = self.orchestration_data_service

        # 1) Generate synthetic orchestration problems
        problems = await ods.generate_problems(
            domain_description=domain_description,
            num_problems=num_problems,
            tool_descriptions=tool_descriptions,
        )

        # 2) Collect trajectories (run agent N times per problem)
        collected = await ods.collect_trajectories(
            problems=problems, agent=agent, n_per_problem=n_per_problem,
        )

        # 3) Score trajectories + build training data
        training_data = await ods.build_training_data(
            collected=collected,
            format=output_format,
            tool_descriptions=tool_descriptions,
            cost_budget=cost_budget,
            time_budget=time_budget,
        )

        # 4) Export to JSONL
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_path = str(output_path / "orchestrator_dataset.jsonl")
        await ods.export(training_data, dataset_path, file_format="jsonl")

        # 5) Fine-tune small model
        load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
        if not load_result.get("success"):
            return {
                "success": False,
                "problems_generated": len(problems),
                "trajectories_collected": sum(len(c["trajectories"]) for c in collected),
                "training_examples": len(training_data),
                "dataset_path": dataset_path,
                "error": load_result,
            }

        train_result = await self.finetuner.train_model(
            dataset=load_result["dataset_object"],
            output_dir=str(output_path / "model"),
            base_model=base_model,
            num_epochs=num_epochs,
        )

        # 6) Optionally deploy
        deploy_result = None
        if deploy and train_result.get("success"):
            config = HostingConfig(
                model_path=base_model or "meta-llama/Llama-3.2-3B-Instruct",
                adapter_path=train_result.get("model_path"),
                port=deploy_port,
            )
            deploy_result = await self.hoster.deploy_as_mcp(config)

        return {
            "success": True,
            "domain": domain_description,
            "problems_generated": len(problems),
            "trajectories_collected": sum(len(c["trajectories"]) for c in collected),
            "training_examples": len(training_data),
            "output_format": output_format,
            "dataset_path": dataset_path,
            "training": train_result,
            "deployment": deploy_result,
        }
