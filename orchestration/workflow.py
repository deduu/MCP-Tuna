"""
End-to-end pipeline orchestrator.

Composes: Extract → Generate → Clean → Normalize → Evaluate → Filter → Train → Test → Host
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import (
    CleaningConfig,
    NormalizationConfig,
    HostingConfig,
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

    async def curriculum_pipeline(
        self,
        file_paths: List[str],
        output_dir: str = "./output",
        technique: str = "sft",
        quality_threshold: float = 0.6,
        base_model: Optional[str] = None,
        num_stages: int = 3,
        num_epochs_per_stage: int = 1,
        difficulty_order: str = "easy_first",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        custom_template: Optional[str] = None,
        deploy: bool = False,
        deploy_port: int = 8001,
    ) -> Dict[str, Any]:
        """Extract → Generate → Clean → Normalize → Evaluate → Filter → Curriculum Train → Compare → Host

        Accepts one or more document files. The evaluate step writes weighted_score
        into every row, so curriculum training always receives a pre-scored dataset
        and skips inline re-scoring.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # ── 1-2. Extract + Generate from every file ───────────────────────
        all_data_points: List[Dict[str, Any]] = []
        per_file_counts: Dict[str, int] = {}

        for fp in file_paths:
            gen_result = await self.generator.generate_from_document(
                technique=technique,
                file_path=fp,
                custom_template=custom_template,
            )
            if not gen_result.get("success"):
                return {
                    "success": False,
                    "error": f"Generation failed for {fp}: {gen_result.get('error')}",
                    "file_path": fp,
                }
            pts = gen_result.get("data_points", [])
            all_data_points.extend(pts)
            per_file_counts[fp] = len(pts)

        total_generated = len(all_data_points)
        if not all_data_points:
            return {"success": False, "error": "No data points generated from provided files."}

        # ── 3. Clean ──────────────────────────────────────────────────────
        clean_result = await self.cleaner.clean_dataset(all_data_points, CleaningConfig())
        all_data_points = clean_result["data_points"]

        # ── 4. Normalize ──────────────────────────────────────────────────
        norm_result = await self.normalizer.normalize_dataset(
            all_data_points, NormalizationConfig(target_format=technique),
        )
        all_data_points = norm_result["data_points"]

        # ── 5. Evaluate (adds weighted_score to every row) ────────────────
        eval_result = await self.evaluator.evaluate_dataset(all_data_points)
        all_data_points = eval_result["data_points"]

        # ── 6. Filter by quality ──────────────────────────────────────────
        filter_result = await self.evaluator.filter_by_quality(all_data_points, quality_threshold)
        all_data_points = filter_result["data_points"]

        if not all_data_points:
            return {
                "success": False,
                "error": (
                    f"No data points passed the quality threshold ({quality_threshold}). "
                    "Lower the threshold or improve your source documents."
                ),
                "pipeline_stages": {
                    "generated": total_generated,
                    "after_cleaning": clean_result["cleaned_count"],
                    "after_normalization": norm_result["count"],
                    "after_evaluation": eval_result["count"],
                    "after_filtering": 0,
                },
            }

        # ── 7. Export scored dataset to JSONL ─────────────────────────────
        dataset_path = str(output_path / "curriculum_dataset.jsonl")
        await self.generator.export_dataset(
            data_points=all_data_points,
            output_path=dataset_path,
            format="jsonl",
        )

        # ── 8. Curriculum training (dataset already has weighted_score) ───
        load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
        if not load_result.get("success"):
            return {
                "success": False,
                "error": f"Failed to load exported dataset: {load_result.get('error')}",
            }

        train_result = await self.finetuner.train_curriculum_model(
            dataset=load_result["dataset_object"],
            output_dir=str(output_path / "model"),
            base_model=base_model,
            num_stages=num_stages,
            num_epochs_per_stage=num_epochs_per_stage,
            difficulty_order=difficulty_order,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # ── 9. Compare: base model vs curriculum-trained model ────────────
        compare_result = None
        if train_result.get("success"):
            resolved_base = base_model or "meta-llama/Llama-3.2-3B-Instruct"
            final_model   = train_result["final_model_path"]

            # Use the first 3 data points as comparison prompts
            test_prompts = []
            for dp in all_data_points[:3]:
                prompt = dp.get("instruction", "")
                inp    = dp.get("input", "")
                if inp:
                    prompt = f"{prompt} {inp}".strip()
                test_prompts.append(prompt or dp.get("prompt", ""))
            test_prompts = [p for p in test_prompts if p]

            if test_prompts:
                compare_result = await self.finetuner.compare_models(
                    prompts=test_prompts,
                    base_model_path=resolved_base,
                    finetuned_adapter_path=final_model,
                )

        # ── 10. Deploy (optional) ─────────────────────────────────────────
        deploy_result = None
        if deploy and train_result.get("success"):
            resolved_base = base_model or "meta-llama/Llama-3.2-3B-Instruct"
            config = HostingConfig(
                model_path=resolved_base,
                adapter_path=train_result["final_model_path"],
                port=deploy_port,
            )
            deploy_result = await self.hoster.deploy_as_mcp(config)

        return {
            "success": True,
            "file_paths": file_paths,
            "per_file_generated": per_file_counts,
            "technique": technique,
            "pipeline_stages": {
                "generated": total_generated,
                "after_cleaning": clean_result["cleaned_count"],
                "after_normalization": norm_result["count"],
                "after_evaluation": eval_result["count"],
                "after_filtering": filter_result["filtered_count"],
            },
            "quality_threshold": quality_threshold,
            "dataset_path": dataset_path,
            "curriculum_training": train_result,
            "comparison": compare_result,
            "deployment": deploy_result,
        }

    async def compare_flat_vs_curriculum(
        self,
        dataset_path: str,
        output_dir: str = "./output/comparison",
        base_model: Optional[str] = None,
        num_epochs_flat: int = 3,
        num_stages: int = 3,
        num_epochs_per_stage: int = 1,
        difficulty_order: str = "easy_first",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        test_data_path: Optional[str] = None,
        test_prompts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run flat SFT and curriculum SFT on the same dataset, then compare.

        Steps:
        1. Load the dataset once
        2. Train flat SFT (single stage, all data)
        3. Train curriculum SFT (staged easy→hard)
        4. Compare both models vs base (if test prompts provided)
        5. Return structured side-by-side results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        resolved_base = base_model or "meta-llama/Llama-3.2-3B-Instruct"

        # 1. Load dataset
        load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
        if not load_result.get("success"):
            return {
                "success": False,
                "error": f"Failed to load dataset: {load_result.get('error')}",
            }

        dataset = load_result["dataset_object"]

        # 2. Flat SFT
        flat_dir = str(output_path / "flat_sft")
        flat_result = await self.finetuner.train_model(
            dataset=dataset,
            output_dir=flat_dir,
            base_model=base_model,
            num_epochs=num_epochs_flat,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # 3. Curriculum SFT
        curriculum_dir = str(output_path / "curriculum_sft")
        curriculum_result = await self.finetuner.train_curriculum_model(
            dataset=dataset,
            output_dir=curriculum_dir,
            base_model=base_model,
            num_stages=num_stages,
            num_epochs_per_stage=num_epochs_per_stage,
            difficulty_order=difficulty_order,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        # 4. Side-by-side comparison (qualitative)
        flat_model = flat_result.get("model_path") if flat_result.get("success") else None
        curriculum_model = (
            curriculum_result.get("final_model_path")
            if curriculum_result.get("success")
            else None
        )

        comparison_result = None
        comparison_prompts = list(test_prompts) if test_prompts else []

        # Extract prompts from test data if provided
        if not comparison_prompts and test_data_path:
            test_load = await self.finetuner.load_dataset_from_file(
                test_data_path, "jsonl"
            )
            if test_load.get("success"):
                test_ds = test_load["dataset_object"]
                for i in range(min(5, len(test_ds))):
                    row = test_ds[i] if hasattr(test_ds, "__getitem__") else {}
                    prompt = row.get("prompt", row.get("instruction", ""))
                    if prompt:
                        comparison_prompts.append(prompt)

        if comparison_prompts and flat_model and curriculum_model:
            flat_vs_base = await self.finetuner.compare_models(
                prompts=comparison_prompts,
                base_model_path=resolved_base,
                finetuned_adapter_path=flat_model,
            )
            curriculum_vs_base = await self.finetuner.compare_models(
                prompts=comparison_prompts,
                base_model_path=resolved_base,
                finetuned_adapter_path=curriculum_model,
            )
            comparison_result = {
                "flat_vs_base": flat_vs_base,
                "curriculum_vs_base": curriculum_vs_base,
            }

        return {
            "success": True,
            "dataset_path": dataset_path,
            "base_model": resolved_base,
            "flat_sft": {
                "training": flat_result,
                "model_path": flat_model,
                "total_epochs": num_epochs_flat,
            },
            "curriculum_sft": {
                "training": curriculum_result,
                "model_path": curriculum_model,
                "num_stages": num_stages,
                "epochs_per_stage": num_epochs_per_stage,
                "difficulty_order": difficulty_order,
            },
            "comparison": comparison_result,
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
