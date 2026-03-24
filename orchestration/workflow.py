"""
End-to-end pipeline orchestrator.

Composes: Extract → Generate → Clean → Normalize → Evaluate → Filter → Train → Test → Host
"""

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from shared.async_utils import call_maybe_async

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

    @staticmethod
    def _resolve_file_paths(
        file_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
    ) -> List[str]:
        resolved: List[str] = []
        if file_path and file_path.strip():
            resolved.append(file_path.strip())
        if file_paths:
            resolved.extend(
                fp.strip() for fp in file_paths
                if isinstance(fp, str) and fp.strip()
            )

        unique: List[str] = []
        seen = set()
        for fp in resolved:
            if fp not in seen:
                seen.add(fp)
                unique.append(fp)

        if not unique:
            raise ValueError("Provide file_path or file_paths")
        return unique

    @staticmethod
    def _cancelled_result() -> Dict[str, Any]:
        return {"success": False, "error": "Pipeline cancelled"}

    @staticmethod
    def _is_cancelled(cancel_event: Optional[threading.Event]) -> bool:
        return bool(cancel_event and cancel_event.is_set())

    @staticmethod
    def _training_output_path(train_result: Dict[str, Any]) -> Optional[str]:
        return train_result.get("model_path") or train_result.get("final_model_path")

    @staticmethod
    def _collect_generation_errors(
        file_path: str,
        gen_result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        for page_error in gen_result.get("page_errors") or []:
            if not isinstance(page_error, dict):
                continue
            errors.append({
                "file_path": file_path,
                "file_name": Path(file_path).name,
                **page_error,
            })
        return errors

    @staticmethod
    def _format_zero_generation_error(
        generation_errors: List[Dict[str, Any]],
    ) -> str:
        if not generation_errors:
            return "No data points generated from provided files."

        first_error = generation_errors[0]
        file_name = first_error.get("file_name") or Path(
            str(first_error.get("file_path", "provided file"))
        ).name
        page = first_error.get("page")
        error = str(first_error.get("error", "Unknown generation error"))
        location = f"{file_name} page {page}" if page else file_name
        extra_failures = len(generation_errors) - 1
        extra_suffix = f" ({extra_failures} more page failures)" if extra_failures > 0 else ""
        return (
            f"No data points generated from provided files. "
            f"First page failure in {location}: {error}{extra_suffix}"
        )

    @classmethod
    def _training_uses_adapter(cls, train_result: Dict[str, Any]) -> bool:
        config = train_result.get("config")
        if isinstance(config, dict):
            use_lora = config.get("use_lora")
            if isinstance(use_lora, bool):
                return use_lora
            if config.get("trainer") == "grpo":
                return False

        stage_results = train_result.get("stage_results")
        if isinstance(stage_results, list) and stage_results:
            last_stage = stage_results[-1]
            if isinstance(last_stage, dict):
                nested = last_stage.get("training_result")
                if isinstance(nested, dict):
                    return cls._training_uses_adapter(nested)

        return True

    @classmethod
    def _deployment_model_args(
        cls,
        train_result: Dict[str, Any],
        base_model: Optional[str],
        default_base_model: str,
    ) -> Dict[str, Optional[str]]:
        trained_model_path = cls._training_output_path(train_result)
        if not trained_model_path:
            return {"model_path": None, "adapter_path": None}

        if cls._training_uses_adapter(train_result):
            return {
                "model_path": base_model or default_base_model,
                "adapter_path": trained_model_path,
            }

        return {
            "model_path": trained_model_path,
            "adapter_path": None,
        }

    async def _report_stage(
        self,
        progress_callback: Optional[Callable[..., Any]],
        stage_name: str,
        current_step: int,
        total_steps: int,
        status_message: Optional[str] = None,
        stage_current: Optional[int] = None,
        stage_total: Optional[int] = None,
        stage_unit: Optional[str] = None,
    ) -> None:
        if progress_callback is None:
            return
        payload = dict(
            current_stage=stage_name,
            current_step=current_step,
            max_steps=total_steps,
            percent_complete=round(current_step / max(total_steps, 1) * 100, 1),
        )
        if status_message is not None:
            payload["status_message"] = status_message
        if stage_current is not None:
            payload["stage_current"] = stage_current
        if stage_total is not None:
            payload["stage_total"] = stage_total
        if stage_unit is not None:
            payload["stage_unit"] = stage_unit
        await call_maybe_async(
            progress_callback,
            **payload,
        )

    async def _run_with_stage_heartbeat(
        self,
        coro: Any,
        *,
        progress_callback: Optional[Callable[..., Any]],
        cancel_event: Optional[threading.Event],
        stage_name: str,
        current_step: int,
        total_steps: int,
        status_message: str,
        stage_current: Optional[int] = None,
        stage_total: Optional[int] = None,
        stage_unit: Optional[str] = None,
        heartbeat_interval: float = 2.0,
    ) -> Any:
        task = asyncio.create_task(coro)
        started = time.monotonic()

        while True:
            done, _ = await asyncio.wait({task}, timeout=heartbeat_interval)
            if task in done:
                return await task

            if self._is_cancelled(cancel_event):
                task.cancel()
                return self._cancelled_result()

            elapsed = max(1, int(time.monotonic() - started))
            await self._report_stage(
                progress_callback,
                stage_name,
                current_step,
                total_steps,
                status_message=f"{status_message} ({elapsed}s elapsed)",
                stage_current=stage_current,
                stage_total=stage_total,
                stage_unit=stage_unit,
            )

    async def _generate_and_filter(
        self,
        file_paths: List[str],
        technique: str,
        quality_threshold: float,
        custom_template: Optional[str] = None,
        progress_callback: Optional[Callable[..., Any]] = None,
        cancel_event: Optional[threading.Event] = None,
        stage_offset: int = 0,
        total_stages: int = 5,
    ) -> Dict[str, Any]:
        await self._report_stage(
            progress_callback,
            "generate",
            stage_offset + 1,
            total_stages,
            status_message=f"Preparing to generate from {len(file_paths)} file(s)",
            stage_current=0,
            stage_total=len(file_paths),
            stage_unit="file",
        )

        all_data_points: List[Dict[str, Any]] = []
        per_file_counts: Dict[str, int] = {}
        generation_errors: List[Dict[str, Any]] = []

        for index, fp in enumerate(file_paths, start=1):
            if self._is_cancelled(cancel_event):
                return self._cancelled_result()

            await self._report_stage(
                progress_callback,
                "generate",
                stage_offset + 1,
                total_stages,
                status_message=f"Generating from file {index} of {len(file_paths)}: {Path(fp).name}",
                stage_current=index,
                stage_total=len(file_paths),
                stage_unit="file",
            )

            gen_result = await self._run_with_stage_heartbeat(
                self.generator.generate_from_document(
                    technique=technique,
                    file_path=fp,
                    custom_template=custom_template,
                ),
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                stage_name="generate",
                current_step=stage_offset + 1,
                total_steps=total_stages,
                status_message=f"Generating from file {index} of {len(file_paths)}: {Path(fp).name}",
                stage_current=index,
                stage_total=len(file_paths),
                stage_unit="file",
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
            generation_errors.extend(self._collect_generation_errors(fp, gen_result))

        total_generated = len(all_data_points)
        if not all_data_points:
            return {
                "success": False,
                "error": self._format_zero_generation_error(generation_errors),
                "generation_errors": generation_errors,
            }

        if self._is_cancelled(cancel_event):
            return self._cancelled_result()

        await self._report_stage(
            progress_callback,
            "clean",
            stage_offset + 2,
            total_stages,
            status_message=f"Cleaning {total_generated} generated record(s)",
            stage_current=total_generated,
            stage_total=total_generated,
            stage_unit="record",
        )
        clean_result = await self._run_with_stage_heartbeat(
            self.cleaner.clean_dataset(all_data_points, CleaningConfig()),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            stage_name="clean",
            current_step=stage_offset + 2,
            total_steps=total_stages,
            status_message=f"Cleaning {total_generated} generated record(s)",
            stage_current=total_generated,
            stage_total=total_generated,
            stage_unit="record",
        )
        data_points = clean_result["data_points"]

        if self._is_cancelled(cancel_event):
            return self._cancelled_result()

        await self._report_stage(
            progress_callback,
            "normalize",
            stage_offset + 3,
            total_stages,
            status_message=f"Normalizing {clean_result['cleaned_count']} record(s) to {technique}",
            stage_current=clean_result["cleaned_count"],
            stage_total=clean_result["cleaned_count"],
            stage_unit="record",
        )
        norm_result = await self._run_with_stage_heartbeat(
            self.normalizer.normalize_dataset(
                data_points, NormalizationConfig(target_format=technique),
            ),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            stage_name="normalize",
            current_step=stage_offset + 3,
            total_steps=total_stages,
            status_message=f"Normalizing {clean_result['cleaned_count']} record(s) to {technique}",
            stage_current=clean_result["cleaned_count"],
            stage_total=clean_result["cleaned_count"],
            stage_unit="record",
        )
        data_points = norm_result["data_points"]

        if self._is_cancelled(cancel_event):
            return self._cancelled_result()

        await self._report_stage(
            progress_callback,
            "evaluate",
            stage_offset + 4,
            total_stages,
            status_message=f"Evaluating {norm_result['count']} normalized record(s)",
            stage_current=norm_result["count"],
            stage_total=norm_result["count"],
            stage_unit="record",
        )
        eval_result = await self._run_with_stage_heartbeat(
            self.evaluator.evaluate_dataset(data_points),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            stage_name="evaluate",
            current_step=stage_offset + 4,
            total_steps=total_stages,
            status_message=f"Evaluating {norm_result['count']} normalized record(s)",
            stage_current=norm_result["count"],
            stage_total=norm_result["count"],
            stage_unit="record",
        )
        if not eval_result.get("success"):
            return eval_result
        data_points = eval_result["data_points"]

        if self._is_cancelled(cancel_event):
            return self._cancelled_result()

        await self._report_stage(
            progress_callback,
            "filter",
            stage_offset + 5,
            total_stages,
            status_message=(
                f"Filtering {eval_result['count']} evaluated record(s) at threshold {quality_threshold:.2f}"
            ),
            stage_current=eval_result["count"],
            stage_total=eval_result["count"],
            stage_unit="record",
        )
        filter_result = await self._run_with_stage_heartbeat(
            self.evaluator.filter_by_quality(data_points, quality_threshold),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            stage_name="filter",
            current_step=stage_offset + 5,
            total_steps=total_stages,
            status_message=(
                f"Filtering {eval_result['count']} evaluated record(s) at threshold {quality_threshold:.2f}"
            ),
            stage_current=eval_result["count"],
            stage_total=eval_result["count"],
            stage_unit="record",
        )
        if not filter_result.get("success"):
            return filter_result
        data_points = filter_result["data_points"]

        stats = await self.evaluator.analyze_statistics(data_points)
        result: Dict[str, Any] = {
            "success": True,
            "technique": technique,
            "file_paths": file_paths,
            "per_file_generated": per_file_counts,
            "pipeline_stages": {
                "generated": total_generated,
                "after_cleaning": clean_result["cleaned_count"],
                "after_normalization": norm_result["count"],
                "after_evaluation": eval_result["count"],
                "after_filtering": filter_result["filtered_count"],
            },
            "quality_threshold": quality_threshold,
            "statistics": stats.get("statistics", {}),
            "data_points": data_points,
        }
        if len(file_paths) == 1:
            result["file_path"] = file_paths[0]
        return result

    async def generate_and_evaluate(
        self,
        file_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        technique: str = "sft",
        quality_threshold: float = 0.7,
        custom_template: Optional[str] = None,
        progress_callback: Optional[Callable[..., Any]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        """Extract → Generate → Clean → Normalize → Evaluate → Filter"""

        resolved_paths = self._resolve_file_paths(file_path=file_path, file_paths=file_paths)
        return await self._generate_and_filter(
            file_paths=resolved_paths,
            technique=technique,
            quality_threshold=quality_threshold,
            custom_template=custom_template,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    async def full_pipeline(
        self,
        file_path: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        technique: str = "sft",
        output_dir: str = "./output",
        quality_threshold: float = 0.7,
        base_model: Optional[str] = None,
        num_epochs: int = 3,
        use_lora: bool = True,
        push_to_hub: Optional[str] = None,
        deploy: bool = False,
        deploy_port: int = 8001,
        quantization: Optional[str] = None,
        progress_callback: Optional[Callable[..., Any]] = None,
        cancel_event: Optional[threading.Event] = None,
        extra_callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Extract → Generate → Clean → Normalize → Evaluate → Filter → Train → Test → Host"""

        resolved_paths = self._resolve_file_paths(file_path=file_path, file_paths=file_paths)

        # Stages 1-5: generate + evaluate
        ge_result = await self._generate_and_filter(
            file_paths=resolved_paths,
            technique=technique,
            quality_threshold=quality_threshold,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            stage_offset=0,
            total_stages=9,
        )
        if not ge_result.get("success"):
            return ge_result

        data_points = ge_result["data_points"]

        if self._is_cancelled(cancel_event):
            return self._cancelled_result()

        # 6) Export dataset for finetuning
        await self._report_stage(
            progress_callback,
            "export",
            6,
            9,
            status_message=f"Exporting {len(data_points)} filtered record(s) to dataset.jsonl",
            stage_current=len(data_points),
            stage_total=len(data_points),
            stage_unit="record",
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        dataset_path = str(output_path / "dataset.jsonl")

        export_result = await self._run_with_stage_heartbeat(
            self.generator.export_dataset(
                data_points=data_points,
                output_path=dataset_path,
                format="jsonl",
            ),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            stage_name="export",
            current_step=6,
            total_steps=9,
            status_message=f"Exporting {len(data_points)} filtered record(s) to dataset.jsonl",
            stage_current=len(data_points),
            stage_total=len(data_points),
            stage_unit="record",
        )
        if isinstance(export_result, dict) and not export_result.get("success", True):
            return {**ge_result, "export": export_result}

        if self._is_cancelled(cancel_event):
            return self._cancelled_result()

        # 7) Train
        await self._report_stage(
            progress_callback,
            "train",
            7,
            9,
            status_message=f"Loading dataset and starting training for {num_epochs} epoch(s)",
        )
        load_result = await self._run_with_stage_heartbeat(
            self.finetuner.load_dataset_from_file(dataset_path, "jsonl"),
            progress_callback=progress_callback,
            cancel_event=cancel_event,
            stage_name="train",
            current_step=7,
            total_steps=9,
            status_message="Loading exported dataset and initializing training",
        )
        if not load_result.get("success"):
            return {**ge_result, "training": load_result}

        train_result = await self.finetuner.train_model(
            dataset=load_result["dataset_object"],
            output_dir=str(output_path / "model"),
            base_model=base_model,
            num_epochs=num_epochs,
            use_lora=use_lora,
            push_to_hub=push_to_hub,
            extra_callbacks=extra_callbacks,
        )

        if not train_result.get("success"):
            return {
                "success": False,
                "error": train_result.get("error", "Training failed"),
                "file_paths": resolved_paths,
                **({"file_path": resolved_paths[0]} if len(resolved_paths) == 1 else {}),
                "technique": technique,
                "pipeline_stages": ge_result["pipeline_stages"],
                "quality_threshold": quality_threshold,
                "dataset_path": dataset_path,
                "training": train_result,
            }

        if self._is_cancelled(cancel_event):
            return self._cancelled_result()

        # 8) Test inference
        await self._report_stage(
            progress_callback,
            "test",
            8,
            9,
            status_message="Running post-training inference smoke test",
        )
        test_result = None
        if train_result.get("success"):
            model_args = self._deployment_model_args(
                train_result,
                base_model=base_model,
                default_base_model="meta-llama/Llama-3.2-3B-Instruct",
            )
            test_prompts = []
            for dp in data_points[:3]:
                test_prompts.append(dp.get("instruction", dp.get("prompt", "")))

            if test_prompts and model_args["model_path"]:
                test_result = await self._run_with_stage_heartbeat(
                    self.finetuner.run_inference(
                        prompts=test_prompts,
                        model_path=model_args["model_path"],
                        adapter_path=model_args["adapter_path"],
                    ),
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                    stage_name="test",
                    current_step=8,
                    total_steps=9,
                    status_message="Running post-training inference smoke test",
                    stage_current=len(test_prompts),
                    stage_total=len(test_prompts),
                    stage_unit="prompt",
                )

        if self._is_cancelled(cancel_event):
            return self._cancelled_result()

        # 9) Deploy (optional)
        await self._report_stage(
            progress_callback,
            "deploy",
            9,
            9,
            status_message=(
                f"Deploying trained model on port {deploy_port}" if deploy else "Skipping deployment"
            ),
        )
        deploy_result = None
        if deploy and train_result.get("success"):
            model_args = self._deployment_model_args(
                train_result,
                base_model=base_model,
                default_base_model="meta-llama/Llama-3.2-3B-Instruct",
            )
            config = HostingConfig(
                model_path=model_args["model_path"],
                adapter_path=model_args["adapter_path"],
                port=deploy_port,
                quantization=quantization,
            )
            deploy_result = await self._run_with_stage_heartbeat(
                self.hoster.deploy_as_mcp(config),
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                stage_name="deploy",
                current_step=9,
                total_steps=9,
                status_message=f"Deploying trained model on port {deploy_port}",
            )
            if not deploy_result.get("success", True):
                return {
                    "success": False,
                    "error": deploy_result.get("error", "Deployment failed"),
                    "file_paths": resolved_paths,
                    **({"file_path": resolved_paths[0]} if len(resolved_paths) == 1 else {}),
                    "technique": technique,
                    "pipeline_stages": ge_result["pipeline_stages"],
                    "quality_threshold": quality_threshold,
                    "dataset_path": dataset_path,
                    "training": train_result,
                    "testing": test_result,
                    "deployment": deploy_result,
                }

        return {
            "success": True,
            "technique": technique,
            "file_paths": resolved_paths,
            **({"file_path": resolved_paths[0]} if len(resolved_paths) == 1 else {}),
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
        quantization: Optional[str] = None,
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
        generation_errors: List[Dict[str, Any]] = []

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
            generation_errors.extend(self._collect_generation_errors(fp, gen_result))

        total_generated = len(all_data_points)
        if not all_data_points:
            return {
                "success": False,
                "error": self._format_zero_generation_error(generation_errors),
                "generation_errors": generation_errors,
            }

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
            model_args = self._deployment_model_args(
                train_result,
                base_model=base_model,
                default_base_model="meta-llama/Llama-3.2-3B-Instruct",
            )
            config = HostingConfig(
                model_path=model_args["model_path"],
                adapter_path=model_args["adapter_path"],
                port=deploy_port,
                quantization=quantization,
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
        score_column: str = "weighted_score",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        load_in_4bit: bool = True,
        learning_rate: float = 2e-4,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        gradient_checkpointing: bool = False,
        max_seq_length: int = 2048,
        warmup_ratio: float = 0.0,
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
        common_training_kwargs = {
            "base_model": base_model,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "load_in_4bit": load_in_4bit,
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "max_seq_length": max_seq_length,
            "warmup_ratio": warmup_ratio,
        }

        # 2. Flat SFT
        flat_dir = str(output_path / "flat_sft")
        flat_result = await self.finetuner.train_model(
            dataset=dataset,
            output_dir=flat_dir,
            num_epochs=num_epochs_flat,
            **common_training_kwargs,
        )

        # 3. Curriculum SFT
        curriculum_dir = str(output_path / "curriculum_sft")
        curriculum_result = await self.finetuner.train_curriculum_model(
            dataset=dataset,
            output_dir=curriculum_dir,
            num_stages=num_stages,
            num_epochs_per_stage=num_epochs_per_stage,
            difficulty_order=difficulty_order,
            score_column=score_column,
            **common_training_kwargs,
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
                "score_column": score_column,
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
        quantization: Optional[str] = None,
        training_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Full orchestration training pipeline:
        Problems -> Trajectories -> Rewards -> Training data -> Fine-tune -> Deploy
        """
        if self.orchestration_data_service is None:
            return {"success": False, "error": "OrchestrationDataService not configured"}

        ods = self.orchestration_data_service
        generated_problems: List[Dict[str, Any]] = []
        collected: List[Dict[str, Any]] = []

        if training_data is None:
            # 1) Generate synthetic orchestration problems
            generated_problems = await ods.generate_problems(
                domain_description=domain_description,
                num_problems=num_problems,
                tool_descriptions=tool_descriptions,
            )

            # 2) Collect trajectories (run agent N times per problem)
            collected = await ods.collect_trajectories(
                problems=generated_problems, agent=agent, n_per_problem=n_per_problem,
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

        # 5) Fine-tune orchestrator with the correct objective for the exported format
        train_output_dir = str(output_path / "model")
        if output_format == "sft":
            train_result = await self.finetuner.train_model(
                dataset=training_data,
                output_dir=train_output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
            )
        elif output_format == "dpo":
            train_result = await self.finetuner.train_dpo_model(
                dataset=training_data,
                output_dir=train_output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
            )
        elif output_format == "grpo":
            train_result = await self.finetuner.train_grpo_model(
                dataset=training_data,
                output_dir=train_output_dir,
                base_model=base_model,
                num_epochs=num_epochs,
            )
        else:
            return {
                "success": False,
                "error": f"Unsupported orchestration output_format: {output_format}",
                "dataset_path": dataset_path,
            }

        # 6) Optionally deploy
        deploy_result = None
        if deploy and train_result.get("success"):
            model_args = self._deployment_model_args(
                train_result,
                base_model=base_model,
                default_base_model="meta-llama/Llama-3.2-3B-Instruct",
            )
            config = HostingConfig(
                model_path=model_args["model_path"],
                adapter_path=model_args["adapter_path"],
                port=deploy_port,
                quantization=quantization,
            )
            deploy_result = await self.hoster.deploy_as_mcp(config)

        return {
            "success": True,
            "domain": domain_description,
            "problems_generated": len(generated_problems),
            "trajectories_collected": sum(len(c["trajectories"]) for c in collected),
            "training_examples": len(training_data),
            "output_format": output_format,
            "dataset_path": dataset_path,
            "training": train_result,
            "deployment": deploy_result,
            "used_provided_training_data": training_data is not None and not collected and not generated_problems,
        }
