"""Curriculum Learning Service
================================

Scores a dataset, buckets it by difficulty, and trains stage-by-stage.
LoRA stages can either continue the same adapter across datasets or merge
stage outputs into progressively updated full-model bases.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import FinetuningConfig
from shared.owned_paths import resolve_owned_output_path
from shared.ownership import normalize_ownership_context
from shared.training_run_artifacts import TrainingRunArtifacts


class CurriculumService:
    """Orchestrates multi-stage curriculum fine-tuning."""

    def __init__(self, config: FinetuningConfig = None) -> None:
        self.config = config or FinetuningConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def train_curriculum_model(
        self,
        dataset: Any,
        output_dir: str,
        base_model: Optional[str] = None,
        num_stages: int = 3,
        num_epochs_per_stage: int = 1,
        score_column: str = "weighted_score",
        difficulty_order: str = "easy_first",
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        stage_datasets: Optional[List[Any]] = None,
        stage_training_overrides: Optional[List[Dict[str, Any]]] = None,
        lora_stage_transition: str = "continue_adapter",
        resume_stage: Optional[int] = None,
        extra_callbacks: Optional[List] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Train a model using curriculum learning.

        Steps:
        1. Normalise explicit stage datasets, or normalise one dataset -> list[dict]
        2. Score dataset if score_column is missing
        3. Sort + bucket into num_stages difficulty groups
        4. Train each stage, either continuing one adapter or merging stage outputs
        5. Return comprehensive result dict
        """
        t_start = time.perf_counter()
        original_base = base_model or self.config.base_model
        current_base = original_base
        current_adapter_path: Optional[str] = None
        adapter_base_model = original_base
        run_source = str(kwargs.get("run_source", "") or "").strip() or None
        job_id = str(kwargs.get("job_id", "") or "").strip() or None
        artifact_note = str(kwargs.get("artifact_note", "") or "").strip() or None
        dataset_path = str(kwargs.get("dataset_path", "") or "").strip() or None
        ownership_context = normalize_ownership_context(kwargs.get("ownership"))
        output_dir = str(resolve_owned_output_path(output_dir, ownership_context))
        run_artifacts = TrainingRunArtifacts(
            output_dir=output_dir,
            trainer="curriculum",
            ownership=ownership_context,
        )
        stage_results: List[Dict[str, Any]] = []

        if lora_stage_transition not in {"continue_adapter", "merge_adapter"}:
            result = {
                "success": False,
                "error": (
                    "lora_stage_transition must be 'continue_adapter' or "
                    "'merge_adapter'."
                ),
            }
            result["artifacts"] = run_artifacts.complete(
                success=False,
                interrupted=False,
                model_path=None,
                error=result["error"],
                training_time_seconds=time.perf_counter() - t_start,
                metrics={"completed_stages": 0},
            )
            return result

        stage_source = "bucketed_scores"
        if stage_datasets is not None:
            buckets: List[List[Dict[str, Any]]] = []
            for stage_index, stage_dataset in enumerate(stage_datasets, start=1):
                bucket = self._normalise_dataset(stage_dataset)
                if not bucket:
                    result = {
                        "success": False,
                        "error": f"Stage dataset {stage_index} is empty.",
                    }
                    result["artifacts"] = run_artifacts.complete(
                        success=False,
                        interrupted=False,
                        model_path=None,
                        error=result["error"],
                        training_time_seconds=time.perf_counter() - t_start,
                        metrics={"completed_stages": 0},
                    )
                    return result
                buckets.append(bucket)

            if not buckets:
                result = {"success": False, "error": "Stage datasets are empty."}
                result["artifacts"] = run_artifacts.complete(
                    success=False,
                    interrupted=False,
                    model_path=None,
                    error=result["error"],
                    training_time_seconds=time.perf_counter() - t_start,
                    metrics={"completed_stages": 0},
                )
                return result

            data = [row for bucket in buckets for row in bucket]
            num_stages = len(buckets)
            pre_scored = all(score_column in item for item in data)
            stage_source = "explicit_stage_datasets"
        else:
            data = self._normalise_dataset(dataset)
            if not data:
                result = {"success": False, "error": "Dataset is empty."}
                result["artifacts"] = run_artifacts.complete(
                    success=False,
                    interrupted=False,
                    model_path=None,
                    error=result["error"],
                    training_time_seconds=time.perf_counter() - t_start,
                    metrics={"completed_stages": 0},
                )
                return result

            pre_scored = score_column in data[0]
            if not pre_scored:
                score_result = await self._score_dataset(data, score_column)
                if score_result is None:
                    result = {
                        "success": False,
                        "error": (
                            f"Dataset has no '{score_column}' column and evaluator pipeline "
                            "is unavailable. Pre-score or install data_evaluator_pipeline."
                        ),
                    }
                    result["artifacts"] = run_artifacts.complete(
                        success=False,
                        interrupted=False,
                        model_path=None,
                        error=result["error"],
                        training_time_seconds=time.perf_counter() - t_start,
                        metrics={"completed_stages": 0},
                    )
                    return result
                data = score_result

            buckets = self._bucket_dataset(
                data,
                num_stages,
                score_column,
                difficulty_order,
            )

        run_artifacts.start(
            base_model=original_base,
            adapter_path=None,
            dataset=data,
            dataset_path=dataset_path,
            training_config={
                "trainer": "curriculum",
                "num_stages": num_stages,
                "num_epochs_per_stage": num_epochs_per_stage,
                "score_column": score_column,
                "difficulty_order": difficulty_order,
                "use_lora": use_lora,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_stage_transition": lora_stage_transition,
                "stage_source": stage_source,
            },
            run_source=run_source,
            job_id=job_id,
            note=artifact_note,
            ownership=ownership_context,
        )
        run_artifacts.write_json_artifact(
            "curriculum_plan",
            "curriculum_plan.json",
            {
                "num_stages": num_stages,
                "score_column": score_column,
                "difficulty_order": difficulty_order,
                "stage_source": stage_source,
                "stage_sizes": [len(bucket) for bucket in buckets],
            },
        )

        if resume_stage is not None and resume_stage > num_stages:
            result = {
                "success": False,
                "error": f"resume_stage={resume_stage} exceeds num_stages={num_stages}.",
            }
            result["artifacts"] = run_artifacts.complete(
                success=False,
                interrupted=False,
                model_path=None,
                error=result["error"],
                training_time_seconds=time.perf_counter() - t_start,
                metrics={"completed_stages": 0},
            )
            return result

        stage_training_overrides = list(stage_training_overrides or [])
        if len(stage_training_overrides) > num_stages:
            result = {
                "success": False,
                "error": (
                    "stage_training_overrides cannot have more entries than the "
                    "number of stages."
                ),
            }
            result["artifacts"] = run_artifacts.complete(
                success=False,
                interrupted=False,
                model_path=None,
                error=result["error"],
                training_time_seconds=time.perf_counter() - t_start,
                metrics={"completed_stages": 0},
            )
            return result

        # 4. Train stage-by-stage
        from .training_service import TrainingService

        training_svc = TrainingService(config=self.config, gpu=None)

        for i, bucket in enumerate(buckets):
            stage_num = i + 1
            stage_path = Path(output_dir) / f"stage_{stage_num}"

            # Support resume_stage (1-indexed)
            if resume_stage is not None and stage_num < resume_stage:
                skipped_result: Dict[str, Any] = {
                    "skipped": True,
                    "reason": "resume_stage",
                }
                stage_dir_exists = stage_path.exists()
                merged_dir = stage_path / "merged"
                if use_lora:
                    if lora_stage_transition == "continue_adapter" and stage_dir_exists:
                        current_adapter_path = str(stage_path)
                        adapter_base_model = original_base
                    elif merged_dir.exists():
                        current_base = str(merged_dir)
                        current_adapter_path = None
                        adapter_base_model = current_base
                        skipped_result["merged_model_path"] = str(merged_dir)
                elif stage_dir_exists:
                    current_base = str(stage_path)

                stage_results.append(
                    {
                        "stage": stage_num,
                        "num_examples": len(bucket),
                        "score_range": self._score_range(bucket, score_column),
                        "base_model": current_base,
                        "adapter_path": current_adapter_path,
                        "training_result": skipped_result,
                    }
                )
                continue

            stage_dir = str(stage_path)
            stage_data = list(self._prepare_training_data(bucket))
            score_range = self._score_range(bucket, score_column)
            stage_base_model = current_base
            stage_adapter_path = (
                current_adapter_path
                if use_lora and lora_stage_transition == "continue_adapter"
                else None
            )

            # Copy kwargs so pop() calls inside train_model don't
            # mutate the original dict for subsequent stages.
            stage_kwargs = dict(kwargs)
            raw_override = (
                stage_training_overrides[i] if i < len(stage_training_overrides) else {}
            )
            if raw_override is None:
                raw_override = {}
            if not isinstance(raw_override, dict):
                result = {
                    "success": False,
                    "error": (
                        "Each stage_training_overrides entry must be an object "
                        f"(stage {stage_num})."
                    ),
                    "stage_results": stage_results,
                }
                result["artifacts"] = run_artifacts.complete(
                    success=False,
                    interrupted=False,
                    model_path=None,
                    error=result["error"],
                    training_time_seconds=time.perf_counter() - t_start,
                    metrics={"completed_stages": len(stage_results)},
                )
                return result

            stage_override = dict(raw_override)
            invalid_override_keys = {
                "dataset",
                "output_dir",
                "base_model",
                "adapter_path",
                "stage_datasets",
                "stage_training_overrides",
                "num_stages",
                "resume_stage",
                "score_column",
                "difficulty_order",
                "lora_stage_transition",
                "resume_from_checkpoint",
                "use_lora",
                "lora_r",
                "lora_alpha",
                "lora_dropout",
            } & set(stage_override)
            if invalid_override_keys:
                invalid_keys = ", ".join(sorted(invalid_override_keys))
                result = {
                    "success": False,
                    "error": (
                        "stage_training_overrides cannot override curriculum-managed "
                        f"keys: {invalid_keys}"
                    ),
                    "stage_results": stage_results,
                }
                result["artifacts"] = run_artifacts.complete(
                    success=False,
                    interrupted=False,
                    model_path=None,
                    error=result["error"],
                    training_time_seconds=time.perf_counter() - t_start,
                    metrics={"completed_stages": len(stage_results)},
                )
                return result

            stage_num_epochs = int(
                stage_override.pop(
                    "num_epochs",
                    stage_override.pop("num_epochs_per_stage", num_epochs_per_stage),
                )
            )
            stage_kwargs.update(stage_override)
            stage_run_source = f"{run_source}.stage" if run_source else "curriculum.stage"
            stage_note = f"stage={stage_num}/{num_stages}; source={stage_source}"
            if artifact_note:
                stage_note = f"{stage_note}; parent_note={artifact_note}"
            stage_kwargs.update(
                {
                    "dataset_path": dataset_path,
                    "run_source": stage_run_source,
                    "job_id": job_id,
                    "artifact_note": stage_note,
                    "ownership": ownership_context,
                }
            )
            train_result = await training_svc.train_model(
                dataset=stage_data,
                output_dir=stage_dir,
                base_model=stage_base_model,
                adapter_path=stage_adapter_path,
                num_epochs=stage_num_epochs,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                extra_callbacks=extra_callbacks,
                **stage_kwargs,
            )

            stage_result: Dict[str, Any] = {
                "stage": stage_num,
                "num_examples": len(bucket),
                "score_range": score_range,
                "base_model": stage_base_model,
                "adapter_path": stage_adapter_path,
                "training_result": train_result,
            }

            stage_results.append(
                stage_result
            )
            run_artifacts.write_json_artifact(
                "stage_results",
                "stage_results.json",
                {"stage_results": stage_results},
            )

            is_last = stage_num == num_stages
            if not train_result.get("success"):
                continue

            if use_lora and lora_stage_transition == "continue_adapter":
                current_adapter_path = stage_dir
                adapter_base_model = original_base
            elif use_lora and not is_last:
                tokenizer_path = current_base
                try:
                    merged_model_path = await self._merge_lora(
                        stage_dir,
                        current_base,
                        tokenizer_path,
                    )
                except Exception as exc:
                    result = {
                        "success": False,
                        "error": f"LoRA merge failed after stage {stage_num}: {exc}",
                        "stage_results": stage_results,
                    }
                    result["artifacts"] = run_artifacts.complete(
                        success=False,
                        interrupted=False,
                        model_path=None,
                        error=result["error"],
                        training_time_seconds=time.perf_counter() - t_start,
                        metrics={"completed_stages": len(stage_results)},
                    )
                    return result
                current_base = merged_model_path
                current_adapter_path = None
                adapter_base_model = current_base
                stage_result["merged_model_path"] = merged_model_path
            elif use_lora and is_last:
                adapter_base_model = current_base
            elif not use_lora and not is_last:
                current_base = stage_dir

        total_seconds = round(time.perf_counter() - t_start, 2)
        final_model_path = str(Path(output_dir) / f"stage_{num_stages}")

        # Check if any stage actually succeeded
        any_success = any(
            sr.get("training_result", {}).get("success")
            for sr in stage_results
        )
        run_artifacts.write_json_artifact(
            "training_diagnostics",
            "training_diagnostics.json",
            {
                "num_stages": num_stages,
                "completed_stages": len(stage_results),
                "successful_stages": sum(
                    1
                    for stage in stage_results
                    if stage.get("training_result", {}).get("success")
                ),
                "total_training_seconds": total_seconds,
            },
        )
        result = {
            "success": any_success,
            "final_model_path": final_model_path,
            "base_model": original_base,
            "num_stages": num_stages,
            "num_training_examples": len(data),
            "score_column": score_column,
            "difficulty_order": difficulty_order,
            "adapter_base_model": adapter_base_model if use_lora else None,
            "lora_stage_transition": lora_stage_transition,
            "stage_source": stage_source,
            "stage_results": stage_results,
            "total_training_seconds": total_seconds,
            "pre_scored": pre_scored,
        }
        result["artifacts"] = run_artifacts.complete(
            success=any_success,
            interrupted=False,
            model_path=final_model_path if any_success else None,
            error=None if any_success else "All curriculum stages failed.",
            training_time_seconds=total_seconds,
            metrics={
                "num_stages": num_stages,
                "completed_stages": len(stage_results),
                "successful_stages": sum(
                    1
                    for stage in stage_results
                    if stage.get("training_result", {}).get("success")
                ),
            },
        )
        return result

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_dataset(dataset: Any) -> List[Dict[str, Any]]:
        """Convert HF Dataset or list-of-dicts to plain list[dict]."""
        if isinstance(dataset, list):
            return dataset
        # HuggingFace Dataset
        try:
            return [dict(row) for row in dataset]
        except Exception:
            return []

    async def _score_dataset(
        self, data: List[Dict[str, Any]], score_column: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Lazy-import EvaluatorService and score the dataset.

        Returns the enriched list[dict] on success, or None if the evaluator
        pipeline is unavailable.
        """
        try:
            from data_evaluator_pipeline.services.pipeline_service import EvaluatorService  # noqa: PLC0415
        except ImportError:
            return None

        try:
            evaluator = EvaluatorService()
            result = await evaluator.evaluate_dataset(data)
            return result.get("data_points", data)
        except Exception:
            return None

    @staticmethod
    def _bucket_dataset(
        data: List[Dict[str, Any]],
        num_stages: int,
        score_column: str,
        difficulty_order: str,
    ) -> List[List[Dict[str, Any]]]:
        """Sort by score_column and split into num_stages even buckets."""
        sorted_data = sorted(
            data,
            key=lambda x: x.get(score_column, 0.0),
            reverse=(difficulty_order == "hard_first"),
        )
        n = len(sorted_data)
        return [
            sorted_data[i * n // num_stages : (i + 1) * n // num_stages]
            for i in range(num_stages)
        ]

    @staticmethod
    def _prepare_training_data(
        bucket: List[Dict[str, Any]],
    ):
        """Yield training rows while preserving supported dataset schemas."""
        for item in bucket:
            if {"system", "user", "assistant"}.issubset(item):
                yield {
                    "system": item.get("system", ""),
                    "user": item.get("user", ""),
                    "assistant": item.get("assistant", ""),
                }
                continue
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            prompt = f"{instruction} {inp}".strip() or item.get("prompt", "")
            response = item.get("output") or item.get("response", "")
            yield {"prompt": prompt, "response": response}

    @staticmethod
    def _score_range(
        bucket: List[Dict[str, Any]], score_column: str
    ) -> List[float]:
        """Return [min_score, max_score] for a bucket."""
        scores = [float(item.get(score_column, 0.0)) for item in bucket]
        if not scores:
            return [0.0, 0.0]
        return [round(min(scores), 4), round(max(scores), 4)]

    @staticmethod
    async def _merge_lora(
        stage_dir: str, base_model_path: str, tokenizer_path: str
    ) -> str:
        """Merge a LoRA adapter into full weights and save to stage_dir/merged/."""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415
        import torch  # noqa: PLC0415

        merged_dir = Path(stage_dir) / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)

        load_attempts = []
        if torch.cuda.is_available():
            # Avoid device_map="auto" here: PEFT/Accelerate can mis-handle
            # partially offloaded models during adapter loading on Windows.
            load_attempts.append(
                {
                    "torch_dtype": torch.float16,
                    "device_map": {"": 0},
                    "low_cpu_mem_usage": True,
                }
            )
        load_attempts.append(
            {
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
            }
        )

        last_exc = None
        for load_kwargs in load_attempts:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    **load_kwargs,
                )
                break
            except Exception as exc:
                last_exc = exc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            raise last_exc
        model = PeftModel.from_pretrained(model, stage_dir)
        model = model.merge_and_unload()
        model.save_pretrained(str(merged_dir))

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(str(merged_dir))

        del model
        torch.cuda.empty_cache()

        return str(merged_dir)
