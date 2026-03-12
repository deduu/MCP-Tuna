"""Curriculum Learning Service
================================

Scores a dataset, buckets it by difficulty, and trains stage-by-stage —
each stage initialising from the previous stage's merged weights.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.config import FinetuningConfig


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
        resume_stage: Optional[int] = None,
        extra_callbacks: Optional[List] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Train a model using curriculum learning.

        Steps:
        1. Normalise dataset → list[dict]
        2. Score dataset if score_column is missing
        3. Sort + bucket into num_stages difficulty groups
        4. Train each stage, merging LoRA adapters between stages
        5. Return comprehensive result dict
        """
        t_start = time.perf_counter()
        original_base = base_model or self.config.base_model
        current_base = original_base

        # 1. Normalise to list[dict]
        data = self._normalise_dataset(dataset)
        if not data:
            return {"success": False, "error": "Dataset is empty."}

        # 2. Score if needed
        pre_scored = score_column in data[0]
        if not pre_scored:
            score_result = await self._score_dataset(data, score_column)
            if score_result is None:
                return {
                    "success": False,
                    "error": (
                        f"Dataset has no '{score_column}' column and evaluator pipeline "
                        "is unavailable. Pre-score or install data_evaluator_pipeline."
                    ),
                }
            data = score_result

        # 3. Bucket
        buckets = self._bucket_dataset(data, num_stages, score_column, difficulty_order)

        # 4. Train stage-by-stage
        from .training_service import TrainingService

        training_svc = TrainingService(config=self.config, gpu=None)
        stage_results: List[Dict[str, Any]] = []

        for i, bucket in enumerate(buckets):
            stage_num = i + 1

            # Support resume_stage (1-indexed)
            if resume_stage is not None and stage_num < resume_stage:
                stage_results.append(
                    {
                        "stage": stage_num,
                        "num_examples": len(bucket),
                        "score_range": self._score_range(bucket, score_column),
                        "training_result": {"skipped": True, "reason": "resume_stage"},
                    }
                )
                # When skipping, the base for the next stage is the merged dir from
                # the previous stage (if it exists).
                merged_dir = Path(output_dir) / f"stage_{stage_num}" / "merged"
                if merged_dir.exists():
                    current_base = str(merged_dir)
                continue

            stage_dir = str(Path(output_dir) / f"stage_{stage_num}")
            stage_data = list(self._prepare_training_data(bucket))
            score_range = self._score_range(bucket, score_column)

            # Copy kwargs so pop() calls inside train_model don't
            # mutate the original dict for subsequent stages.
            stage_kwargs = dict(kwargs)
            train_result = await training_svc.train_model(
                dataset=stage_data,
                output_dir=stage_dir,
                base_model=current_base,
                num_epochs=num_epochs_per_stage,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                extra_callbacks=extra_callbacks,
                **stage_kwargs,
            )

            stage_results.append(
                {
                    "stage": stage_num,
                    "num_examples": len(bucket),
                    "score_range": score_range,
                    "training_result": train_result,
                }
            )

            # Merge LoRA adapter after every stage except the last one
            is_last = stage_num == num_stages
            if use_lora and not is_last and train_result.get("success"):
                tokenizer_path = original_base
                try:
                    current_base = await self._merge_lora(
                        stage_dir, original_base, tokenizer_path
                    )
                except Exception as exc:
                    return {
                        "success": False,
                        "error": f"LoRA merge failed after stage {stage_num}: {exc}",
                        "stage_results": stage_results,
                    }
            elif not use_lora and not is_last and train_result.get("success"):
                current_base = stage_dir

        total_seconds = round(time.perf_counter() - t_start, 2)
        final_model_path = str(Path(output_dir) / f"stage_{num_stages}")

        # Check if any stage actually succeeded
        any_success = any(
            sr.get("training_result", {}).get("success")
            for sr in stage_results
        )
        return {
            "success": any_success,
            "final_model_path": final_model_path,
            "base_model": original_base,
            "num_stages": num_stages,
            "num_training_examples": len(data),
            "score_column": score_column,
            "difficulty_order": difficulty_order,
            "stage_results": stage_results,
            "total_training_seconds": total_seconds,
            "pre_scored": pre_scored,
        }

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
        """Yield prompt/response dicts from instruction/input/output or prompt/response format."""
        for item in bucket:
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
        stage_dir: str, original_base: str, tokenizer_path: str
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
                    original_base,
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
