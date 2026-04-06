"""Sequential Training Service
================================

Chains multiple training methods (SFT -> DPO -> GRPO -> KTO) where each
stage's output model_path becomes the next stage's base_model.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentsoul.utils.logger import get_logger
from shared.config import FinetuningConfig
from shared.training_run_artifacts import TrainingRunArtifacts

logger = get_logger(__name__)


class SequentialTrainingService:
    """Chains multiple training techniques sequentially."""

    TECHNIQUE_MAP: Dict[str, str] = {
        "sft": "train_model",
        "dpo": "train_dpo_model",
        "grpo": "train_grpo_model",
        "kto": "train_kto_model",
    }

    def __init__(self, config: FinetuningConfig | None = None) -> None:
        self.config = config or FinetuningConfig()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def train_sequential(
        self,
        stages: List[Dict[str, Any]],
        output_dir: str,
        base_model: Optional[str] = None,
        merge_between_stages: bool = True,
        extra_callbacks: Optional[List] = None,
        run_source: Optional[str] = None,
        job_id: Optional[str] = None,
        artifact_note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run multiple training stages sequentially.

        Each stage specifies:
          - technique: "sft" | "dpo" | "grpo" | "kto"
          - dataset_path: path to JSONL dataset for this stage
          - num_epochs: (optional) defaults to 3
          - Additional technique-specific params (beta, num_generations, etc.)

        Each stage's output model_path becomes the next stage's base_model.
        Between stages with use_lora=True, LoRA adapters are merged into
        full weights so the next stage starts from a complete model.

        Returns:
            Dict with per-stage results, final_model_path, and total timing.
        """
        t_start = time.perf_counter()
        original_base = base_model or self.config.base_model
        current_base = original_base
        run_artifacts = TrainingRunArtifacts(output_dir=output_dir, trainer="sequential")
        stage_results: List[Dict[str, Any]] = []

        run_artifacts.start(
            base_model=original_base,
            adapter_path=None,
            dataset=stages or [],
            dataset_path=None,
            training_config={
                "trainer": "sequential",
                "num_stages": len(stages),
                "merge_between_stages": merge_between_stages,
            },
            run_source=run_source,
            job_id=job_id,
            note=artifact_note,
        )
        run_artifacts.write_json_artifact("stage_plan", "stage_plan.json", {"stages": stages})

        if not stages:
            result = {"success": False, "error": "No stages provided."}
            result["artifacts"] = run_artifacts.complete(
                success=False,
                interrupted=False,
                model_path=None,
                error=result["error"],
                training_time_seconds=time.perf_counter() - t_start,
                metrics={"completed_stages": 0},
            )
            return result

        # Validate all techniques up front
        for i, stage in enumerate(stages):
            technique = stage.get("technique", "").lower()
            if technique not in self.TECHNIQUE_MAP:
                result = {
                    "success": False,
                    "error": (
                        f"Stage {i + 1}: unknown technique '{technique}'. "
                        f"Must be one of: {list(self.TECHNIQUE_MAP.keys())}"
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
            if not stage.get("dataset_path"):
                result = {
                    "success": False,
                    "error": f"Stage {i + 1}: dataset_path is required.",
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

        from .training_service import TrainingService

        training_svc = TrainingService(config=self.config, gpu=None)

        for i, stage in enumerate(stages):
            stage_num = i + 1
            technique = stage["technique"].lower()
            dataset_path = stage["dataset_path"]
            stage_output = stage.get("output_dir") or str(
                Path(output_dir) / f"stage_{stage_num}_{technique}"
            )

            logger.info(
                "sequential_stage_start stage=%d technique=%s dataset=%s base=%s",
                stage_num, technique, dataset_path, current_base,
            )

            # Load dataset
            fmt = "jsonl" if dataset_path.endswith(".jsonl") else "json"
            load_result = await training_svc.load_dataset_from_file(dataset_path, fmt)
            if not load_result.get("success"):
                result = {
                    "success": False,
                    "error": (
                        f"Stage {stage_num}: failed to load dataset: "
                        f"{load_result.get('error')}"
                    ),
                    "stage_results": stage_results,
                }
                run_artifacts.write_json_artifact(
                    "stage_results",
                    "stage_results.json",
                    {"stage_results": stage_results},
                )
                result["artifacts"] = run_artifacts.complete(
                    success=False,
                    interrupted=False,
                    model_path=None,
                    error=result["error"],
                    training_time_seconds=time.perf_counter() - t_start,
                    metrics={"completed_stages": len(stage_results)},
                )
                return result

            dataset = load_result["dataset_object"]

            # Build kwargs for the training method
            train_kwargs = self._build_train_kwargs(
                technique=technique,
                dataset=dataset,
                output_dir=stage_output,
                base_model=current_base,
                stage_config=stage,
            )
            stage_run_source = f"{run_source}.stage" if run_source else "sequential.stage"
            stage_note = f"stage={stage_num}/{len(stages)}; technique={technique}"
            if artifact_note:
                stage_note = f"{stage_note}; parent_note={artifact_note}"
            train_kwargs.update(
                {
                    "dataset_path": dataset_path,
                    "run_source": stage_run_source,
                    "job_id": job_id,
                    "artifact_note": stage_note,
                }
            )
            if extra_callbacks:
                train_kwargs["extra_callbacks"] = extra_callbacks

            # Dispatch to the appropriate training method
            method_name = self.TECHNIQUE_MAP[technique]
            train_method = getattr(training_svc, method_name)
            train_result = await train_method(**train_kwargs)

            stage_results.append(
                {
                    "stage": stage_num,
                    "technique": technique,
                    "dataset_path": dataset_path,
                    "output_dir": stage_output,
                    "training_result": train_result,
                }
            )
            run_artifacts.write_json_artifact(
                "stage_results",
                "stage_results.json",
                {"stage_results": stage_results},
            )

            if not train_result.get("success"):
                result = {
                    "success": False,
                    "error": (
                        f"Stage {stage_num} ({technique}) failed: "
                        f"{train_result.get('error')}"
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

            # Merge LoRA and update current_base for next stage
            is_last = stage_num == len(stages)
            train_config = train_result.get("config")
            if isinstance(train_config, dict) and isinstance(train_config.get("use_lora"), bool):
                use_lora = train_config["use_lora"]
            else:
                use_lora = bool(stage.get("use_lora", technique != "grpo"))

            if merge_between_stages and use_lora and not is_last:
                try:
                    current_base = await self._merge_lora(
                        stage_output, original_base, original_base
                    )
                    logger.info(
                        "sequential_lora_merged stage=%d merged_path=%s",
                        stage_num, current_base,
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
            elif not is_last:
                current_base = stage_output

        total_seconds = round(time.perf_counter() - t_start, 2)
        final_path = stage_results[-1]["output_dir"] if stage_results else output_dir

        logger.info(
            "sequential_training_complete stages=%d seconds=%.2f final=%s",
            len(stages), total_seconds, final_path,
        )

        run_artifacts.write_json_artifact(
            "training_diagnostics",
            "training_diagnostics.json",
            {
                "num_stages": len(stages),
                "completed_stages": len(stage_results),
                "total_training_seconds": total_seconds,
            },
        )
        result = {
            "success": True,
            "final_model_path": final_path,
            "base_model": original_base,
            "num_stages": len(stages),
            "stage_results": stage_results,
            "total_training_seconds": total_seconds,
        }
        result["artifacts"] = run_artifacts.complete(
            success=True,
            interrupted=False,
            model_path=final_path,
            error=None,
            training_time_seconds=total_seconds,
            metrics={"num_stages": len(stages), "completed_stages": len(stage_results)},
        )
        return result

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_train_kwargs(
        technique: str,
        dataset: Any,
        output_dir: str,
        base_model: str,
        stage_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build keyword arguments for the appropriate training method."""
        kwargs: Dict[str, Any] = {
            "dataset": dataset,
            "output_dir": output_dir,
            "base_model": base_model,
            "num_epochs": stage_config.get("num_epochs", 3),
            "load_in_4bit": stage_config.get("load_in_4bit", True),
        }

        if technique == "sft":
            kwargs["use_lora"] = stage_config.get("use_lora", True)
            kwargs["lora_r"] = stage_config.get("lora_r", 8)
            kwargs["lora_alpha"] = stage_config.get("lora_alpha", 16)
            kwargs["completion_only_loss"] = stage_config.get(
                "completion_only_loss", True
            )
        elif technique == "dpo":
            kwargs["beta"] = stage_config.get("beta", 0.1)
            kwargs["use_lora"] = stage_config.get("use_lora", True)
            kwargs["lora_r"] = stage_config.get("lora_r", 8)
        elif technique == "grpo":
            kwargs["num_generations"] = stage_config.get("num_generations", 4)
            if (
                "use_lora" in stage_config
                or "lora_r" in stage_config
                or "lora_alpha" in stage_config
                or "lora_dropout" in stage_config
            ):
                kwargs["use_lora"] = stage_config.get("use_lora", True)
                kwargs["lora_r"] = stage_config.get("lora_r", 8)
                kwargs["lora_alpha"] = stage_config.get("lora_alpha", 16)
                kwargs["lora_dropout"] = stage_config.get("lora_dropout", 0.05)
            if "generation_batch_size" in stage_config:
                kwargs["generation_batch_size"] = stage_config["generation_batch_size"]
            if "steps_per_generation" in stage_config:
                kwargs["steps_per_generation"] = stage_config["steps_per_generation"]
        elif technique == "kto":
            kwargs["beta"] = stage_config.get("beta", 0.1)
            kwargs["use_lora"] = stage_config.get("use_lora", True)
            kwargs["lora_r"] = stage_config.get("lora_r", 8)

        return kwargs

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
