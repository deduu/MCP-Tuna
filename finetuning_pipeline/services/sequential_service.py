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

        if not stages:
            return {"success": False, "error": "No stages provided."}

        # Validate all techniques up front
        for i, stage in enumerate(stages):
            technique = stage.get("technique", "").lower()
            if technique not in self.TECHNIQUE_MAP:
                return {
                    "success": False,
                    "error": (
                        f"Stage {i + 1}: unknown technique '{technique}'. "
                        f"Must be one of: {list(self.TECHNIQUE_MAP.keys())}"
                    ),
                }
            if not stage.get("dataset_path"):
                return {
                    "success": False,
                    "error": f"Stage {i + 1}: dataset_path is required.",
                }

        from .training_service import TrainingService

        training_svc = TrainingService(config=self.config, gpu=None)
        stage_results: List[Dict[str, Any]] = []

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
                return {
                    "success": False,
                    "error": (
                        f"Stage {stage_num}: failed to load dataset: "
                        f"{load_result.get('error')}"
                    ),
                    "stage_results": stage_results,
                }

            dataset = load_result["dataset_object"]

            # Build kwargs for the training method
            train_kwargs = self._build_train_kwargs(
                technique=technique,
                dataset=dataset,
                output_dir=stage_output,
                base_model=current_base,
                stage_config=stage,
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

            if not train_result.get("success"):
                return {
                    "success": False,
                    "error": (
                        f"Stage {stage_num} ({technique}) failed: "
                        f"{train_result.get('error')}"
                    ),
                    "stage_results": stage_results,
                }

            # Merge LoRA and update current_base for next stage
            is_last = stage_num == len(stages)
            use_lora = stage.get("use_lora", True) and technique != "grpo"

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
                    return {
                        "success": False,
                        "error": f"LoRA merge failed after stage {stage_num}: {exc}",
                        "stage_results": stage_results,
                    }
            elif not is_last:
                current_base = stage_output

        total_seconds = round(time.perf_counter() - t_start, 2)
        final_path = stage_results[-1]["output_dir"] if stage_results else output_dir

        logger.info(
            "sequential_training_complete stages=%d seconds=%.2f final=%s",
            len(stages), total_seconds, final_path,
        )

        return {
            "success": True,
            "final_model_path": final_path,
            "base_model": original_base,
            "num_stages": len(stages),
            "stage_results": stage_results,
            "total_training_seconds": total_seconds,
        }

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
            # GRPO trainer manages its own LoRA/architecture
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

        model = AutoModelForCausalLM.from_pretrained(
            original_base,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, stage_dir)
        model = model.merge_and_unload()
        model.save_pretrained(str(merged_dir))

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        tokenizer.save_pretrained(str(merged_dir))

        del model
        torch.cuda.empty_cache()

        return str(merged_dir)
