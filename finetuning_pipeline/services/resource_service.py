"""Resource Service
===================

System resource inventory and training feasibility estimation.
Provides GPU/RAM/disk introspection and VRAM preflight checks so
agents can verify hardware before launching training pipelines.
"""
from __future__ import annotations

import os
import re
import shutil
from typing import Any, Dict, List, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]


class ResourceService:
    """System resource inventory and training feasibility estimation."""

    # Known model sizes (parameters in billions), sourced from HuggingFace
    MODEL_PARAMS: Dict[str, float] = {
        # Llama 3.2
        "meta-llama/Llama-3.2-1B": 1.24,
        "meta-llama/Llama-3.2-1B-Instruct": 1.24,
        "meta-llama/Llama-3.2-3B": 3.21,
        "meta-llama/Llama-3.2-3B-Instruct": 3.21,
        # Llama 3.1
        "meta-llama/Llama-3.1-8B": 8.03,
        "meta-llama/Llama-3.1-8B-Instruct": 8.03,
        "meta-llama/Llama-3.1-70B": 70.6,
        "meta-llama/Llama-3.1-70B-Instruct": 70.6,
        # Llama 3
        "meta-llama/Meta-Llama-3-8B": 8.03,
        "meta-llama/Meta-Llama-3-8B-Instruct": 8.03,
        "meta-llama/Meta-Llama-3-70B": 70.6,
        "meta-llama/Meta-Llama-3-70B-Instruct": 70.6,
        # Mistral
        "mistralai/Mistral-7B-v0.1": 7.24,
        "mistralai/Mistral-7B-Instruct-v0.2": 7.24,
        "mistralai/Mixtral-8x7B-v0.1": 46.7,
        # Phi
        "microsoft/phi-2": 2.78,
        "microsoft/Phi-3-mini-4k-instruct": 3.82,
        # Gemma
        "google/gemma-2b": 2.51,
        "google/gemma-7b": 8.54,
        "google/gemma-2-9b": 9.24,
        "google/gemma-2-27b": 27.2,
        # Qwen
        "Qwen/Qwen2-0.5B": 0.49,
        "Qwen/Qwen2-1.5B": 1.54,
        "Qwen/Qwen2-7B": 7.07,
        "Qwen/Qwen2-72B": 72.7,
    }

    # Bytes per parameter by quantization level
    _BYTES_PER_PARAM: Dict[str, float] = {
        "4bit": 0.55,   # 4 bits + overhead (group scales, etc.)
        "8bit": 1.1,    # 8 bits + overhead
        "none": 2.1,    # FP16 / BF16
        "fp16": 2.1,
        "bf16": 2.1,
        "fp32": 4.2,
    }

    def __init__(self, output_dir: str = "output") -> None:
        self._output_dir = output_dir

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def check_resources(self) -> Dict[str, Any]:
        """Return current GPU, RAM, and disk status."""
        return {
            "gpu": self._get_gpu_info(),
            "ram": self._get_ram_info(),
            "disk": self._get_disk_info(),
        }

    def preflight_check(
        self,
        model_name: str,
        quantization: str = "4bit",
        batch_size: int = 1,
        max_seq_length: int = 512,
        technique: str = "sft",
        use_lora: bool = True,
        lora_r: int = 8,
        gradient_checkpointing: bool = False,
    ) -> Dict[str, Any]:
        """Estimate VRAM requirements and check feasibility."""
        warnings: List[str] = []
        recommendations: List[str] = []

        # Resolve parameter count
        params_b = self._get_model_params(model_name)
        if params_b is None:
            # Try to infer from model name (e.g. "some-org/model-7B")
            params_b = self._infer_params_from_name(model_name)
            if params_b is not None:
                warnings.append(
                    f"Model '{model_name}' not in known list. "
                    f"Estimated {params_b:.1f}B params from name pattern."
                )
            else:
                params_b = 7.0  # Default fallback
                warnings.append(
                    f"Model '{model_name}' not in known list and no size pattern found. "
                    f"Defaulting to {params_b}B params estimate."
                )

        # Estimate VRAM breakdown
        breakdown = self._estimate_vram(
            params_b=params_b,
            quantization=quantization,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            use_lora=use_lora,
            lora_r=lora_r,
            gradient_checkpointing=gradient_checkpointing,
        )
        estimated_total = round(sum(breakdown.values()), 2)

        # Check GPU availability
        gpu_available = torch is not None and torch.cuda.is_available()
        if not gpu_available:
            return {
                "can_run": False,
                "estimated_vram_gb": estimated_total,
                "available_vram_gb": 0.0,
                "headroom_gb": -estimated_total,
                "breakdown": breakdown,
                "recommendations": recommendations,
                "warnings": warnings + ["No CUDA GPU detected. Training requires a GPU."],
            }

        # Get available VRAM
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / 1024**3
        used_vram = torch.cuda.memory_allocated(0) / 1024**3
        available_vram = round(total_vram - used_vram, 2)

        can_run = estimated_total <= available_vram
        headroom = round(available_vram - estimated_total, 2)

        # Generate recommendations
        if not gradient_checkpointing and estimated_total > available_vram * 0.7:
            recommendations.append(
                "gradient_checkpointing=True recommended to save ~30% activation memory"
            )
        if quantization not in ("4bit",) and estimated_total > available_vram * 0.8:
            recommendations.append(
                "Use 4-bit quantization (quantization='4bit') to reduce model footprint"
            )
        if use_lora and lora_r > 16:
            recommendations.append(
                f"lora_r={lora_r} is high. Consider lora_r=8 or 16 to reduce LoRA memory"
            )
        if batch_size > 1 and estimated_total > available_vram * 0.8:
            recommendations.append(
                "Reduce batch_size to 1 and increase gradient_accumulation_steps instead"
            )
        if not can_run:
            recommendations.append(
                "Use paged_adamw_8bit optimizer to reduce optimizer footprint"
            )

        if headroom < 0.5 and can_run:
            warnings.append(
                f"Only {headroom:.1f} GB headroom. May OOM under load. "
                "Consider reducing max_seq_length or batch_size."
            )

        return {
            "can_run": can_run,
            "estimated_vram_gb": estimated_total,
            "available_vram_gb": available_vram,
            "headroom_gb": headroom,
            "breakdown": breakdown,
            "recommendations": recommendations,
            "warnings": warnings,
        }

    def prescribe(
        self,
        model_name: str,
        dataset_row_count: int,
        dataset_avg_text_length: int,
        technique: str = "sft",
    ) -> Dict[str, Any]:
        """Recommend optimal training configuration from resources + dataset.

        Inverts :meth:`preflight_check`: instead of *validating* a config,
        this method *generates* one that fits the user's hardware.
        """
        warnings: List[str] = []
        rationale: List[str] = []

        # -- Resolve model params -------------------------------------------
        params_b = self._get_model_params(model_name)
        if params_b is None:
            params_b = self._infer_params_from_name(model_name)
            if params_b is not None:
                warnings.append(
                    f"Model '{model_name}' not in known list. "
                    f"Estimated {params_b:.1f}B params from name pattern."
                )
            else:
                params_b = 7.0
                warnings.append(
                    f"Model '{model_name}' not in known list. "
                    f"Defaulting to {params_b}B params."
                )

        # -- Snapshot resources ---------------------------------------------
        gpu_info = self._get_gpu_info()
        ram_info = self._get_ram_info()
        disk_info = self._get_disk_info()

        resource_snapshot = {
            "gpu_name": gpu_info.get("name", "unknown"),
            "vram_total_gb": gpu_info.get("vram_total_gb", 0.0),
            "vram_free_gb": gpu_info.get("vram_free_gb", 0.0),
            "ram_free_gb": ram_info.get("free_gb", 0.0),
            "disk_free_gb": disk_info.get("free_gb", 0.0),
        }

        if not gpu_info.get("available", False):
            return {
                "can_run": False,
                "config": {},
                "dataset_plan": {
                    "total_rows": dataset_row_count,
                    "recommended_rows": dataset_row_count,
                    "sampling_needed": False,
                    "reason": None,
                },
                "resource_snapshot": resource_snapshot,
                "vram_estimate": {},
                "warnings": warnings + ["No CUDA GPU detected. Training requires a GPU."],
                "rationale": [],
            }

        available_vram = resource_snapshot["vram_free_gb"]

        # -- Step 1: Determine quantization ---------------------------------
        # Try quantization levels best-to-worst quality: fp16 → 8bit → 4bit
        # Pick the highest quality that fits with conservative settings.
        quantization = None
        for q_candidate in ("none", "8bit", "4bit"):
            min_est = sum(self._estimate_vram(
                params_b, q_candidate, 1, 128, True, 8, True,
            ).values())
            if min_est <= available_vram:
                quantization = q_candidate
                break

        if quantization is None:
            min_4bit = sum(self._estimate_vram(
                params_b, "4bit", 1, 128, True, 8, True,
            ).values())
            return {
                "can_run": False,
                "config": {},
                "dataset_plan": {
                    "total_rows": dataset_row_count,
                    "recommended_rows": dataset_row_count,
                    "sampling_needed": False,
                    "reason": None,
                },
                "resource_snapshot": resource_snapshot,
                "vram_estimate": {"estimated_gb": round(min_4bit, 2)},
                "warnings": warnings + [
                    f"Model requires ~{min_4bit:.1f} GB even at 4-bit with minimal settings. "
                    f"Only {available_vram:.1f} GB available."
                ],
                "rationale": ["Model too large for available VRAM."],
            }

        q_label = {"none": "fp16 (no quantization)", "8bit": "8-bit", "4bit": "4-bit"}
        rationale.append(
            f"{q_label[quantization]}: "
            f"{'best quality — VRAM is abundant' if quantization == 'none' else 'fits ' + str(params_b) + 'B model in ' + str(available_vram) + ' GB VRAM'}"
        )

        # -- Step 2: Find max sequence length (sweep) -----------------------
        target_headroom = 0.15  # reserve 15% of available VRAM
        vram_budget = available_vram * (1 - target_headroom)

        best_seq = 128
        for seq in range(128, 4097, 128):
            est = sum(self._estimate_vram(
                params_b, quantization, 1, seq, True, 8, True,
            ).values())
            if est <= vram_budget:
                best_seq = seq
            else:
                break

        # Cap by dataset text — don't waste VRAM on padding
        data_cap = max(128, int(dataset_avg_text_length * 1.5))
        # Round data_cap up to nearest 128
        data_cap = ((data_cap + 127) // 128) * 128
        max_seq_length = min(best_seq, data_cap)
        rationale.append(
            f"max_seq_length={max_seq_length}: "
            f"VRAM allows up to {best_seq}, data avg is ~{dataset_avg_text_length} chars"
        )

        # -- Step 3: Find batch size ----------------------------------------
        batch_size = 1
        for bs in [1, 2, 4, 8, 16, 32]:
            est = sum(self._estimate_vram(
                params_b, quantization, bs, max_seq_length, True, 8, True,
            ).values())
            if est <= vram_budget:
                batch_size = bs
            else:
                break

        # -- Step 4: Gradient accumulation ----------------------------------
        # Target effective batch size scales with dataset: larger data benefits
        # from larger effective batch; small data doesn't need it.
        if dataset_row_count < 100:
            target_effective = 8
        elif dataset_row_count < 1000:
            target_effective = 16
        else:
            target_effective = 32
        grad_accum = max(1, target_effective // batch_size)
        if batch_size >= target_effective:
            grad_accum = 1
            rationale.append(
                f"batch_size={batch_size}: large enough — no gradient accumulation needed"
            )

        # -- Step 5: Gradient checkpointing ---------------------------------
        est_no_gc = sum(self._estimate_vram(
            params_b, quantization, batch_size, max_seq_length, True, 8, False,
        ).values())
        headroom_pct = (available_vram - est_no_gc) / available_vram if available_vram > 0 else 0
        gradient_checkpointing = headroom_pct < 0.20
        if gradient_checkpointing:
            rationale.append("gradient_checkpointing=True: VRAM headroom < 20%")
        else:
            rationale.append(
                f"gradient_checkpointing=False: {headroom_pct:.0%} headroom — "
                "not needed, faster training"
            )

        # -- Step 6: LoRA rank + full fine-tune consideration ---------------
        final_est = sum(self._estimate_vram(
            params_b, quantization, batch_size, max_seq_length,
            True, 8, gradient_checkpointing,
        ).values())
        headroom_gb = available_vram - final_est

        # Check if full fine-tuning (no LoRA) fits — better quality for small
        # models when VRAM is abundant
        full_ft_est = sum(self._estimate_vram(
            params_b, quantization, batch_size, max_seq_length,
            False, 0, gradient_checkpointing,
        ).values())
        use_lora = True
        if full_ft_est <= vram_budget and params_b <= 3.0:
            use_lora = False
            lora_r = 0
            lora_alpha = 0
            headroom_gb = available_vram - full_ft_est
            rationale.append(
                f"Full fine-tuning (no LoRA): model is small ({params_b}B) and "
                f"VRAM is sufficient ({full_ft_est:.1f}/{available_vram:.1f} GB)"
            )
        else:
            # Scale LoRA rank with available headroom
            if headroom_gb < 1.0:
                lora_r = 8
            elif headroom_gb < 3.0:
                lora_r = 16
            elif headroom_gb < 8.0:
                lora_r = 32
            else:
                lora_r = 64
            lora_alpha = lora_r * 2
            rationale.append(f"lora_r={lora_r}: {headroom_gb:.1f} GB headroom")

        # -- Step 7: Learning rate ------------------------------------------
        base_lr = 2e-4 if use_lora else 5e-5  # full FT needs lower LR
        if dataset_row_count < 50:
            learning_rate = base_lr * 0.5
            rationale.append(
                f"learning_rate={learning_rate:.1e}: halved for small dataset "
                f"({dataset_row_count} rows)"
            )
        elif dataset_row_count > 5000:
            learning_rate = base_lr * 1.5
            rationale.append(
                f"learning_rate={learning_rate:.1e}: increased for large dataset"
            )
        else:
            learning_rate = base_lr

        # -- Step 8: Epochs -------------------------------------------------
        num_epochs = min(5, max(1, 1000 // max(dataset_row_count, 1)))
        rationale.append(
            f"{num_epochs} epoch(s): {'small' if dataset_row_count < 100 else 'standard'} "
            f"dataset ({dataset_row_count} rows)"
        )

        # -- Step 9: Dataset sampling ---------------------------------------
        free_ram_bytes = ram_info.get("free_gb", 0.0) * 1024**3
        est_dataset_bytes = dataset_row_count * dataset_avg_text_length * 4  # rough: 4 bytes/char with overhead
        sampling_needed = est_dataset_bytes > free_ram_bytes * 0.5
        if sampling_needed:
            recommended_rows = max(
                100,
                int(free_ram_bytes * 0.5 / max(dataset_avg_text_length * 4, 1)),
            )
            rationale.append(
                f"Dataset sampling: {dataset_row_count} rows → {recommended_rows} "
                f"(RAM limited to {ram_info.get('free_gb', 0):.1f} GB free)"
            )
        else:
            recommended_rows = dataset_row_count

        # -- Step 10: Warmup ratio ------------------------------------------
        warmup_ratio = 0.05 if dataset_row_count > 100 else 0.1

        # -- Step 11: Optimizer ---------------------------------------------
        optim = "paged_adamw_8bit" if headroom_gb < 1.5 else "adamw_torch"

        # -- Step 12: Disk warning ------------------------------------------
        free_disk = disk_info.get("free_gb", 0.0)
        if free_disk < 5.0:
            warnings.append(
                f"Low disk space ({free_disk:.1f} GB). "
                f"Free at least 15 GB for stable training with checkpoints."
            )

        # -- Compute final VRAM estimate ------------------------------------
        final_breakdown = self._estimate_vram(
            params_b, quantization, batch_size, max_seq_length,
            use_lora, lora_r if use_lora else 0, gradient_checkpointing,
        )
        final_total = round(sum(final_breakdown.values()), 2)

        config: Dict[str, Any] = {
            "quantization": quantization,
            "max_seq_length": max_seq_length,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "gradient_checkpointing": gradient_checkpointing,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": "cosine",
            "weight_decay": 0.01,
            "optim": optim,
        }

        return {
            "can_run": True,
            "config": config,
            "dataset_plan": {
                "total_rows": dataset_row_count,
                "recommended_rows": recommended_rows,
                "sampling_needed": sampling_needed,
                "reason": (
                    f"Dataset too large for available RAM ({ram_info.get('free_gb', 0):.1f} GB free)"
                    if sampling_needed else None
                ),
            },
            "resource_snapshot": resource_snapshot,
            "vram_estimate": {
                "estimated_gb": final_total,
                "headroom_gb": round(available_vram - final_total, 2),
                "breakdown": final_breakdown,
            },
            "warnings": warnings,
            "rationale": rationale,
        }

    def disk_preflight(
        self,
        output_dir: str = "",
        estimated_size_gb: float = 5.0,
    ) -> Dict[str, Any]:
        """Check if sufficient disk space is available for an operation.

        Args:
            output_dir: Directory where output will be written.
            estimated_size_gb: Estimated disk space needed (GB).

        Returns:
            Dict with ``can_run``, ``free_gb``, ``required_gb``, and warnings.
        """
        target = output_dir or self._output_dir
        try:
            usage = shutil.disk_usage(os.path.abspath(target))
            free_gb = usage.free / 1024 ** 3
        except OSError:
            return {
                "can_run": False,
                "error": f"Cannot access directory: {target}",
            }

        buffer = estimated_size_gb * 1.2  # 20% safety margin
        can_run = free_gb >= buffer
        warnings: List[str] = []
        if not can_run:
            warnings.append(
                f"Need ~{estimated_size_gb:.1f} GB but only {free_gb:.1f} GB free "
                f"in {target}. Free disk space before proceeding."
            )
        elif free_gb < 10:
            warnings.append(
                f"Low disk space ({free_gb:.1f} GB free). "
                "Consider freeing space to avoid interruptions."
            )

        return {
            "can_run": can_run,
            "free_gb": round(free_gb, 2),
            "required_gb": estimated_size_gb,
            "directory": target,
            "warnings": warnings,
        }

    def prescribe_pipeline(
        self,
        model_name: str,
        dataset_row_count: int,
        dataset_avg_text_length: int,
        stages: Optional[List[str]] = None,
        technique: str = "sft",
    ) -> Dict[str, Any]:
        """End-to-end resource recommendations across pipeline stages.

        Analyzes feasibility for evaluate → train → deploy stages and returns
        per-stage recommendations with a combined feasibility assessment.

        Args:
            model_name: HuggingFace model ID or local path.
            dataset_row_count: Number of rows in the dataset.
            dataset_avg_text_length: Average text length in characters.
            stages: Pipeline stages to analyze (default: all).
            technique: Training technique (sft, dpo, grpo, kto).
        """
        all_stages = ["evaluate", "train", "deploy"]
        requested = stages or all_stages
        requested = [s for s in requested if s in all_stages]

        ram_info = self._get_ram_info()
        gpu_info = self._get_gpu_info()
        disk_info = self._get_disk_info()

        ram_free_gb = ram_info.get("free_gb", 0)
        gpu_available = gpu_info.get("available", False)
        vram_free_gb = gpu_info.get("vram_free_gb", 0) if gpu_available else 0

        results: Dict[str, Any] = {}
        all_feasible = True
        all_warnings: List[str] = []

        # ---- Evaluate stage ----
        if "evaluate" in requested:
            # spaCy + BERT tokenizer ≈ 500 MB RAM
            # Dataset in memory ≈ row_count * avg_text_length * 4 bytes
            eval_model_ram_gb = 0.5
            dataset_ram_gb = (dataset_row_count * dataset_avg_text_length * 4) / (1024 ** 3)
            eval_total_ram_gb = eval_model_ram_gb + dataset_ram_gb
            eval_feasible = ram_free_gb > eval_total_ram_gb * 1.2

            eval_warnings: List[str] = []
            eval_recommendations: List[str] = []

            if not eval_feasible:
                sample_rows = int((ram_free_gb * 0.5 * (1024 ** 3)) / (dataset_avg_text_length * 4))
                eval_warnings.append(
                    f"Dataset ({dataset_row_count} rows) may exceed RAM. "
                    f"Consider sampling to ~{sample_rows} rows for evaluation."
                )
                eval_recommendations.append(f"Sample dataset to {sample_rows} rows before evaluating")
            else:
                eval_recommendations.append("Dataset fits in RAM for evaluation")

            results["evaluate"] = {
                "feasible": eval_feasible,
                "estimated_ram_gb": round(eval_total_ram_gb, 2),
                "available_ram_gb": round(ram_free_gb, 2),
                "warnings": eval_warnings,
                "recommendations": eval_recommendations,
            }
            if not eval_feasible:
                all_feasible = False
                all_warnings.extend(eval_warnings)

        # ---- Train stage ----
        if "train" in requested:
            train_result = self.prescribe(
                model_name=model_name,
                dataset_row_count=dataset_row_count,
                dataset_avg_text_length=dataset_avg_text_length,
                technique=technique,
            )
            results["train"] = {
                "feasible": train_result.get("can_run", False),
                "config": train_result.get("config"),
                "vram_estimate": train_result.get("vram_estimate"),
                "dataset_plan": train_result.get("dataset_plan"),
                "warnings": train_result.get("warnings", []),
                "rationale": train_result.get("rationale", []),
            }
            if not train_result.get("can_run", False):
                all_feasible = False
                all_warnings.extend(train_result.get("warnings", []))

            # Disk check for training checkpoints
            disk_check = self.disk_preflight(estimated_size_gb=5.0)
            if not disk_check.get("can_run", True):
                all_feasible = False
                all_warnings.extend(disk_check.get("warnings", []))

        # ---- Deploy stage ----
        if "deploy" in requested:
            params_b = self._get_model_params(model_name)
            # 4-bit deployment VRAM = params * 0.55 bytes
            deploy_vram_gb = (params_b * 1e9 * 0.55) / (1024 ** 3)
            deploy_feasible = gpu_available and vram_free_gb > deploy_vram_gb * 1.2

            deploy_warnings: List[str] = []
            if not gpu_available:
                deploy_warnings.append("No GPU available for model deployment")
            elif not deploy_feasible:
                deploy_warnings.append(
                    f"Model needs ~{deploy_vram_gb:.1f} GB VRAM for 4-bit deployment, "
                    f"but only {vram_free_gb:.1f} GB free"
                )

            results["deploy"] = {
                "feasible": deploy_feasible,
                "estimated_vram_gb": round(deploy_vram_gb, 2),
                "available_vram_gb": round(vram_free_gb, 2),
                "quantization": "4bit",
                "warnings": deploy_warnings,
            }
            if not deploy_feasible:
                all_feasible = False
                all_warnings.extend(deploy_warnings)

        return {
            "all_feasible": all_feasible,
            "stages": results,
            "resource_snapshot": {
                "gpu": gpu_info,
                "ram": ram_info,
                "disk": disk_info,
            },
            "warnings": all_warnings,
        }

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        if torch is None or not torch.cuda.is_available():
            return {"available": False}

        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3

        return {
            "available": True,
            "name": torch.cuda.get_device_name(0),
            "vram_total_gb": round(total, 2),
            "vram_free_gb": round(total - allocated, 2),
            "vram_used_gb": round(allocated, 2),
            "vram_reserved_gb": round(reserved, 2),
            "compute_capability": f"{props.major}.{props.minor}",
            "cuda_version": getattr(torch.version, "cuda", "unknown"),
        }

    @staticmethod
    def _get_ram_info() -> Dict[str, Any]:
        """Get system RAM information."""
        if psutil is None:
            return {"available": False, "error": "psutil not installed"}

        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / 1024**3, 2),
            "free_gb": round(mem.available / 1024**3, 2),
            "used_gb": round(mem.used / 1024**3, 2),
            "percent_used": round(mem.percent, 1),
        }

    def _get_disk_info(self) -> Dict[str, Any]:
        """Get disk space for the output directory."""
        try:
            usage = shutil.disk_usage(os.path.abspath(self._output_dir))
            return {
                "output_dir": self._output_dir,
                "total_gb": round(usage.total / 1024**3, 2),
                "free_gb": round(usage.free / 1024**3, 2),
                "used_gb": round(usage.used / 1024**3, 2),
            }
        except OSError:
            # Fallback to current directory
            usage = shutil.disk_usage(".")
            return {
                "output_dir": ".",
                "total_gb": round(usage.total / 1024**3, 2),
                "free_gb": round(usage.free / 1024**3, 2),
                "used_gb": round(usage.used / 1024**3, 2),
            }

    def _get_model_params(self, model_name: str) -> Optional[float]:
        """Look up parameter count for a known model."""
        return self.MODEL_PARAMS.get(model_name)

    @staticmethod
    def _infer_params_from_name(model_name: str) -> Optional[float]:
        """Try to extract parameter count from the model name string.

        Matches patterns like "7B", "70B", "1.5B", "0.5B" in the model name.
        """
        match = re.search(r"(\d+\.?\d*)\s*[Bb]", model_name)
        if match:
            return float(match.group(1))
        return None

    def _estimate_vram(
        self,
        params_b: float,
        quantization: str = "4bit",
        batch_size: int = 1,
        max_seq_length: int = 512,
        use_lora: bool = True,
        lora_r: int = 8,
        gradient_checkpointing: bool = False,
    ) -> Dict[str, float]:
        """Break down VRAM into model + lora + optimizer + activations + overhead."""
        bytes_per_param = self._BYTES_PER_PARAM.get(quantization, 2.1)

        # Model weights
        model_gb = round(params_b * bytes_per_param, 2)

        # LoRA adapter (~0.5-2% of model params depending on r)
        if use_lora:
            # LoRA adds 2 * r * hidden_dim * num_layers parameters (very rough)
            # Typically ~0.5-2% of base model params
            lora_fraction = (lora_r / 64) * 0.02  # scale with rank
            lora_gb = round(params_b * lora_fraction * 2.0, 3)  # FP16 for LoRA weights
        else:
            lora_gb = 0.0

        # Optimizer states
        if use_lora:
            # Only LoRA params are optimized → small optimizer footprint
            # 8-bit Adam: ~2 bytes per param (momentum + variance, 8-bit each)
            trainable_params_b = params_b * (lora_r / 64) * 0.02
            optimizer_gb = round(trainable_params_b * 2.0, 3)
        else:
            # Full fine-tuning: all params optimized
            # AdamW FP32: 8 bytes per param (2 FP32 states)
            optimizer_gb = round(params_b * 8.0 / 1e0, 2)
            # But typically use 8-bit optimizer
            optimizer_gb = round(params_b * 2.0, 2)

        # Activation memory (rough estimate)
        # Depends on batch_size, seq_length, hidden_dim, num_layers
        # Rough formula: batch * seq * hidden * layers * bytes / 1e9
        # Simplified: proportional to params, batch, seq
        base_activation = params_b * 0.15 * (max_seq_length / 512) * batch_size
        if gradient_checkpointing:
            # GC saves ~60-70% of activation memory at cost of ~30% slower
            base_activation *= 0.35
        activations_gb = round(base_activation, 3)

        # CUDA overhead (context, kernels, fragmentation)
        overhead_gb = round(0.2 + params_b * 0.02, 2)

        return {
            "model_gb": model_gb,
            "lora_gb": lora_gb,
            "optimizer_gb": optimizer_gb,
            "activations_gb": activations_gb,
            "overhead_gb": overhead_gb,
        }
