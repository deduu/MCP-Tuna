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
