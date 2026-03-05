"""GPU memory management for fine-tuning operations."""
from __future__ import annotations

import gc
import torch
from typing import Any, Dict
from transformers import BitsAndBytesConfig


class GPUService:
    """Manages GPU resources: quantization config, memory allocation, cleanup."""

    def __init__(self):
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.max_memory = self._detect_max_memory()

    @staticmethod
    def _detect_max_memory() -> Dict:
        """Set GPU memory limit to 85% of total VRAM (15% headroom)."""
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            usable_gib = (total * 0.85) / (1024 ** 3)
            return {0: f"{usable_gib:.1f}GiB", "cpu": "30GiB"}
        return {"cpu": "30GiB"}

    def clear_gpu_memory(self) -> Dict[str, Any]:
        """Clear GPU memory cache and return current stats."""
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        stats = {}
        if torch.cuda.is_available():
            stats = {
                "allocated_gb": torch.cuda.memory_allocated(0) / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved(0) / 1024**3,
            }

        return {"success": True, "memory_stats": stats}
