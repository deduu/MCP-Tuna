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
            max_memory = {}
            for device_index in range(int(torch.cuda.device_count())):
                total = torch.cuda.get_device_properties(device_index).total_memory
                usable_gib = (total * 0.85) / (1024 ** 3)
                max_memory[device_index] = f"{usable_gib:.1f}GiB"
            max_memory["cpu"] = "30GiB"
            return max_memory
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
