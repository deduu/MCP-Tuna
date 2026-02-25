"""Data normalization service — format standardization, key renaming, text cleanup."""

import unicodedata
from typing import Any, Dict, List, Optional
from shared.config import NormalizationConfig


class DataNormalizationService:
    """Normalizes datasets to a consistent format for downstream pipelines."""

    async def normalize_dataset(
        self,
        data_points: List[Dict[str, Any]],
        config: Optional[NormalizationConfig] = None,
    ) -> Dict[str, Any]:
        """Apply all normalization steps in sequence."""
        config = config or NormalizationConfig()
        original_count = len(data_points)

        if config.strip_whitespace:
            result = await self.strip_and_clean_text(data_points)
            data_points = result["data_points"]

        if config.merge_instruction_input:
            result = await self.merge_instruction_input(data_points)
            data_points = result["data_points"]

        result = await self.standardize_keys(data_points, config.target_format)
        data_points = result["data_points"]

        return {
            "success": True,
            "data_points": data_points,
            "count": len(data_points),
            "target_format": config.target_format,
        }

    async def merge_instruction_input(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Combine instruction + input into a single prompt field."""
        merged = []
        for dp in data_points:
            dp = dict(dp)  # shallow copy
            instruction = dp.get("instruction", "")
            inp = dp.get("input", "")
            if inp:
                dp["instruction"] = f"{instruction} {inp}".strip()
                dp["input"] = ""
            merged.append(dp)

        return {
            "success": True,
            "data_points": merged,
            "count": len(merged),
        }

    async def standardize_keys(
        self,
        data_points: List[Dict[str, Any]],
        target_format: str = "sft",
    ) -> Dict[str, Any]:
        """Rename keys to match target format (sft/dpo/grpo)."""
        key_maps = {
            "sft": {
                "prompt": "instruction",
                "question": "instruction",
                "answer": "output",
                "response": "output",
                "completion": "output",
            },
            "dpo": {
                "instruction": "prompt",
                "question": "prompt",
                "positive": "chosen",
                "negative": "rejected",
            },
            "grpo": {
                "instruction": "prompt",
                "question": "prompt",
            },
        }
        mapping = key_maps.get(target_format, {})

        standardized = []
        for dp in data_points:
            new_dp = {}
            for k, v in dp.items():
                new_key = mapping.get(k, k)
                new_dp[new_key] = v
            standardized.append(new_dp)

        return {
            "success": True,
            "data_points": standardized,
            "count": len(standardized),
            "target_format": target_format,
        }

    async def strip_and_clean_text(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Strip whitespace, normalize unicode, fix encoding issues."""
        cleaned = []
        for dp in data_points:
            dp = dict(dp)
            for key in ("instruction", "input", "output", "prompt", "chosen", "rejected"):
                if key in dp and isinstance(dp[key], str):
                    text = dp[key].strip()
                    text = unicodedata.normalize("NFC", text)
                    dp[key] = text
            cleaned.append(dp)

        return {
            "success": True,
            "data_points": cleaned,
            "count": len(cleaned),
        }
