"""Data normalization service — format standardization, key renaming, text cleanup."""

import unicodedata
from typing import Any, Dict, List, Optional

from shared.async_utils import run_sync
from shared.config import NormalizationConfig


class DataNormalizationService:
    """Normalizes datasets to a consistent format for downstream pipelines."""

    @staticmethod
    def _count_changed_rows(
        before: List[Dict[str, Any]],
        after: List[Dict[str, Any]],
    ) -> int:
        return sum(1 for prev, curr in zip(before, after) if prev != curr)

    async def normalize_dataset(
        self,
        data_points: List[Dict[str, Any]],
        config: Optional[NormalizationConfig] = None,
    ) -> Dict[str, Any]:
        """Apply all normalization steps in sequence."""
        config = config or NormalizationConfig()
        original_data = [dict(dp) for dp in data_points]
        steps: Dict[str, Dict[str, Any]] = {}

        if config.strip_whitespace:
            result = await self.strip_and_clean_text(data_points)
            steps["strip_text"] = {
                "enabled": True,
                "before": len(data_points),
                "after": len(result["data_points"]),
                "changed_rows": result.get("changed_rows", 0),
                "changed_fields": result.get("changed_fields", 0),
            }
            data_points = result["data_points"]
        else:
            steps["strip_text"] = {"enabled": False}

        if config.merge_instruction_input:
            result = await self.merge_instruction_input(data_points)
            steps["merge_fields"] = {
                "enabled": True,
                "before": len(data_points),
                "after": len(result["data_points"]),
                "changed_rows": result.get("changed_rows", 0),
                "merged_rows": result.get("merged_rows", 0),
            }
            data_points = result["data_points"]
        else:
            steps["merge_fields"] = {"enabled": False}

        result = await self.standardize_keys(data_points, config.target_format)
        steps["standardize_keys"] = {
            "enabled": True,
            "before": len(data_points),
            "after": len(result["data_points"]),
            "changed_rows": result.get("changed_rows", 0),
            "renamed_fields": result.get("renamed_fields", 0),
            "target_format": config.target_format,
        }
        data_points = result["data_points"]
        changed_rows = self._count_changed_rows(original_data, data_points)

        return {
            "success": True,
            "data_points": data_points,
            "count": len(data_points),
            "target_format": config.target_format,
            "changed_rows": changed_rows,
            "unchanged": changed_rows == 0,
            "steps": steps,
        }

    async def merge_instruction_input(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Combine instruction + input into a single prompt field."""
        return await run_sync(self._merge_instruction_input_sync, data_points)

    def _merge_instruction_input_sync(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = []
        merged_rows = 0
        for dp in data_points:
            dp = dict(dp)  # shallow copy
            instruction = dp.get("instruction", "")
            inp = dp.get("input", "")
            if inp:
                dp["instruction"] = f"{instruction} {inp}".strip()
                dp["input"] = ""
                merged_rows += 1
            merged.append(dp)

        return {
            "success": True,
            "data_points": merged,
            "count": len(merged),
            "changed_rows": merged_rows,
            "merged_rows": merged_rows,
        }

    async def standardize_keys(
        self,
        data_points: List[Dict[str, Any]],
        target_format: str = "sft",
    ) -> Dict[str, Any]:
        """Rename keys to match target format (sft/dpo/grpo)."""
        return await run_sync(self._standardize_keys_sync, data_points, target_format)

    def _standardize_keys_sync(
        self,
        data_points: List[Dict[str, Any]],
        target_format: str = "sft",
    ) -> Dict[str, Any]:
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
        changed_rows = 0
        renamed_fields = 0
        for dp in data_points:
            new_dp = {}
            row_changed = False
            for k, v in dp.items():
                new_key = mapping.get(k, k)
                if new_key != k:
                    row_changed = True
                    renamed_fields += 1
                new_dp[new_key] = v
            if row_changed:
                changed_rows += 1
            standardized.append(new_dp)

        return {
            "success": True,
            "data_points": standardized,
            "count": len(standardized),
            "target_format": target_format,
            "changed_rows": changed_rows,
            "renamed_fields": renamed_fields,
        }

    async def strip_and_clean_text(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Strip whitespace, normalize unicode, fix encoding issues."""
        return await run_sync(self._strip_and_clean_text_sync, data_points)

    def _strip_and_clean_text_sync(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        cleaned = []
        changed_rows = 0
        changed_fields = 0
        for dp in data_points:
            dp = dict(dp)
            row_changed = False
            for key in ("instruction", "input", "output", "prompt", "chosen", "rejected"):
                if key in dp and isinstance(dp[key], str):
                    original = dp[key]
                    text = original.strip()
                    text = unicodedata.normalize("NFC", text)
                    dp[key] = text
                    if text != original:
                        row_changed = True
                        changed_fields += 1
            if row_changed:
                changed_rows += 1
            cleaned.append(dp)

        return {
            "success": True,
            "data_points": cleaned,
            "count": len(cleaned),
            "changed_rows": changed_rows,
            "changed_fields": changed_fields,
        }
