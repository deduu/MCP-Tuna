"""Data cleaning service — deduplication, schema validation, short-entry filtering."""

from typing import Any, Dict, List, Optional

from shared.async_utils import run_sync
from shared.config import CleaningConfig


class DataCleaningService:
    """Cleans datasets by removing duplicates, empty fields, and malformed entries."""

    async def clean_dataset(
        self,
        data_points: List[Dict[str, Any]],
        config: Optional[CleaningConfig] = None,
    ) -> Dict[str, Any]:
        """Run all cleaning steps with the given config."""
        config = config or CleaningConfig()
        original_count = len(data_points)
        steps: Dict[str, Dict[str, Any]] = {}

        if config.remove_empty_fields:
            result = await self.remove_empty_fields(data_points)
            steps["remove_empty_fields"] = {
                "enabled": True,
                "before": len(data_points),
                "after": len(result["data_points"]),
                "removed": result.get("removed", 0),
            }
            data_points = result["data_points"]
        else:
            steps["remove_empty_fields"] = {"enabled": False}

        if config.remove_duplicates:
            result = await self.deduplicate(data_points)
            steps["deduplicate"] = {
                "enabled": True,
                "before": len(data_points),
                "after": len(result["data_points"]),
                "removed": result.get("duplicates_removed", 0),
                "key": "instruction",
            }
            data_points = result["data_points"]
        else:
            steps["deduplicate"] = {"enabled": False}

        result = await self.remove_short_entries(
            data_points,
            min_instruction=config.min_instruction_length,
            min_output=config.min_output_length,
        )
        steps["remove_short_entries"] = {
            "enabled": True,
            "before": len(data_points),
            "after": len(result["data_points"]),
            "removed": result.get("removed", 0),
            "min_instruction": config.min_instruction_length,
            "min_output": config.min_output_length,
        }
        data_points = result["data_points"]

        return {
            "success": True,
            "data_points": data_points,
            "original_count": original_count,
            "cleaned_count": len(data_points),
            "removed": original_count - len(data_points),
            "unchanged": original_count == len(data_points),
            "steps": steps,
        }

    async def deduplicate(
        self,
        data_points: List[Dict[str, Any]],
        key: str = "instruction",
    ) -> Dict[str, Any]:
        """Remove exact duplicates based on the given key."""
        return await run_sync(self._deduplicate_sync, data_points, key)

    def _deduplicate_sync(
        self,
        data_points: List[Dict[str, Any]],
        key: str = "instruction",
    ) -> Dict[str, Any]:
        seen = set()
        unique = []
        for dp in data_points:
            val = dp.get(key, "")
            if val not in seen:
                seen.add(val)
                unique.append(dp)

        return {
            "success": True,
            "data_points": unique,
            "original_count": len(data_points),
            "deduplicated_count": len(unique),
            "duplicates_removed": len(data_points) - len(unique),
            "key": key,
        }

    async def validate_schema(
        self,
        data_points: List[Dict[str, Any]],
        technique: str = "sft",
    ) -> Dict[str, Any]:
        """Validate each entry has the required fields for the technique."""
        return await run_sync(self._validate_schema_sync, data_points, technique)

    def _validate_schema_sync(
        self,
        data_points: List[Dict[str, Any]],
        technique: str = "sft",
    ) -> Dict[str, Any]:
        required_fields = {
            "sft": ["instruction", "output"],
            "dpo": ["prompt", "chosen", "rejected"],
            "grpo": ["prompt", "responses"],
        }
        fields = required_fields.get(technique, ["instruction", "output"])

        valid = []
        invalid = []
        for dp in data_points:
            if all(dp.get(f) for f in fields):
                valid.append(dp)
            else:
                invalid.append(dp)

        return {
            "success": True,
            "data_points": valid,
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "required_fields": fields,
        }

    async def remove_short_entries(
        self,
        data_points: List[Dict[str, Any]],
        min_instruction: int = 10,
        min_output: int = 20,
    ) -> Dict[str, Any]:
        """Filter entries below minimum length thresholds."""
        return await run_sync(
            self._remove_short_entries_sync, data_points, min_instruction, min_output,
        )

    def _remove_short_entries_sync(
        self,
        data_points: List[Dict[str, Any]],
        min_instruction: int = 10,
        min_output: int = 20,
    ) -> Dict[str, Any]:
        filtered = [
            dp for dp in data_points
            if len(dp.get("instruction", "")) >= min_instruction
            and len(dp.get("output", "")) >= min_output
        ]

        return {
            "success": True,
            "data_points": filtered,
            "original_count": len(data_points),
            "filtered_count": len(filtered),
            "removed": len(data_points) - len(filtered),
            "min_instruction": min_instruction,
            "min_output": min_output,
        }

    async def remove_empty_fields(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Remove entries with empty instruction or output."""
        return await run_sync(self._remove_empty_fields_sync, data_points)

    def _remove_empty_fields_sync(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        filtered = [
            dp for dp in data_points
            if dp.get("instruction", "").strip() and dp.get("output", "").strip()
        ]

        return {
            "success": True,
            "data_points": filtered,
            "original_count": len(data_points),
            "filtered_count": len(filtered),
            "removed": len(data_points) - len(filtered),
        }
