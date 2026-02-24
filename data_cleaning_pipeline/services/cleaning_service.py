"""Data cleaning service — deduplication, schema validation, short-entry filtering."""

from typing import Any, Dict, List, Optional
from AgentY.shared.config import CleaningConfig


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

        if config.remove_empty_fields:
            result = await self.remove_empty_fields(data_points)
            data_points = result["data_points"]

        if config.remove_duplicates:
            result = await self.deduplicate(data_points)
            data_points = result["data_points"]

        result = await self.remove_short_entries(
            data_points,
            min_instruction=config.min_instruction_length,
            min_output=config.min_output_length,
        )
        data_points = result["data_points"]

        return {
            "success": True,
            "data_points": data_points,
            "original_count": original_count,
            "cleaned_count": len(data_points),
            "removed": original_count - len(data_points),
        }

    async def deduplicate(
        self,
        data_points: List[Dict[str, Any]],
        key: str = "instruction",
    ) -> Dict[str, Any]:
        """Remove exact duplicates based on the given key."""
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
        }

    async def validate_schema(
        self,
        data_points: List[Dict[str, Any]],
        technique: str = "sft",
    ) -> Dict[str, Any]:
        """Validate each entry has the required fields for the technique."""
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
        }

    async def remove_empty_fields(
        self,
        data_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Remove entries with empty instruction or output."""
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
