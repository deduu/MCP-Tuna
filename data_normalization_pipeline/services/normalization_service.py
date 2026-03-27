"""Data normalization service for key renaming, text cleanup, and schema remapping."""

import re
import unicodedata
from typing import Any, Dict, List, Optional

from shared.async_utils import run_sync
from shared.config import NormalizationConfig


class DataNormalizationService:
    """Normalizes datasets to a consistent format for downstream pipelines."""

    _REMAPPABLE_PRESETS: Dict[str, Dict[str, Any]] = {
        "chat_triplet_to_sft": {
            "target_format": "sft",
            "templates": {
                "instruction": "System: {{system}}\n\nUser: {{user}}",
                "input": "",
                "output": "{{assistant}}",
            },
        },
        "prompt_response_to_sft": {
            "target_format": "sft",
            "templates": {
                "instruction": "{{prompt}}",
                "input": "",
                "output": "{{response}}",
            },
        },
        "qa_to_sft": {
            "target_format": "sft",
            "templates": {
                "instruction": "{{question}}",
                "input": "",
                "output": "{{answer}}",
            },
        },
    }
    _TEMPLATE_PATTERN = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")

    @staticmethod
    def _count_changed_rows(
        before: List[Dict[str, Any]],
        after: List[Dict[str, Any]],
    ) -> int:
        return sum(1 for prev, curr in zip(before, after) if prev != curr)

    @staticmethod
    def _clean_text_value(value: str) -> str:
        return unicodedata.normalize("NFC", value.strip())

    def _render_template(
        self,
        template: str,
        data_point: Dict[str, Any],
        strip_whitespace: bool,
    ) -> str:
        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            value = data_point.get(key, "")
            if value is None:
                return ""
            text = value if isinstance(value, str) else str(value)
            return self._clean_text_value(text) if strip_whitespace else text

        rendered = self._TEMPLATE_PATTERN.sub(replace, template)
        return self._clean_text_value(rendered) if strip_whitespace else rendered

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
            dp = dict(dp)
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

    async def remap_fields(
        self,
        data_points: List[Dict[str, Any]],
        preset: str = "chat_triplet_to_sft",
        keep_unmapped_fields: bool = False,
        strip_whitespace: bool = True,
    ) -> Dict[str, Any]:
        """Reshape rows into a target schema using a named preset."""
        return await run_sync(
            self._remap_fields_sync,
            data_points,
            preset,
            keep_unmapped_fields,
            strip_whitespace,
        )

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
                    text = self._clean_text_value(original)
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

    def _remap_fields_sync(
        self,
        data_points: List[Dict[str, Any]],
        preset: str,
        keep_unmapped_fields: bool,
        strip_whitespace: bool,
    ) -> Dict[str, Any]:
        preset_config = self._REMAPPABLE_PRESETS.get(preset)
        if preset_config is None:
            return {
                "success": False,
                "error": f"Unknown remap preset: {preset}",
                "available_presets": sorted(self._REMAPPABLE_PRESETS),
            }

        templates = preset_config["templates"]
        target_format = preset_config["target_format"]
        remapped: List[Dict[str, Any]] = []
        changed_rows = 0
        dropped_fields: set[str] = set()
        created_fields = list(templates.keys())

        for data_point in data_points:
            new_point: Dict[str, Any] = {}
            for field_name, template in templates.items():
                new_point[field_name] = self._render_template(
                    template,
                    data_point,
                    strip_whitespace,
                )

            if keep_unmapped_fields:
                for key, value in data_point.items():
                    if key not in new_point:
                        new_point[key] = value
            else:
                dropped_fields.update(
                    key for key in data_point.keys() if key not in new_point
                )

            if new_point != data_point:
                changed_rows += 1
            remapped.append(new_point)

        return {
            "success": True,
            "data_points": remapped,
            "count": len(remapped),
            "changed_rows": changed_rows,
            "preset": preset,
            "target_format": target_format,
            "created_fields": created_fields,
            "dropped_fields": sorted(dropped_fields),
            "unchanged": changed_rows == 0,
        }
