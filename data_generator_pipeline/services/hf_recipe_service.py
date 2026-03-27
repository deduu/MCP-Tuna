"""Compose Hugging Face datasets into training-ready MCP Tuna rows."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from shared.async_utils import run_sync
from shared.recipe_registry import get_hf_dataset_recipe, list_hf_dataset_recipes


class HFDatasetRecipeService:
    """Build small training datasets from one or more HF dataset sources."""

    async def list_recipes(self) -> Dict[str, Any]:
        return {
            "success": True,
            "recipes": list_hf_dataset_recipes(),
        }

    async def get_recipe(self, recipe_name: str) -> Dict[str, Any]:
        recipe = get_hf_dataset_recipe(recipe_name)
        if recipe is None:
            return {
                "success": False,
                "error": f"Unknown recipe: {recipe_name}",
            }
        return {
            "success": True,
            "recipe_name": recipe_name,
            "recipe": recipe,
        }

    async def compose_dataset(
        self,
        *,
        recipe_name: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        shuffle: bool = True,
        seed: int = 42,
        max_rows_per_source: Optional[int] = None,
        target_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await run_sync(
            self._compose_dataset_sync,
            recipe_name=recipe_name,
            sources=sources,
            shuffle=shuffle,
            seed=seed,
            max_rows_per_source=max_rows_per_source,
            target_format=target_format,
        )

    def _compose_dataset_sync(
        self,
        *,
        recipe_name: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        shuffle: bool = True,
        seed: int = 42,
        max_rows_per_source: Optional[int] = None,
        target_format: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            from datasets import load_dataset as hf_load_dataset
        except Exception as exc:
            return {
                "success": False,
                "error": (
                    "datasets not installed. Install training/data dependencies before "
                    f"using HF composition tools. ({exc})"
                ),
            }

        self._ensure_writable_hf_cache()

        recipe = None
        if recipe_name:
            recipe = get_hf_dataset_recipe(recipe_name)
            if recipe is None:
                return {"success": False, "error": f"Unknown recipe: {recipe_name}"}

        resolved_sources = list(sources or (recipe or {}).get("sources", []))
        if not resolved_sources:
            return {
                "success": False,
                "error": "Provide either recipe_name or a non-empty sources list.",
            }

        resolved_target = target_format or (recipe or {}).get("target_format")
        all_rows: List[Dict[str, Any]] = []
        per_source: List[Dict[str, Any]] = []

        for source in resolved_sources:
            source_name = str(source.get("dataset_name") or "").strip()
            subset = source.get("subset")
            split = str(source.get("split") or "train")
            rename_columns = dict(source.get("rename_columns") or {})
            drop_columns = set(source.get("drop_columns") or [])
            source_limit = source.get("max_rows")

            if max_rows_per_source is not None:
                if source_limit is None:
                    source_limit = max_rows_per_source
                else:
                    source_limit = min(int(source_limit), int(max_rows_per_source))

            if not source_name:
                return {"success": False, "error": "Each source needs dataset_name."}

            try:
                ds = hf_load_dataset(source_name, subset, split=split)
            except Exception as exc:
                return {
                    "success": False,
                    "error": self._format_load_error(
                        source_name=source_name,
                        subset=subset,
                        split=split,
                        error=exc,
                    ),
                }
            total_requested = int(source_limit) if source_limit is not None else len(ds)
            if source_limit is not None:
                ds = ds.select(range(min(int(source_limit), len(ds))))

            kept = 0
            skipped = 0
            original_columns = list(ds.column_names)
            for row in ds:
                prepared = self._prepare_row(
                    row=row,
                    rename_columns=rename_columns,
                    drop_columns=drop_columns,
                    source_name=source_name,
                    subset=subset,
                    split=split,
                )
                normalized = self._normalize_row(prepared, resolved_target)
                if normalized is None:
                    skipped += 1
                    continue
                all_rows.append(normalized)
                kept += 1

            per_source.append(
                {
                    "dataset_name": source_name,
                    "subset": subset,
                    "split": split,
                    "requested_rows": total_requested,
                    "kept_rows": kept,
                    "skipped_rows": skipped,
                    "original_columns": original_columns,
                    "target_format": resolved_target,
                }
            )

        if shuffle:
            random.Random(seed).shuffle(all_rows)

        if not all_rows:
            return {
                "success": False,
                "error": "No rows were produced. Check the selected sources and target_format.",
                "per_source_counts": per_source,
            }

        return {
            "success": True,
            "recipe_name": recipe_name,
            "target_format": resolved_target,
            "shuffle": shuffle,
            "seed": seed,
            "count": len(all_rows),
            "per_source_counts": per_source,
            "data_points": all_rows,
        }

    @staticmethod
    def _ensure_writable_hf_cache() -> None:
        workspace_hf_home = (Path.cwd() / ".cache" / "huggingface").resolve()
        workspace_hf_home.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(workspace_hf_home))
        os.environ.setdefault("HF_DATASETS_CACHE", str(workspace_hf_home / "datasets"))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(workspace_hf_home / "hub"))

    @staticmethod
    def _format_load_error(
        *,
        source_name: str,
        subset: Any,
        split: str,
        error: Exception,
    ) -> str:
        details = (
            f"Failed to load Hugging Face dataset '{source_name}'"
            f" (subset={subset!r}, split={split!r}). {error}"
        )
        if "BuilderConfig" in str(error) and "Available:" in str(error):
            return (
                f"{details} This usually means the config/subset is wrong. "
                "For example, HuggingFaceTB/smoltalk2 currently uses config 'SFT' and "
                "the dataset names like 'OpenThoughts3_1.2M' or "
                "'Llama_Nemotron_Post_Training_Dataset_reasoning_r1' belong in the split field."
            )
        return details

    @staticmethod
    def _prepare_row(
        *,
        row: Dict[str, Any],
        rename_columns: Dict[str, str],
        drop_columns: set[str],
        source_name: str,
        subset: Any,
        split: str,
    ) -> Dict[str, Any]:
        prepared: Dict[str, Any] = {}
        for key, value in row.items():
            if key in drop_columns:
                continue
            target_key = rename_columns.get(key, key)
            prepared[target_key] = value

        prepared["_source_dataset"] = source_name
        prepared["_source_subset"] = subset
        prepared["_source_split"] = split
        return prepared

    def _normalize_row(
        self,
        row: Dict[str, Any],
        target_format: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not target_format or target_format == "raw":
            return row
        if target_format == "sft":
            return self._to_sft_row(row)
        if target_format == "dpo":
            return self._to_dpo_row(row)
        raise ValueError(f"Unsupported target_format: {target_format}")

    def _to_sft_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = row.get("prompt")
        response = row.get("response")

        if prompt is None and "instruction" in row:
            prompt = f"{row.get('instruction', '')} {row.get('input', '')}".strip()
        if response is None:
            for key in ("completion", "output", "answer"):
                if row.get(key) is not None:
                    response = row.get(key)
                    break

        if (prompt is None or response is None) and isinstance(row.get("messages"), list):
            prompt, response = self._messages_to_prompt_response(row["messages"])

        if prompt is None or response is None:
            return None

        return {
            "prompt": self._stringify_value(prompt),
            "response": self._stringify_value(response),
            "_source_dataset": row.get("_source_dataset"),
            "_source_subset": row.get("_source_subset"),
            "_source_split": row.get("_source_split"),
        }

    def _to_dpo_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = row.get("prompt")
        chosen = row.get("chosen")
        rejected = row.get("rejected")

        if prompt is None and "instruction" in row:
            prompt = f"{row.get('instruction', '')} {row.get('input', '')}".strip()

        if prompt is None or chosen is None or rejected is None:
            return None

        return {
            "prompt": self._stringify_value(prompt),
            "chosen": self._stringify_value(chosen),
            "rejected": self._stringify_value(rejected),
            "_source_dataset": row.get("_source_dataset"),
            "_source_subset": row.get("_source_subset"),
            "_source_split": row.get("_source_split"),
        }

    def _messages_to_prompt_response(
        self,
        messages: Iterable[Dict[str, Any]],
    ) -> tuple[Optional[str], Optional[str]]:
        normalized_messages = [
            {
                "role": str(message.get("role", "")).strip().lower(),
                "content": self._content_to_text(message.get("content")),
            }
            for message in messages
            if isinstance(message, dict)
        ]
        if not normalized_messages:
            return None, None

        last_assistant_index = -1
        for idx in range(len(normalized_messages) - 1, -1, -1):
            if normalized_messages[idx]["role"] == "assistant":
                last_assistant_index = idx
                break

        if last_assistant_index < 0:
            return None, None

        response = normalized_messages[last_assistant_index]["content"]
        prior = normalized_messages[:last_assistant_index]
        if not prior:
            return None, response

        user_messages = [msg for msg in prior if msg["role"] == "user" and msg["content"]]
        system_messages = [msg for msg in prior if msg["role"] == "system" and msg["content"]]
        assistant_messages = [
            msg for msg in prior if msg["role"] == "assistant" and msg["content"]
        ]

        if len(user_messages) == 1 and not system_messages and not assistant_messages:
            return user_messages[0]["content"], response

        transcript = []
        for message in prior:
            content = message["content"]
            if not content:
                continue
            role = message["role"] or "unknown"
            transcript.append(f"{role.title()}: {content}")
        return "\n".join(transcript), response

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"].strip())
                    elif item_type == "input_text" and isinstance(item.get("text"), str):
                        parts.append(item["text"].strip())
                    elif isinstance(item.get("content"), str):
                        parts.append(item["content"].strip())
            return "\n".join(part for part in parts if part)
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"].strip()
            if isinstance(content.get("content"), str):
                return content["content"].strip()
        return str(content).strip()

    @staticmethod
    def _stringify_value(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value)
        return json.dumps(value, ensure_ascii=False)
