from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass
class PreferenceDatasetNormalizationResult:
    dataset: Any
    summary: dict[str, Any]


_FIELD_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "dpo": {
        "text_fields": ("prompt", "chosen", "rejected"),
        "list_text_fields": (),
    },
    "grpo": {
        "text_fields": ("prompt",),
        "list_text_fields": ("responses",),
    },
    "kto": {
        "text_fields": ("prompt", "completion"),
        "list_text_fields": (),
    },
}


def _normalize_text(value: str) -> str:
    return value.strip()


def normalize_preference_dataset(
    dataset: Any,
    trainer: str,
) -> PreferenceDatasetNormalizationResult:
    spec = _FIELD_SPECS.get(str(trainer or "").strip().lower())
    if spec is None:
        raise ValueError(f"Unsupported preference trainer: {trainer}")

    text_fields = spec["text_fields"]
    list_text_fields = spec["list_text_fields"]
    invalid_rows: list[dict[str, Any]] = []
    if isinstance(dataset, list):
        column_names: list[str] = []
        seen: set[str] = set()
        for row in dataset[:32]:
            if not isinstance(row, Mapping):
                continue
            for key in row:
                key_text = str(key)
                if key_text not in seen:
                    seen.add(key_text)
                    column_names.append(key_text)
    else:
        column_names = [str(column) for column in getattr(dataset, "column_names", [])]

    summary = {
        "schema_version": 1,
        "trainer": trainer,
        "normalized_text_fields": list(text_fields),
        "normalized_list_text_fields": list(list_text_fields),
        "num_rows": int(len(dataset)) if hasattr(dataset, "__len__") else None,
        "trimmed_scalar_value_count": 0,
        "trimmed_list_value_count": 0,
        "trimmed_row_count": 0,
        "invalid_row_count": 0,
        "invalid_rows": invalid_rows,
        "preserved_extra_columns": [
            column
            for column in column_names
            if column not in set(text_fields) | set(list_text_fields)
        ],
    }

    def normalize_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        row_trimmed = False
        issues: list[str] = []

        for field in text_fields:
            value = row.get(field)
            if not isinstance(value, str):
                issues.append(f"{field}: expected non-empty string")
                continue
            stripped = _normalize_text(value)
            if stripped != value:
                summary["trimmed_scalar_value_count"] += 1
                row_trimmed = True
            if not stripped:
                issues.append(f"{field}: empty after strip")
            updates[field] = stripped

        for field in list_text_fields:
            value = row.get(field)
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                issues.append(f"{field}: expected list of text values")
                continue
            normalized_items: list[str] = []
            for item in value:
                text = str(item)
                stripped = _normalize_text(text)
                if stripped != text:
                    summary["trimmed_list_value_count"] += 1
                    row_trimmed = True
                normalized_items.append(stripped)
            updates[field] = normalized_items

        if row_trimmed:
            summary["trimmed_row_count"] += 1
        if issues:
            summary["invalid_row_count"] += 1
            if len(invalid_rows) < 10:
                invalid_rows.append({"index": index, "issues": issues})
        return updates

    if isinstance(dataset, list):
        normalized_rows: list[dict[str, Any]] = []
        for index, row in enumerate(dataset):
            row_mapping = dict(row) if isinstance(row, Mapping) else {}
            normalized_rows.append({**row_mapping, **normalize_row(row_mapping, index)})
        normalized_dataset = normalized_rows
    else:
        normalized_dataset = dataset.map(
            lambda row, index: normalize_row(row, index),
            with_indices=True,
        )

    return PreferenceDatasetNormalizationResult(
        dataset=normalized_dataset,
        summary=summary,
    )
