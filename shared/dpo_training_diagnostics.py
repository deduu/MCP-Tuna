from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Optional, Sequence


def _dataset_length(dataset: Any) -> int | None:
    try:
        return int(len(dataset))
    except Exception:
        return None


def _sample_rows(dataset: Any, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(dataset, list):
        iterable = dataset[:limit]
        for row in iterable:
            if isinstance(row, Mapping):
                rows.append(dict(row))
        return rows

    length = _dataset_length(dataset)
    if length is None:
        return rows

    for index in range(min(length, limit)):
        try:
            row = dataset[index]
        except Exception:
            break
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows


def _preview(text: str, limit: int) -> str:
    flattened = " ".join(str(text or "").split())
    if len(flattened) <= limit:
        return flattened
    return f"{flattened[:limit - 3].rstrip()}..."


def _token_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
    if isinstance(input_ids, Sequence) and not isinstance(input_ids, (str, bytes)):
        return len(input_ids)
    return 0


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _sanitize_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_sanitize_json(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _sanitize_json(value.tolist())
        except Exception:
            pass
    return str(value)


def serialize_config_object(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        try:
            return _sanitize_json(config.to_dict())
        except Exception:
            pass
    if hasattr(config, "__dict__"):
        try:
            return _sanitize_json(
                {
                    key: value
                    for key, value in vars(config).items()
                    if not key.startswith("_")
                }
            )
        except Exception:
            pass
    return {"value": _sanitize_json(config)}


def _summarize_trainer_value(value: Any, *, preview_chars: int = 120) -> Any:
    if isinstance(value, str):
        return {
            "type": "str",
            "length": len(value),
            "preview": _preview(value, preview_chars),
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "preview": _sanitize_json(list(value[:8])),
        }
    if isinstance(value, Mapping):
        return {
            "type": "dict",
            "keys": sorted(str(key) for key in value.keys()),
        }
    return _sanitize_json(value)


def summarize_dpo_trainer_dataset(
    dataset: Any,
    *,
    sample_limit: int = 3,
) -> dict[str, Any]:
    sample_rows = _sample_rows(dataset, sample_limit)
    return {
        "schema_version": 1,
        "num_rows": _dataset_length(dataset),
        "column_names": list(getattr(dataset, "column_names", []) or []),
        "sample_size": len(sample_rows),
        "samples": [
            {
                "index": index,
                "fields": {
                    key: _summarize_trainer_value(value)
                    for key, value in row.items()
                },
            }
            for index, row in enumerate(sample_rows)
        ],
    }


def summarize_dpo_preprocessing(
    dataset: Any,
    tokenizer: Any,
    *,
    max_prompt_length: int,
    max_length: int,
    sample_limit: int = 16,
    preview_chars: int = 160,
) -> dict[str, Any]:
    sample_rows = _sample_rows(dataset, sample_limit)
    prompt_lengths: list[int] = []
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []
    combined_chosen_lengths: list[int] = []
    combined_rejected_lengths: list[int] = []
    prompt_overflow_count = 0
    chosen_response_overflow_count = 0
    rejected_response_overflow_count = 0
    combined_chosen_overflow_count = 0
    combined_rejected_overflow_count = 0
    samples: list[dict[str, Any]] = []

    for index, row in enumerate(sample_rows):
        prompt = str(row.get("prompt") or "").strip()
        chosen = str(row.get("chosen") or "").strip()
        rejected = str(row.get("rejected") or "").strip()

        prompt_tokens = _token_length(tokenizer, prompt)
        chosen_tokens = _token_length(tokenizer, chosen)
        rejected_tokens = _token_length(tokenizer, rejected)
        combined_chosen_tokens = prompt_tokens + chosen_tokens
        combined_rejected_tokens = prompt_tokens + rejected_tokens
        effective_response_budget = max(max_length - min(prompt_tokens, max_prompt_length), 0)

        prompt_lengths.append(prompt_tokens)
        chosen_lengths.append(chosen_tokens)
        rejected_lengths.append(rejected_tokens)
        combined_chosen_lengths.append(combined_chosen_tokens)
        combined_rejected_lengths.append(combined_rejected_tokens)

        prompt_overflow = prompt_tokens > max_prompt_length
        chosen_response_overflow = chosen_tokens > effective_response_budget
        rejected_response_overflow = rejected_tokens > effective_response_budget
        combined_chosen_overflow = combined_chosen_tokens > max_length
        combined_rejected_overflow = combined_rejected_tokens > max_length

        prompt_overflow_count += int(prompt_overflow)
        chosen_response_overflow_count += int(chosen_response_overflow)
        rejected_response_overflow_count += int(rejected_response_overflow)
        combined_chosen_overflow_count += int(combined_chosen_overflow)
        combined_rejected_overflow_count += int(combined_rejected_overflow)

        samples.append(
            {
                "index": index,
                "prompt_preview": _preview(prompt, preview_chars),
                "chosen_preview": _preview(chosen, preview_chars),
                "rejected_preview": _preview(rejected, preview_chars),
                "prompt_tokens": prompt_tokens,
                "chosen_tokens": chosen_tokens,
                "rejected_tokens": rejected_tokens,
                "combined_chosen_tokens": combined_chosen_tokens,
                "combined_rejected_tokens": combined_rejected_tokens,
                "effective_response_budget": effective_response_budget,
                "prompt_overflow": prompt_overflow,
                "chosen_response_overflow": chosen_response_overflow,
                "rejected_response_overflow": rejected_response_overflow,
                "combined_chosen_overflow": combined_chosen_overflow,
                "combined_rejected_overflow": combined_rejected_overflow,
            }
        )

    total = len(sample_rows)

    def avg(values: Sequence[int]) -> float | None:
        return round(mean(values), 2) if values else None

    def ratio(count: int) -> float | None:
        return _rounded(count / total) if total else None

    return {
        "schema_version": 1,
        "num_examples": _dataset_length(dataset),
        "sample_size": total,
        "max_prompt_length": max_prompt_length,
        "max_length": max_length,
        "avg_prompt_tokens": avg(prompt_lengths),
        "avg_chosen_tokens": avg(chosen_lengths),
        "avg_rejected_tokens": avg(rejected_lengths),
        "avg_combined_chosen_tokens": avg(combined_chosen_lengths),
        "avg_combined_rejected_tokens": avg(combined_rejected_lengths),
        "prompt_overflow_ratio": ratio(prompt_overflow_count),
        "chosen_response_overflow_ratio": ratio(chosen_response_overflow_count),
        "rejected_response_overflow_ratio": ratio(rejected_response_overflow_count),
        "combined_chosen_overflow_ratio": ratio(combined_chosen_overflow_count),
        "combined_rejected_overflow_ratio": ratio(combined_rejected_overflow_count),
        "samples": samples,
    }
