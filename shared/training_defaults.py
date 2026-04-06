"""Shared training defaults used across gateway, orchestrators, and scripts."""
from __future__ import annotations

import math
from typing import Any, Optional

DEFAULT_NUM_EPOCHS = 3
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE = 1
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 4
DEFAULT_DPO_MAX_PROMPT_LENGTH = 384
DEFAULT_DPO_MAX_LENGTH = 512
DEFAULT_PREFERENCE_WEIGHT_DECAY = 0.01
DEFAULT_PREFERENCE_MAX_GRAD_NORM = 0.0
DEFAULT_PREFERENCE_GRADIENT_CHECKPOINTING = True
DEFAULT_SMALL_PREFERENCE_NUM_EPOCHS = 1
DEFAULT_SMALL_PREFERENCE_LEARNING_RATE = 1e-4
DEFAULT_GRPO_TRUNCATION_PENALTY = 0.15

_PREFERENCE_SMALL_DATASET_ROW_THRESHOLDS = {
    "dpo": 200,
    "grpo": 100,
    "kto": 100,
}


def estimate_max_steps_for_epochs(
    num_examples: int,
    per_device_train_batch_size: int = DEFAULT_PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
) -> int:
    """Approximate optimizer steps for a given dataset size and epoch target."""
    effective_batch = max(1, int(per_device_train_batch_size))
    accumulation = max(1, int(gradient_accumulation_steps))
    epochs = max(1, int(num_epochs))
    if num_examples <= 0:
        return epochs

    micro_batches = math.ceil(num_examples / effective_batch)
    steps_per_epoch = max(1, math.ceil(micro_batches / accumulation))
    return max(1, steps_per_epoch * epochs)


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_default_epoch_value(value: Any) -> bool:
    parsed = _safe_int(value)
    return parsed is None or parsed == DEFAULT_NUM_EPOCHS


def _is_default_learning_rate(value: Any) -> bool:
    parsed = _safe_float(value)
    if parsed is None:
        return True
    return math.isclose(parsed, DEFAULT_LEARNING_RATE, rel_tol=0.0, abs_tol=1e-12)


def get_small_preference_dataset_threshold(technique: str) -> int | None:
    return _PREFERENCE_SMALL_DATASET_ROW_THRESHOLDS.get(str(technique or "").strip().lower())


def build_preference_starting_recipe(
    technique: str,
    row_count: int,
) -> dict[str, Any]:
    normalized = str(technique or "").strip().lower()
    threshold = get_small_preference_dataset_threshold(normalized)
    lower_data = threshold is not None and row_count > 0 and row_count < threshold

    recipe: dict[str, Any] = {
        "start_from_sft_checkpoint": True,
        "epochs": DEFAULT_SMALL_PREFERENCE_NUM_EPOCHS if lower_data else DEFAULT_NUM_EPOCHS,
        "learning_rate": (
            DEFAULT_SMALL_PREFERENCE_LEARNING_RATE
            if lower_data
            else DEFAULT_LEARNING_RATE
        ),
    }

    if normalized == "grpo":
        recipe["epochs"] = DEFAULT_SMALL_PREFERENCE_NUM_EPOCHS

    return recipe


def auto_tune_preference_training_defaults(
    *,
    technique: str,
    row_count: int,
    num_epochs: Any = None,
    learning_rate: Any = None,
    auto_tune_defaults: bool = True,
) -> dict[str, Any]:
    normalized = str(technique or "").strip().lower()
    threshold = get_small_preference_dataset_threshold(normalized)
    recommended = build_preference_starting_recipe(normalized, row_count)
    effective_num_epochs = _safe_int(num_epochs) or DEFAULT_NUM_EPOCHS
    effective_learning_rate = _safe_float(learning_rate) or DEFAULT_LEARNING_RATE
    adjustments: dict[str, dict[str, Any]] = {}

    lower_data = threshold is not None and row_count > 0 and row_count < threshold
    if auto_tune_defaults and lower_data:
        if _is_default_epoch_value(num_epochs):
            tuned_epochs = int(recommended["epochs"])
            if tuned_epochs != effective_num_epochs:
                adjustments["num_epochs"] = {
                    "from": effective_num_epochs,
                    "to": tuned_epochs,
                }
                effective_num_epochs = tuned_epochs

        if _is_default_learning_rate(learning_rate):
            tuned_learning_rate = float(recommended["learning_rate"])
            if not math.isclose(
                tuned_learning_rate,
                effective_learning_rate,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                adjustments["learning_rate"] = {
                    "from": effective_learning_rate,
                    "to": tuned_learning_rate,
                }
                effective_learning_rate = tuned_learning_rate

    return {
        "enabled": bool(auto_tune_defaults),
        "applied": bool(adjustments),
        "technique": normalized,
        "row_count": max(0, int(row_count)),
        "small_dataset_threshold": threshold,
        "recommended": recommended,
        "effective": {
            "num_epochs": effective_num_epochs,
            "learning_rate": effective_learning_rate,
        },
        "adjustments": adjustments,
        "reason": (
            f"small_{normalized}_dataset"
            if adjustments
            else None
        ),
    }
