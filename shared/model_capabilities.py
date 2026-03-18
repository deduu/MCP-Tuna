"""Helpers for inferring local model modality and capability metadata."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


TEXT_TRAINING_TECHNIQUES = ["sft", "dpo", "grpo", "kto", "curriculum", "sequential"]
VLM_TRAINING_TECHNIQUES = ["vlm_sft"]

_VLM_MODEL_MARKERS = (
    "qwen2.5-vl",
    "qwen-vl",
    "llava",
    "llava-next",
    "internvl",
    "idefics",
    "paligemma",
    "phi-3-vision",
    "phi3-vision",
    "minicpm-v",
    "cogvlm",
    "molmo",
    "deepseek-vl",
    "vision-language",
)


def infer_model_modality(
    identifier: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    file_names: Optional[Iterable[str]] = None,
) -> str:
    """Infer whether a model is text-only or vision-language."""
    haystack_parts = [identifier]

    if config:
        haystack_parts.append(str(config.get("model_type", "")))
        haystack_parts.extend(str(value) for value in (config.get("architectures") or []))
        if config.get("vision_config") or config.get("vision_feature_select_strategy"):
            return "vision-language"

    if file_names:
        haystack_parts.extend(str(name) for name in file_names)

    haystack = " ".join(haystack_parts).lower()
    if any(marker in haystack for marker in _VLM_MODEL_MARKERS):
        return "vision-language"

    return "text" if identifier.strip() else "unknown"


def build_model_capabilities(modality: str) -> Dict[str, Any]:
    """Return normalized capability metadata for a discovered model."""
    if modality == "vision-language":
        return {
            "modality": modality,
            "supported_techniques": list(VLM_TRAINING_TECHNIQUES),
            "supports_training": True,
            "supports_inference": True,
            "supports_deployment": True,
            "usable_for": ["training", "inference", "deployment"],
        }

    if modality == "text":
        return {
            "modality": modality,
            "supported_techniques": list(TEXT_TRAINING_TECHNIQUES),
            "supports_training": True,
            "supports_inference": True,
            "supports_deployment": True,
            "usable_for": ["training", "inference", "deployment"],
        }

    return {
        "modality": "unknown",
        "supported_techniques": [],
        "supports_training": False,
        "supports_inference": False,
        "supports_deployment": False,
        "usable_for": [],
    }
