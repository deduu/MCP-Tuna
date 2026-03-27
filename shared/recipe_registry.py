"""Shared registries for dataset-composition and training recipe presets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional


HF_DATASET_RECIPES: Dict[str, Dict[str, Any]] = {
    "tiny_reasoning_stage_1": {
        "description": (
            "Stage 1 published non-reasoning SFT dataset from the "
            "tiny-reasoning-language-model workflow. Produces prompt/response rows."
        ),
        "target_format": "sft",
        "sources": [
            {
                "dataset_name": "Shekswess/trlm-sft-stage-1-final-2",
                "split": "train",
            },
        ],
    },
    "tiny_reasoning_stage_2": {
        "description": (
            "Stage 2 published reasoning SFT dataset from the "
            "tiny-reasoning-language-model workflow. Produces prompt/response rows."
        ),
        "target_format": "sft",
        "sources": [
            {
                "dataset_name": "Shekswess/trlm-sft-stage-2-final-2",
                "split": "train",
            },
        ],
    },
    "tiny_reasoning_stage_3": {
        "description": (
            "Stage 3 published DPO preference dataset from the "
            "tiny-reasoning-language-model workflow. Produces prompt/chosen/rejected rows."
        ),
        "target_format": "dpo",
        "sources": [
            {
                "dataset_name": "Shekswess/trlm-dpo-stage-3-final-2",
                "split": "train",
            },
        ],
    },
}


TRAINING_RECIPES: Dict[str, Dict[str, Any]] = {
    "tiny_reasoning_stage_1": {
        "description": "Stage 1 non-reasoning SFT preset.",
        "system_prompt": (
            "You are Tiny Reasoning Language Model. Be helpful and concise. "
            "Do not reveal chain-of-thought."
        ),
        "special_tokens": [],
        "sft_defaults": {
            "report_to": ["wandb"],
        },
        "dpo_defaults": {},
    },
    "tiny_reasoning_stage_2": {
        "description": "Stage 2 reasoning SFT preset with explicit think tags.",
        "system_prompt": (
            "You are Tiny Reasoning Language Model. When reasoning is useful, "
            "write your thoughts inside <think>...</think> before the final answer."
        ),
        "special_tokens": ["<think>", "</think>"],
        "sft_defaults": {
            "report_to": ["wandb"],
        },
        "dpo_defaults": {},
    },
    "tiny_reasoning_stage_3": {
        "description": "Stage 3 DPO preset inspired by the upstream recipe.",
        "system_prompt": None,
        "special_tokens": [],
        "sft_defaults": {},
        "dpo_defaults": {
            "learning_rate": 1e-5,
            "lr_scheduler_type": "cosine",
            "max_grad_norm": 0.2,
            "report_to": ["wandb"],
        },
    },
}


def list_hf_dataset_recipes() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "description": recipe["description"],
            "target_format": recipe["target_format"],
            "source_count": len(recipe["sources"]),
        }
        for name, recipe in HF_DATASET_RECIPES.items()
    ]


def get_hf_dataset_recipe(name: str) -> Optional[Dict[str, Any]]:
    recipe = HF_DATASET_RECIPES.get(name)
    return deepcopy(recipe) if recipe is not None else None


def list_training_recipes() -> List[Dict[str, Any]]:
    return [
        {
            "name": name,
            "description": recipe["description"],
            "special_tokens": list(recipe.get("special_tokens", [])),
        }
        for name, recipe in TRAINING_RECIPES.items()
    ]


def get_training_recipe(name: str) -> Optional[Dict[str, Any]]:
    recipe = TRAINING_RECIPES.get(name)
    return deepcopy(recipe) if recipe is not None else None
