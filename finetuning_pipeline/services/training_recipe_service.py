"""Helpers for optional training recipe presets."""

from __future__ import annotations

from typing import Any, Dict, Optional

from shared.recipe_registry import get_training_recipe, list_training_recipes


class TrainingRecipeService:
    """Resolves opt-in training recipe defaults without changing baseline behavior."""

    @staticmethod
    def list_recipes() -> Dict[str, Any]:
        return {
            "success": True,
            "recipes": list_training_recipes(),
        }

    @staticmethod
    def get_recipe(recipe_name: Optional[str]) -> Optional[Dict[str, Any]]:
        if not recipe_name:
            return None
        return get_training_recipe(recipe_name)

    @staticmethod
    def apply_defaults(
        *,
        recipe_name: Optional[str],
        trainer_type: str,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        recipe = TrainingRecipeService.get_recipe(recipe_name)
        if recipe is None:
            return kwargs

        defaults = recipe.get(f"{trainer_type}_defaults", {}) or {}
        merged = dict(kwargs)
        for key, value in defaults.items():
            merged.setdefault(key, value)
        return merged
