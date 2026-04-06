from __future__ import annotations

from shared.training_defaults import (
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    auto_tune_preference_training_defaults,
    build_preference_starting_recipe,
)


def test_build_preference_starting_recipe_is_conservative_for_small_dpo_dataset():
    recipe = build_preference_starting_recipe("dpo", 110)

    assert recipe["start_from_sft_checkpoint"] is True
    assert recipe["epochs"] == 1
    assert recipe["learning_rate"] == 1e-4


def test_auto_tune_preference_training_defaults_adjusts_only_generic_defaults():
    summary = auto_tune_preference_training_defaults(
        technique="dpo",
        row_count=110,
        num_epochs=DEFAULT_NUM_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
        auto_tune_defaults=True,
    )

    assert summary["applied"] is True
    assert summary["effective"]["num_epochs"] == 1
    assert summary["effective"]["learning_rate"] == 1e-4
    assert summary["adjustments"]["num_epochs"]["from"] == DEFAULT_NUM_EPOCHS
    assert summary["adjustments"]["learning_rate"]["from"] == DEFAULT_LEARNING_RATE


def test_auto_tune_preference_training_defaults_preserves_custom_values():
    summary = auto_tune_preference_training_defaults(
        technique="dpo",
        row_count=110,
        num_epochs=2,
        learning_rate=5e-5,
        auto_tune_defaults=True,
    )

    assert summary["applied"] is False
    assert summary["effective"]["num_epochs"] == 2
    assert summary["effective"]["learning_rate"] == 5e-5


def test_auto_tune_preference_training_defaults_can_be_disabled():
    summary = auto_tune_preference_training_defaults(
        technique="grpo",
        row_count=55,
        num_epochs=DEFAULT_NUM_EPOCHS,
        learning_rate=DEFAULT_LEARNING_RATE,
        auto_tune_defaults=False,
    )

    assert summary["enabled"] is False
    assert summary["applied"] is False
    assert summary["effective"]["num_epochs"] == DEFAULT_NUM_EPOCHS
    assert summary["effective"]["learning_rate"] == DEFAULT_LEARNING_RATE
