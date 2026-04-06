from __future__ import annotations

from statistics import mean
from typing import Any, Mapping, Sequence


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def summarize_grpo_log_history(
    log_history: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    entries = [entry for entry in (log_history or []) if isinstance(entry, Mapping)]
    if not entries:
        return {
            "steps_logged": 0,
            "positive_reward_steps": 0,
            "zero_reward_steps": 0,
            "negative_reward_steps": 0,
            "positive_reward_step_rate": 0.0,
            "clip_ratio_ge_0_5_steps": 0,
            "clip_ratio_eq_1_0_steps": 0,
            "clip_ratio_ge_0_5_rate": 0.0,
            "clip_ratio_eq_1_0_rate": 0.0,
            "avg_reward": None,
            "avg_clipped_ratio": None,
            "avg_mean_completion_length": None,
            "avg_mean_terminated_length": None,
            "max_clipped_ratio": None,
            "max_completion_length_observed": None,
            "max_terminated_length_observed": None,
        }

    rewards = [_as_float(entry.get("reward")) for entry in entries]
    reward_values = [value for value in rewards if value is not None]
    clipped_ratios = [
        value
        for value in (
            _as_float(entry.get("completions/clipped_ratio")) for entry in entries
        )
        if value is not None
    ]
    mean_completion_lengths = [
        value
        for value in (
            _as_float(entry.get("completions/mean_length")) for entry in entries
        )
        if value is not None
    ]
    mean_terminated_lengths = [
        value
        for value in (
            _as_float(entry.get("completions/mean_terminated_length"))
            for entry in entries
        )
        if value is not None
    ]
    max_completion_lengths = [
        value
        for value in (
            _as_float(entry.get("completions/max_length")) for entry in entries
        )
        if value is not None
    ]
    max_terminated_lengths = [
        value
        for value in (
            _as_float(entry.get("completions/max_terminated_length"))
            for entry in entries
        )
        if value is not None
    ]
    step_count = len(entries)
    positive_reward_steps = sum(1 for value in reward_values if value > 0)
    zero_reward_steps = sum(1 for value in reward_values if value == 0)
    negative_reward_steps = sum(1 for value in reward_values if value < 0)
    clipped_half_steps = sum(1 for value in clipped_ratios if value >= 0.5)
    clipped_full_steps = sum(1 for value in clipped_ratios if value >= 1.0)
    return {
        "steps_logged": step_count,
        "positive_reward_steps": positive_reward_steps,
        "zero_reward_steps": zero_reward_steps,
        "negative_reward_steps": negative_reward_steps,
        "positive_reward_step_rate": _round(positive_reward_steps / step_count),
        "clip_ratio_ge_0_5_steps": clipped_half_steps,
        "clip_ratio_eq_1_0_steps": clipped_full_steps,
        "clip_ratio_ge_0_5_rate": _round(clipped_half_steps / step_count),
        "clip_ratio_eq_1_0_rate": _round(clipped_full_steps / step_count),
        "avg_reward": _round(mean(reward_values)) if reward_values else None,
        "avg_clipped_ratio": _round(mean(clipped_ratios)) if clipped_ratios else None,
        "avg_mean_completion_length": (
            _round(mean(mean_completion_lengths)) if mean_completion_lengths else None
        ),
        "avg_mean_terminated_length": (
            _round(mean(mean_terminated_lengths)) if mean_terminated_lengths else None
        ),
        "max_clipped_ratio": max(clipped_ratios) if clipped_ratios else None,
        "max_completion_length_observed": (
            max(max_completion_lengths) if max_completion_lengths else None
        ),
        "max_terminated_length_observed": (
            max(max_terminated_lengths) if max_terminated_lengths else None
        ),
    }
