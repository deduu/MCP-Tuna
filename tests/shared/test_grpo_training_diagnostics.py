from __future__ import annotations

from shared.grpo_training_diagnostics import summarize_grpo_log_history


def test_summarize_grpo_log_history_tracks_reward_and_clipping_rates():
    summary = summarize_grpo_log_history(
        [
            {
                "reward": 0.5,
                "completions/clipped_ratio": 0.25,
                "completions/mean_length": 40.0,
                "completions/mean_terminated_length": 38.0,
                "completions/max_length": 48.0,
                "completions/max_terminated_length": 42.0,
            },
            {
                "reward": 0.0,
                "completions/clipped_ratio": 0.5,
                "completions/mean_length": 55.0,
                "completions/mean_terminated_length": 30.0,
                "completions/max_length": 64.0,
                "completions/max_terminated_length": 30.0,
            },
            {
                "reward": -0.25,
                "completions/clipped_ratio": 1.0,
                "completions/mean_length": 64.0,
                "completions/mean_terminated_length": 0.0,
                "completions/max_length": 64.0,
                "completions/max_terminated_length": 0.0,
            },
        ]
    )

    assert summary["steps_logged"] == 3
    assert summary["positive_reward_steps"] == 1
    assert summary["zero_reward_steps"] == 1
    assert summary["negative_reward_steps"] == 1
    assert summary["positive_reward_step_rate"] == 0.3333
    assert summary["clip_ratio_ge_0_5_steps"] == 2
    assert summary["clip_ratio_eq_1_0_steps"] == 1
    assert summary["avg_reward"] == 0.0833
    assert summary["avg_clipped_ratio"] == 0.5833
    assert summary["avg_mean_completion_length"] == 53.0
    assert summary["avg_mean_terminated_length"] == 22.6667
    assert summary["max_completion_length_observed"] == 64.0


def test_summarize_grpo_log_history_handles_empty_input():
    summary = summarize_grpo_log_history([])

    assert summary["steps_logged"] == 0
    assert summary["positive_reward_steps"] == 0
    assert summary["clip_ratio_ge_0_5_steps"] == 0
    assert summary["avg_reward"] is None
