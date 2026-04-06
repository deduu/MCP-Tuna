from __future__ import annotations

from typing import Any, Dict, List, Mapping

from shared.training_defaults import build_preference_starting_recipe


def _base_hidden_factors() -> List[str]:
    return [
        "Preference tuning usually works best when it starts from a strong SFT checkpoint instead of the raw base model.",
        "A clean-looking dataset can still underperform if the reward signal is too easy or if training runs too long for the dataset size.",
        "When preference tuning beats SFT, it is usually because the dataset teaches a sharper ranking signal, not just because you trained for more steps.",
    ]


def _dpo_starting_recipe(
    row_count: int,
    warnings: List[str],
    dpo: Mapping[str, Any],
) -> Dict[str, Any]:
    lower_data = row_count < 200
    repetitive_rejects = float(dpo.get("dominant_rejected_ratio", 0.0) or 0.0) >= 0.2
    weak_hard_negatives = float(dpo.get("hard_negative_ratio", 0.0) or 0.0) < 0.25
    low_overlap = float(dpo.get("low_overlap_ratio", 0.0) or 0.0) >= 0.35

    config = {
        **build_preference_starting_recipe("dpo", row_count),
        "batching": "Keep batch size small and prefer gradient accumulation if GPU memory is tight.",
        "evaluation": "Benchmark against the exact SFT checkpoint you started from using deterministic decoding.",
    }

    watchouts = [
        "Users often forget that DPO can get worse when the dataset is small but training still runs for multiple epochs.",
        "If chosen and rejected answers are too far apart, DPO learns obvious filtering instead of better ranking.",
        "If many rejected answers reuse one template, DPO may memorize that one failure mode and still answer badly on new prompts.",
    ]
    if repetitive_rejects:
        watchouts.append(
            "Your rejected answers are still concentrated around a few patterns, so improving negative diversity is more important than increasing steps."
        )
    if weak_hard_negatives:
        watchouts.append(
            "Your DPO pairs are too easy; on-topic but slightly worse negatives usually help more than obviously bad replies."
        )
    if low_overlap:
        watchouts.append(
            "Many rejected answers are too far from the chosen answer, so DPO may learn off-topic rejection instead of subtle preference ranking."
        )

    actions = [
        "Start from the best available SFT adapter, not the base model.",
        "Use a short first run and only increase epochs after the first benchmark shows a real gain over SFT.",
    ]
    if repetitive_rejects:
        actions.append("Add more distinct rejected failure modes before increasing training budget.")
    if weak_hard_negatives:
        actions.append("Rewrite more rejected answers so they stay on-topic but miss nuance, specificity, or policy constraints.")

    if not warnings:
        actions.append("The dataset structure looks healthy, so the next variable to tune is training budget rather than schema cleanup.")

    return {
        "headline": (
            "Safe starting point: short DPO continuation from SFT."
            if lower_data
            else "Start with a conservative DPO continuation and benchmark against SFT."
        ),
        "starting_recipe": config,
        "hidden_factors": watchouts,
        "recommended_actions": actions,
    }


def _grpo_starting_recipe(
    row_count: int,
    warnings: List[str],
    grpo: Mapping[str, Any],
) -> Dict[str, Any]:
    lower_data = row_count < 100
    low_variance = float(grpo.get("zero_reward_variance_ratio", 0.0) or 0.0) >= 0.2
    low_diversity = float(grpo.get("response_stats", {}).get("unique_ratio", 0.0) or 0.0) < 0.7
    too_few = float(grpo.get("avg_responses_per_row", 0.0) or 0.0) < 4.0

    config = {
        **build_preference_starting_recipe("grpo", row_count),
        "num_generations": "4 to 6" if too_few else "keep current candidate count",
        "reward_design": "Make sure each prompt contains clearly better and worse candidates with numeric separation.",
        "evaluation": "Check whether generated responses still match the candidate style you rewarded, not just whether training completed.",
    }

    watchouts = [
        "Users often assume GRPO is learning just because it runs, but low reward variance can make the update nearly useless.",
        "If generated completions drift too far from your rewarded candidates, training can consume budget without improving behavior.",
        "GRPO quality depends heavily on candidate diversity and reward spread, not only on epoch count.",
    ]
    if low_variance:
        watchouts.append("Many rows have weak reward separation, so adding more steps will likely not help much.")
    if low_diversity:
        watchouts.append("Response diversity is low, so the model may relearn one style rather than improving preference ranking.")

    actions = [
        "Start from the best available SFT adapter.",
        "Prioritize reward coverage and candidate diversity before increasing run length.",
    ]
    if too_few:
        actions.append("Increase candidate responses per prompt so GRPO sees a stronger ranking signal.")
    if low_variance:
        actions.append("Spread reward values more clearly within each prompt instead of using nearly tied scores.")

    if not warnings:
        actions.append("The dataset structure looks healthy, so the next check should be reward coverage during training.")

    return {
        "headline": "Safe starting point: GRPO continuation from SFT with explicit reward spread.",
        "starting_recipe": config,
        "hidden_factors": watchouts,
        "recommended_actions": actions,
    }


def _kto_starting_recipe(
    row_count: int,
    warnings: List[str],
    kto: Mapping[str, Any],
) -> Dict[str, Any]:
    lower_data = row_count < 100
    skew = min(
        float(kto.get("positive_ratio", 0.0) or 0.0),
        float(kto.get("negative_ratio", 0.0) or 0.0),
    ) < 0.25

    actions = [
        "Start from SFT instead of the raw base model.",
        "Keep both positive and negative labels represented in the same task distribution.",
    ]
    if skew:
        actions.append("Improve label balance before increasing epochs.")
    if not warnings:
        actions.append("The dataset structure looks healthy, so tune run budget conservatively and benchmark against SFT.")

    return {
        "headline": "Safe starting point: conservative KTO continuation from SFT.",
        "starting_recipe": {
            **build_preference_starting_recipe("kto", row_count),
            "evaluation": "Compare against SFT on a held-out pack before increasing budget.",
        },
        "hidden_factors": _base_hidden_factors(),
        "recommended_actions": actions,
    }


def build_preference_training_guidance(
    *,
    technique: str,
    row_count: int,
    warnings: List[str],
    prompt_stats: Mapping[str, Any],
    technique_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    common = {
        "missed_contributors": _base_hidden_factors(),
        "prompt_diversity_ratio": prompt_stats.get("unique_ratio", 0.0),
    }

    if technique == "dpo":
        guidance = _dpo_starting_recipe(row_count, warnings, technique_stats)
    elif technique == "grpo":
        guidance = _grpo_starting_recipe(row_count, warnings, technique_stats)
    else:
        guidance = _kto_starting_recipe(row_count, warnings, technique_stats)

    merged = dict(common)
    merged.update(guidance)
    return merged
