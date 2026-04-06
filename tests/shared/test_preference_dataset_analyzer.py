from __future__ import annotations

import json
from pathlib import Path

import pytest

from shared.preference_dataset_analyzer import PreferenceDatasetAnalyzer


@pytest.mark.asyncio
async def test_analyze_dpo_dataset_flags_repetitive_rejections(tmp_path: Path):
    dataset_path = tmp_path / "dpo.jsonl"
    rows = [
        {
            "prompt": "How does the workflow help sales teams?",
            "chosen": "It centralizes WhatsApp chats, routing, and follow-up reminders for the team.",
            "rejected": "I do not know.",
        },
        {
            "prompt": "Can small teams use it?",
            "chosen": "Yes, small teams can coordinate leads without sharing one phone.",
            "rejected": "I do not know.",
        },
        {
            "prompt": "What does it replace?",
            "chosen": "It replaces scattered chats with one shared inbox and audit trail.",
            "rejected": "It replaces scattered chats with one shared inbox and audit trail.",
        },
    ]
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = await PreferenceDatasetAnalyzer().analyze(str(dataset_path), technique="dpo")

    assert result["success"] is True
    assert result["technique_analyzed"] == "dpo"
    assert result["status"] == "warn"
    assert result["dpo"]["rejected_stats"]["unique_count"] == 2
    assert result["dpo"]["identical_pair_rows"] == 1
    assert result["dpo"]["dominant_rejected_count"] == 2
    assert result["dpo"]["dominant_rejected_ratio"] == 0.6667
    assert any("Rejected responses are highly repetitive" in warning for warning in result["warnings"])
    assert any("dominates a large share" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_analyze_dpo_allows_multi_negative_prompt_repeats(tmp_path: Path):
    dataset_path = tmp_path / "dpo_multi_negative.jsonl"
    rows = []
    for index in range(20):
        prompt = f"How should sales rep {index} handle WhatsApp follow-up?"
        chosen = (
            "Use one shared inbox, keep follow-up visible to the team, and avoid promising outcomes you cannot guarantee."
        )
        rows.extend(
            [
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": "Promise a guaranteed sales increase after the first campaign.",
                },
                {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": "Give a vague answer and tell them to figure it out themselves.",
                },
            ]
        )
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = await PreferenceDatasetAnalyzer().analyze(str(dataset_path), technique="dpo")

    assert result["success"] is True
    assert result["status"] == "warn"
    assert result["dpo"]["avg_rows_per_prompt"] == 2.0
    assert result["dpo"]["multi_variant_prompt_ratio"] == 1.0
    assert not any("Prompt diversity is low relative to analyzed rows" in warning for warning in result["warnings"])


@pytest.mark.asyncio
async def test_analyze_dpo_dataset_tracks_hard_negative_coverage(tmp_path: Path):
    dataset_path = tmp_path / "dpo_hard_negatives.jsonl"
    rows = [
        {
            "prompt": "How should we answer a pricing question?",
            "chosen": "Explain that pricing depends on business needs and route the lead to sales for a tailored quote.",
            "rejected": "Say pricing depends on their needs, but guess a rough quote yourself instead of routing to sales.",
        },
        {
            "prompt": "How should we answer a closing guarantee question?",
            "chosen": "Do not promise closing percentages; focus on faster follow-up and better team visibility.",
            "rejected": "Avoid promising fixed closing percentages, but imply follow-up and visibility will probably improve results after setup.",
        },
        {
            "prompt": "How should we answer an admin replacement question?",
            "chosen": "Clarify that the system supports admins with triage and monitoring rather than replacing them.",
            "rejected": "Say the system helps admins with triage and monitoring, but hint that most admin work disappears.",
        },
        {
            "prompt": "How should we answer a small business fit question?",
            "chosen": "Say it can fit small teams when chat volume and follow-up needs justify shared visibility.",
            "rejected": "Say it can fit small teams, but oversell it as automatically ideal for every small business.",
        },
    ]
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = await PreferenceDatasetAnalyzer().analyze(str(dataset_path), technique="dpo")

    assert result["success"] is True
    assert result["dpo"]["hard_negative_rows"] == 4
    assert result["dpo"]["hard_negative_ratio"] == 1.0
    assert result["dpo"]["low_overlap_rows"] == 0
    assert not any("Too few DPO pairs look like hard negatives" in warning for warning in result["warnings"])
    assert result["guidance"]["starting_recipe"]["start_from_sft_checkpoint"] is True
    assert result["guidance"]["starting_recipe"]["epochs"] == 1
    assert result["guidance"]["starting_recipe"]["learning_rate"] == 1e-4
    assert any("Start from the best available SFT adapter" in item for item in result["guidance"]["recommended_actions"])


@pytest.mark.asyncio
async def test_analyze_grpo_dataset_flags_reward_signal_issues(tmp_path: Path):
    dataset_path = tmp_path / "grpo.jsonl"
    rows = [
        {
            "prompt": "Handle a pricing question.",
            "responses": ["Ask about business size first.", "Ask about business size first."],
            "rewards": [1.0, 1.0],
        },
        {
            "prompt": "Handle a support question.",
            "responses": ["Route to support.", "Promise a discount."],
            "rewards": [0.4],
        },
    ]
    dataset_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    result = await PreferenceDatasetAnalyzer().analyze(str(dataset_path), technique="grpo")

    assert result["success"] is True
    assert result["technique_analyzed"] == "grpo"
    assert result["status"] == "warn"
    assert result["grpo"]["mismatched_rows"] == 1
    assert result["grpo"]["zero_reward_variance_rows"] == 1
    assert result["grpo"]["identical_response_rows"] == 1
    assert any("reward variance" in warning.lower() for warning in result["warnings"])
    assert result["guidance"]["starting_recipe"]["start_from_sft_checkpoint"] is True
    assert result["guidance"]["starting_recipe"]["learning_rate"] == 1e-4
    assert any("reward coverage" in item.lower() or "reward" in item.lower() for item in result["guidance"]["recommended_actions"])
