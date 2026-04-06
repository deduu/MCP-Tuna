from shared.preference_pair_builder import build_dpo_pairs_from_ranked_preferences


def test_build_dpo_pairs_from_ranked_preferences_adjacent_top_k_limits_pairs():
    pairs = build_dpo_pairs_from_ranked_preferences(
        "Prompt",
        ["best", "second", "third", "bad"],
        [1.0, 0.82, 0.7, -0.2],
        strategy="adjacent",
        top_k=3,
        max_pairs=2,
        min_reward_gap=0.05,
    )

    assert len(pairs) == 2
    assert pairs[0]["chosen"] == "best"
    assert pairs[0]["rejected"] == "second"
    assert pairs[0]["reward_gap"] == 0.18
    assert pairs[1]["chosen"] == "second"
    assert pairs[1]["rejected"] == "third"
    assert pairs[1]["reward_gap"] == 0.12


def test_build_dpo_pairs_from_ranked_preferences_dedupes_and_skips_non_improving_pairs():
    pairs = build_dpo_pairs_from_ranked_preferences(
        "Prompt",
        ["best", "best", "equal", "equal", "worse"],
        [1.0, 0.8, 0.5, 0.5, 0.2],
        strategy="best_vs_all",
        min_reward_gap=0.25,
    )

    assert len(pairs) == 2
    assert [pair["rejected"] for pair in pairs] == ["equal", "worse"]
