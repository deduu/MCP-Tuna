from __future__ import annotations

from shared.preference_reward_lookup import (
    build_grpo_reward_lookup,
    resolve_completion_termination_ids,
)


def test_reward_lookup_matches_normalized_completion_variants():
    lookup = build_grpo_reward_lookup(
        [
            {
                "prompt": "System: jawab singkat. User: Salestify itu apa? Assistant:",
                "responses": [
                    "Salestify membantu bisnis menangani chat WhatsApp lebih rapi.",
                    "Salestify adalah aplikasi game.",
                ],
                "rewards": [1.0, 0.0],
            }
        ]
    )

    match = lookup.match(
        "System: jawab singkat. User: Salestify itu apa? Assistant:",
        "Assistant: Salestify membantu bisnis menangani chat WhatsApp lebih rapi. <|eot_id|>",
    )

    assert match.reward == 1.0
    assert match.matched_via == "exact"
    assert match.similarity == 1.0


def test_reward_lookup_can_fallback_to_similarity_match():
    lookup = build_grpo_reward_lookup(
        [
            {
                "prompt": "Bagaimana jawab pertanyaan follow-up?",
                "responses": [
                    "Bisa bantu follow-up lebih cepat dan rapi untuk tim sales.",
                    "Tidak perlu follow-up sama sekali.",
                ],
                "rewards": [0.9, -0.5],
            }
        ]
    )

    match = lookup.match(
        "Bagaimana jawab pertanyaan follow-up?",
        "Bisa membantu follow-up lebih cepat dan lebih rapi untuk tim sales.",
    )

    assert match.reward == 0.9
    assert match.matched_via == "similarity"
    assert match.similarity >= 0.55


def test_reward_lookup_returns_zero_for_unrelated_completion():
    lookup = build_grpo_reward_lookup(
        [
            {
                "prompt": "Bagaimana jawab pertanyaan harga?",
                "responses": [
                    "Arahkan ke tim sales untuk penawaran yang sesuai kebutuhan.",
                ],
                "rewards": [1.0],
            }
        ]
    )

    match = lookup.match(
        "Bagaimana jawab pertanyaan harga?",
        "Cuaca hari ini cukup cerah dan tidak ada hubungannya dengan penjualan.",
    )

    assert match.reward == 0.0
    assert match.matched_via == "none"


def test_reward_lookup_tracks_match_statistics():
    lookup = build_grpo_reward_lookup(
        [
            {
                "prompt": "Bagaimana jawab pertanyaan follow-up?",
                "responses": [
                    "Bisa bantu follow-up lebih cepat dan rapi untuk tim sales.",
                    "Tidak perlu follow-up sama sekali.",
                ],
                "rewards": [0.9, -0.5],
            }
        ]
    )

    lookup.match(
        "Bagaimana jawab pertanyaan follow-up?",
        "Assistant: Bisa bantu follow-up lebih cepat dan rapi untuk tim sales.",
    )
    lookup.match(
        "Bagaimana jawab pertanyaan follow-up?",
        "Bisa membantu follow-up lebih cepat dan lebih rapi untuk tim sales.",
    )
    lookup.match(
        "Bagaimana jawab pertanyaan follow-up?",
        "Cuaca hari ini cukup cerah dan tidak ada hubungannya dengan penjualan.",
    )

    stats = lookup.stats_snapshot()
    assert stats["queries"] == 3
    assert stats["exact_matches"] == 1
    assert stats["similarity_matches"] == 1
    assert stats["misses"] == 1
    assert stats["positive_rewards"] == 2
    assert stats["zero_rewards"] == 1
    assert stats["match_rate"] == 0.6667


def test_reward_lookup_penalizes_truncated_unmatched_completions():
    lookup = build_grpo_reward_lookup(
        [
            {
                "prompt": "Bagaimana jawab pertanyaan harga?",
                "responses": [
                    "Arahkan ke tim sales untuk penawaran yang sesuai kebutuhan.",
                ],
                "rewards": [1.0],
            }
        ]
    )

    match = lookup.match(
        "Bagaimana jawab pertanyaan harga?",
        "Cuaca hari ini cukup cerah dan tidak ada hubungannya dengan penjualan.",
        completion_ids=[11, 12, 13],
        eos_token_ids=[2],
        pad_token_ids=[0],
    )

    assert match.base_reward == 0.0
    assert match.reward == -0.15
    assert match.reward_adjustment == -0.15
    assert match.truncation_checked is True
    assert match.truncated is True

    stats = lookup.stats_snapshot()
    assert stats["negative_rewards"] == 1
    assert stats["avg_base_reward"] == 0.0
    assert stats["avg_reward_adjustment"] == -0.15
    assert stats["truncation_checks"] == 1
    assert stats["truncated_queries"] == 1
    assert stats["penalized_queries"] == 1


def test_reward_lookup_keeps_terminated_exact_matches_unchanged():
    lookup = build_grpo_reward_lookup(
        [
            {
                "prompt": "Salestify itu apa?",
                "responses": [
                    "Salestify membantu bisnis menangani chat WhatsApp lebih rapi.",
                ],
                "rewards": [1.0],
            }
        ]
    )

    match = lookup.match(
        "Salestify itu apa?",
        "Assistant: Salestify membantu bisnis menangani chat WhatsApp lebih rapi. <|eot_id|>",
        completion_ids=[21, 2],
        eos_token_ids=[2],
        pad_token_ids=[0],
    )

    assert match.reward == 1.0
    assert match.reward_adjustment == 0.0
    assert match.truncation_checked is True
    assert match.truncated is False


def test_resolve_completion_termination_ids_reads_single_token_fields():
    class _Tokenizer:
        eos_token_id = 2
        pad_token_id = 0

    eos_ids, pad_ids = resolve_completion_termination_ids(_Tokenizer())
    assert eos_ids == (2,)
    assert pad_ids == (0,)
