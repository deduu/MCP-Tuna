from __future__ import annotations

from typing import Any, Iterable, Literal, Sequence

PairStrategy = Literal["adjacent", "best_vs_all"]


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_reward(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dedupe_ranked_candidates(
    responses: Sequence[Any],
    rewards: Sequence[Any],
) -> list[tuple[str, float]]:
    ranked = sorted(
        (
            (_normalize_text(response), _normalize_reward(reward))
            for response, reward in zip(responses, rewards)
        ),
        key=lambda item: item[1] if item[1] is not None else float("-inf"),
        reverse=True,
    )
    deduped: list[tuple[str, float]] = []
    seen: set[str] = set()
    for response, reward in ranked:
        if not response or reward is None or response in seen:
            continue
        seen.add(response)
        deduped.append((response, reward))
    return deduped


def build_dpo_pairs_from_ranked_preferences(
    prompt: Any,
    responses: Sequence[Any],
    rewards: Sequence[Any],
    *,
    strategy: PairStrategy = "adjacent",
    top_k: int | None = None,
    max_pairs: int | None = None,
    min_reward_gap: float = 0.0,
) -> list[dict[str, Any]]:
    """Convert ranked preference candidates into DPO pairs.

    This keeps pair construction generic so DPO can reuse richer ranked signals
    instead of relying only on handcrafted chosen/rejected rows.
    """

    prompt_text = _normalize_text(prompt)
    if not prompt_text:
        return []

    ranked_candidates = _dedupe_ranked_candidates(responses, rewards)
    if top_k is not None and top_k > 0:
        ranked_candidates = ranked_candidates[:top_k]
    if len(ranked_candidates) < 2:
        return []

    index_pairs: Iterable[tuple[int, int]]
    if strategy == "adjacent":
        index_pairs = ((index, index + 1) for index in range(len(ranked_candidates) - 1))
    elif strategy == "best_vs_all":
        index_pairs = ((0, index) for index in range(1, len(ranked_candidates)))
    else:
        raise ValueError(f"Unsupported pair strategy: {strategy}")

    pairs: list[dict[str, Any]] = []
    for chosen_index, rejected_index in index_pairs:
        chosen, chosen_reward = ranked_candidates[chosen_index]
        rejected, rejected_reward = ranked_candidates[rejected_index]
        reward_gap = chosen_reward - rejected_reward
        if reward_gap <= 0 or reward_gap < min_reward_gap:
            continue
        pairs.append(
            {
                "prompt": prompt_text,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_reward": round(chosen_reward, 4),
                "rejected_reward": round(rejected_reward, 4),
                "reward_gap": round(reward_gap, 4),
                "chosen_rank": chosen_index,
                "rejected_rank": rejected_index,
            }
        )
        if max_pairs is not None and max_pairs > 0 and len(pairs) >= max_pairs:
            break
    return pairs
