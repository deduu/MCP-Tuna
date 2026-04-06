from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Sequence

from shared.training_defaults import DEFAULT_GRPO_TRUNCATION_PENALTY

_SPECIAL_TOKEN_RE = re.compile(
    r"<\|[^>]+?\|>|</?s>|\[(?:/)?INST\]|\b(?:bos|eos)_token\b",
    re.IGNORECASE,
)
_ROLE_PREFIX_RE = re.compile(
    r"^(?:assistant|assistant response|response|answer|jawaban)\s*[:：\-]\s*",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_MIN_SIMILARITY = 0.55
_MIN_CONTAINMENT_CHARS = 32


def _round(value: float) -> float:
    return round(float(value), 4)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_coerce_text(item) for item in value) if part)
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
        text = value.get("text")
        if isinstance(text, str):
            return text
    return str(value)


def normalize_preference_text(value: Any) -> str:
    text = _coerce_text(value).replace("\r", "\n")
    text = _SPECIAL_TOKEN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _ROLE_PREFIX_RE.sub("", text).strip()


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(text.lower()))


def _strip_prompt_echo(completion: str, prompt: str) -> str:
    if not completion or not prompt:
        return completion
    if not completion.startswith(prompt):
        return completion
    stripped = completion[len(prompt):].lstrip(" \n\t:.-")
    return stripped or completion


def _completion_variants(prompt: Any, completion: Any) -> List[str]:
    prompt_text = normalize_preference_text(prompt)
    completion_text = normalize_preference_text(completion)
    variants: List[str] = []
    for candidate in (
        completion_text,
        _strip_prompt_echo(completion_text, prompt_text),
    ):
        if not candidate:
            continue
        trimmed = _ROLE_PREFIX_RE.sub("", candidate).strip()
        for variant in (candidate, trimmed):
            if variant and variant not in variants:
                variants.append(variant)
    return variants


def _similarity(left: str, right: str, left_tokens: frozenset[str]) -> float:
    right_tokens = _tokenize(right)
    if left == right:
        return 1.0
    overlap = 0.0
    if left_tokens or right_tokens:
        union = left_tokens | right_tokens
        if union:
            overlap = len(left_tokens & right_tokens) / len(union)
    containment = 0.0
    shortest = min(len(left), len(right))
    if shortest >= _MIN_CONTAINMENT_CHARS and (left in right or right in left):
        containment = shortest / max(len(left), len(right))
    sequence = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return max(containment, round((sequence * 0.65) + (overlap * 0.35), 4))


@dataclass(frozen=True)
class RewardMatch:
    reward: float
    similarity: float = 0.0
    matched_response: str | None = None
    matched_via: str = "none"
    base_reward: float | None = None
    reward_adjustment: float = 0.0
    truncation_checked: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class _RewardCandidate:
    text: str
    reward: float
    tokens: frozenset[str]


class PreferenceRewardLookup:
    """Resolve GRPO rewards against normalized candidate responses.

    Generated completions rarely match stored candidates byte-for-byte because of
    whitespace, special tokens, or assistant prefixes. This matcher keeps GRPO
    from silently becoming a no-op on those formatting differences.
    """

    def __init__(
        self,
        rows: Sequence[Dict[str, Any]],
        *,
        similarity_threshold: float = _MIN_SIMILARITY,
        truncation_penalty: float = DEFAULT_GRPO_TRUNCATION_PENALTY,
    ) -> None:
        self._similarity_threshold = similarity_threshold
        self._truncation_penalty = max(0.0, float(truncation_penalty))
        self._entries: Dict[str, Dict[str, Any]] = {}
        self.reset_stats()
        for row in rows:
            prompt_key = normalize_preference_text(row.get("prompt"))
            if not prompt_key:
                continue
            entry = self._entries.setdefault(
                prompt_key,
                {"exact": {}, "candidates": []},
            )
            responses = row.get("responses") or []
            rewards = row.get("rewards") or []
            for response, reward in zip(responses, rewards):
                response_text = normalize_preference_text(response)
                if not response_text or response_text in entry["exact"]:
                    continue
                reward_value = float(reward)
                entry["exact"][response_text] = reward_value
                entry["candidates"].append(
                    _RewardCandidate(
                        text=response_text,
                        reward=reward_value,
                        tokens=_tokenize(response_text),
                    )
                )

    def score(self, prompt: Any, completion: Any) -> float:
        return self.match(prompt, completion).reward

    def score_batch(
        self,
        prompts: Iterable[Any],
        completions: Iterable[Any],
        *,
        completion_ids: Iterable[Any] | None = None,
        eos_token_ids: Any = None,
        pad_token_ids: Any = None,
    ) -> List[float]:
        prompt_list = list(prompts)
        completion_list = list(completions)
        completion_id_list = list(completion_ids) if completion_ids is not None else []
        if len(completion_id_list) < len(prompt_list):
            completion_id_list.extend([None] * (len(prompt_list) - len(completion_id_list)))
        return [
            self.match(
                prompt,
                completion,
                completion_ids=ids,
                eos_token_ids=eos_token_ids,
                pad_token_ids=pad_token_ids,
            ).reward
            for prompt, completion, ids in zip(
                prompt_list,
                completion_list,
                completion_id_list or [None] * len(prompt_list),
            )
        ]

    def reset_stats(self) -> None:
        self._stats = {
            "queries": 0,
            "exact_matches": 0,
            "similarity_matches": 0,
            "misses": 0,
            "positive_rewards": 0,
            "zero_rewards": 0,
            "negative_rewards": 0,
            "reward_sum": 0.0,
            "base_reward_sum": 0.0,
            "reward_adjustment_sum": 0.0,
            "similarity_sum": 0.0,
            "matched_similarity_sum": 0.0,
            "matched_similarity_count": 0,
            "max_similarity": 0.0,
            "truncation_checks": 0,
            "truncated_queries": 0,
            "penalized_queries": 0,
        }

    def stats_snapshot(self) -> Dict[str, Any]:
        queries = self._stats["queries"]
        matched = self._stats["exact_matches"] + self._stats["similarity_matches"]
        matched_similarity_count = self._stats["matched_similarity_count"]
        truncation_checks = self._stats["truncation_checks"]
        return {
            "queries": queries,
            "exact_matches": self._stats["exact_matches"],
            "similarity_matches": self._stats["similarity_matches"],
            "misses": self._stats["misses"],
            "positive_rewards": self._stats["positive_rewards"],
            "zero_rewards": self._stats["zero_rewards"],
            "negative_rewards": self._stats["negative_rewards"],
            "match_rate": _round(matched / queries) if queries else 0.0,
            "exact_match_rate": _round(self._stats["exact_matches"] / queries) if queries else 0.0,
            "similarity_match_rate": _round(self._stats["similarity_matches"] / queries) if queries else 0.0,
            "positive_reward_rate": _round(self._stats["positive_rewards"] / queries) if queries else 0.0,
            "avg_reward": _round(self._stats["reward_sum"] / queries) if queries else 0.0,
            "avg_base_reward": _round(self._stats["base_reward_sum"] / queries) if queries else 0.0,
            "avg_reward_adjustment": (
                _round(self._stats["reward_adjustment_sum"] / queries) if queries else 0.0
            ),
            "avg_similarity": _round(self._stats["similarity_sum"] / queries) if queries else 0.0,
            "avg_matched_similarity": (
                _round(self._stats["matched_similarity_sum"] / matched_similarity_count)
                if matched_similarity_count
                else 0.0
            ),
            "max_similarity": _round(self._stats["max_similarity"]),
            "similarity_threshold": self._similarity_threshold,
            "truncation_checks": truncation_checks,
            "truncated_queries": self._stats["truncated_queries"],
            "truncated_query_rate": (
                _round(self._stats["truncated_queries"] / truncation_checks)
                if truncation_checks
                else 0.0
            ),
            "penalized_queries": self._stats["penalized_queries"],
            "truncation_penalty": self._truncation_penalty,
        }

    def _record_match(self, match: RewardMatch) -> None:
        base_reward = match.base_reward if match.base_reward is not None else match.reward
        self._stats["queries"] += 1
        self._stats["reward_sum"] += match.reward
        self._stats["base_reward_sum"] += base_reward
        self._stats["reward_adjustment_sum"] += match.reward_adjustment
        self._stats["similarity_sum"] += match.similarity
        self._stats["max_similarity"] = max(self._stats["max_similarity"], match.similarity)
        if match.matched_via == "exact":
            self._stats["exact_matches"] += 1
        elif match.matched_via == "similarity":
            self._stats["similarity_matches"] += 1
        else:
            self._stats["misses"] += 1

        if match.matched_via != "none":
            self._stats["matched_similarity_count"] += 1
            self._stats["matched_similarity_sum"] += match.similarity

        if match.truncation_checked:
            self._stats["truncation_checks"] += 1
            if match.truncated:
                self._stats["truncated_queries"] += 1
            if match.reward_adjustment < 0:
                self._stats["penalized_queries"] += 1

        if match.reward > 0:
            self._stats["positive_rewards"] += 1
        elif match.reward < 0:
            self._stats["negative_rewards"] += 1
        else:
            self._stats["zero_rewards"] += 1

    def match(
        self,
        prompt: Any,
        completion: Any,
        *,
        completion_ids: Any = None,
        eos_token_ids: Any = None,
        pad_token_ids: Any = None,
    ) -> RewardMatch:
        match = self._resolve_match(prompt, completion)
        match = self._apply_truncation_penalty(
            match,
            completion_ids=completion_ids,
            eos_token_ids=eos_token_ids,
            pad_token_ids=pad_token_ids,
        )
        self._record_match(match)
        return match

    def _resolve_match(self, prompt: Any, completion: Any) -> RewardMatch:
        prompt_key = normalize_preference_text(prompt)
        entry = self._entries.get(prompt_key)
        if not entry:
            return RewardMatch(reward=0.0, base_reward=0.0)

        candidates: Sequence[_RewardCandidate] = entry["candidates"]
        exact_rewards: Dict[str, float] = entry["exact"]
        best_match = RewardMatch(reward=0.0)
        for completion_text in _completion_variants(prompt, completion):
            exact_reward = exact_rewards.get(completion_text)
            if exact_reward is not None:
                return RewardMatch(
                    reward=exact_reward,
                    similarity=1.0,
                    matched_response=completion_text,
                    matched_via="exact",
                    base_reward=exact_reward,
                )
            completion_tokens = _tokenize(completion_text)
            for candidate in candidates:
                similarity = _similarity(
                    completion_text,
                    candidate.text,
                    completion_tokens,
                )
                if similarity < self._similarity_threshold:
                    continue
                if similarity > best_match.similarity:
                    best_match = RewardMatch(
                        reward=candidate.reward,
                        similarity=similarity,
                        matched_response=candidate.text,
                        matched_via="similarity",
                        base_reward=candidate.reward,
                    )
        return best_match

    def _apply_truncation_penalty(
        self,
        match: RewardMatch,
        *,
        completion_ids: Any,
        eos_token_ids: Any,
        pad_token_ids: Any,
    ) -> RewardMatch:
        truncated, checked = _detect_truncated_completion(
            completion_ids,
            eos_token_ids=eos_token_ids,
            pad_token_ids=pad_token_ids,
        )
        base_reward = match.base_reward if match.base_reward is not None else match.reward
        if not checked or not truncated or self._truncation_penalty <= 0:
            return RewardMatch(
                reward=match.reward,
                similarity=match.similarity,
                matched_response=match.matched_response,
                matched_via=match.matched_via,
                base_reward=base_reward,
                reward_adjustment=0.0,
                truncation_checked=checked,
                truncated=truncated,
            )
        reward_adjustment = -self._truncation_penalty * max(0.0, 1.0 - match.similarity)
        return RewardMatch(
            reward=base_reward + reward_adjustment,
            similarity=match.similarity,
            matched_response=match.matched_response,
            matched_via=match.matched_via,
            base_reward=base_reward,
            reward_adjustment=reward_adjustment,
            truncation_checked=True,
            truncated=True,
        )


def _normalize_token_ids(value: Any) -> tuple[int, ...]:
    if value is None or isinstance(value, bool):
        return ()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, int):
        return (int(value),)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    token_ids: list[int] = []
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            token_ids.append(int(item))
        except (TypeError, ValueError):
            continue
    return tuple(token_ids)


def _merge_token_ids(*values: Any) -> tuple[int, ...]:
    merged: dict[int, None] = {}
    for value in values:
        for token_id in _normalize_token_ids(value):
            merged[token_id] = None
    return tuple(merged)


def resolve_completion_termination_ids(tokenizer: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        _merge_token_ids(
            getattr(tokenizer, "eos_token_ids", None),
            getattr(tokenizer, "eos_token_id", None),
        ),
        _merge_token_ids(
            getattr(tokenizer, "pad_token_ids", None),
            getattr(tokenizer, "pad_token_id", None),
        ),
    )


def _detect_truncated_completion(
    completion_ids: Any,
    *,
    eos_token_ids: Any,
    pad_token_ids: Any,
) -> tuple[bool, bool]:
    ids = _normalize_token_ids(completion_ids)
    termination_ids = set(_merge_token_ids(eos_token_ids, pad_token_ids))
    if not ids or not termination_ids:
        return False, False
    return ids[-1] not in termination_ids, True


def build_grpo_reward_lookup(
    rows: Sequence[Dict[str, Any]],
    *,
    similarity_threshold: float = _MIN_SIMILARITY,
    truncation_penalty: float = DEFAULT_GRPO_TRUNCATION_PENALTY,
) -> PreferenceRewardLookup:
    return PreferenceRewardLookup(
        rows,
        similarity_threshold=similarity_threshold,
        truncation_penalty=truncation_penalty,
    )
