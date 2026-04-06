from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence

from shared.dataset_service import DatasetService
from shared.preference_training_guidance import build_preference_training_guidance

PreferenceTechnique = Literal["dpo", "grpo", "kto"]

_SUPPORTED_TECHNIQUES = {"dpo", "grpo", "kto"}
_SHORT_TEXT_THRESHOLD = 32
_LOW_OVERLAP_THRESHOLD = 0.1
_HARD_NEGATIVE_MIN_OVERLAP = 0.2
_HARD_NEGATIVE_MAX_OVERLAP = 0.75


def _round_ratio(numerator: int | float, denominator: int | float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(part for part in (_normalize_text(item) for item in value) if part)
    if isinstance(value, dict):
        text_value = value.get("text")
        if isinstance(text_value, str):
            return text_value.strip()
    return str(value).strip()


def _preview_text(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _length_stats(values: Sequence[str]) -> Dict[str, float | int]:
    nonempty = [len(value) for value in values if value]
    if not nonempty:
        return {"avg": 0, "min": 0, "max": 0, "p95": 0}

    ordered = sorted(nonempty)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    return {
        "avg": round(sum(ordered) / len(ordered), 1),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[p95_index],
    }


def _duplicate_rows(counter: Counter[str]) -> int:
    return sum(count - 1 for count in counter.values() if count > 1)


def _top_repeats(counter: Counter[str], top_k: int) -> List[Dict[str, Any]]:
    repeated = [
        {"preview": _preview_text(text), "count": count}
        for text, count in counter.most_common()
        if text and count > 1
    ]
    return repeated[:top_k]


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"\w+", left.lower()))
    right_tokens = set(re.findall(r"\w+", right.lower()))
    if not left_tokens and not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_label(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "chosen", "positive"}:
            return True
        if normalized in {"false", "0", "no", "rejected", "negative"}:
            return False
    return None


class PreferenceDatasetAnalyzer:
    def __init__(self, dataset_service: Optional[DatasetService] = None) -> None:
        self._dataset_service = dataset_service or DatasetService()

    async def analyze(
        self,
        dataset_path: str,
        technique: Optional[PreferenceTechnique] = None,
        *,
        max_rows: int = 2000,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        meta = await self._dataset_service.info(dataset_path)
        if not meta.get("success"):
            return meta

        metadata = meta["metadata"]
        detected = metadata.get("technique")
        analyzed = technique or detected
        if analyzed not in _SUPPORTED_TECHNIQUES:
            return {
                "success": False,
                "error": (
                    "Preference dataset analysis only supports dpo, grpo, or kto datasets. "
                    f"Detected technique: {detected or 'unknown'}."
                ),
            }

        loaded = await self._dataset_service.load(dataset_path)
        if not loaded.get("success"):
            return loaded

        rows = loaded.get("data_points", [])
        analyzed_rows = rows[:max_rows] if max_rows > 0 else rows
        warnings: List[str] = []
        recommendations: List[str] = []

        row_count = len(rows)
        prompt_stats = self._build_text_stats(
            [_normalize_text(row.get("prompt")) for row in analyzed_rows],
            len(analyzed_rows),
            top_k,
        )
        if row_count < 100:
            warnings.append(
                "Preference dataset is small; DPO/GRPO usually need a strong SFT checkpoint or more preference rows."
            )
            recommendations.append(
                "Add more prompt coverage before expecting preference tuning to beat a strong SFT baseline."
            )

        result: Dict[str, Any] = {
            "success": True,
            "dataset_path": metadata.get("file_path", dataset_path),
            "technique_detected": detected,
            "technique_analyzed": analyzed,
            "columns": metadata.get("columns", []),
            "row_count": row_count,
            "analyzed_row_count": len(analyzed_rows),
            "truncated": len(analyzed_rows) < row_count,
            "status": "pass",
            "risk_level": "low",
            "prompt_stats": prompt_stats,
            "warnings": warnings,
            "recommendations": recommendations,
        }

        if analyzed == "dpo":
            result["dpo"] = self._analyze_dpo(analyzed_rows, warnings, recommendations, top_k)
        elif analyzed == "grpo":
            result["grpo"] = self._analyze_grpo(analyzed_rows, warnings, recommendations, top_k)
        else:
            result["kto"] = self._analyze_kto(analyzed_rows, warnings, recommendations, top_k)

        if prompt_stats["unique_ratio"] < 0.8 and not self._allow_repeated_prompts(prompt_stats, analyzed, result):
            warnings.append(
                "Prompt diversity is low relative to analyzed rows, so the trainer may overfit a narrow prompt set."
            )
            recommendations.append(
                "Increase prompt variety so preference optimization learns broader behavior, not a few repeated prompts."
            )

        deduped_warnings = list(dict.fromkeys(warnings))
        deduped_recommendations = list(dict.fromkeys(recommendations))
        result["warnings"] = deduped_warnings
        result["recommendations"] = deduped_recommendations
        result["status"] = "warn" if deduped_warnings else "pass"
        result["risk_level"] = self._risk_level(len(deduped_warnings))
        technique_stats = result.get(analyzed, {})
        result["guidance"] = build_preference_training_guidance(
            technique=analyzed,
            row_count=row_count,
            warnings=deduped_warnings,
            prompt_stats=prompt_stats,
            technique_stats=technique_stats if isinstance(technique_stats, dict) else {},
        )
        return result

    @staticmethod
    def _risk_level(warning_count: int) -> str:
        if warning_count >= 3:
            return "high"
        if warning_count >= 1:
            return "medium"
        return "low"

    @staticmethod
    def _build_text_stats(
        values: Sequence[str],
        total_rows: int,
        top_k: int,
    ) -> Dict[str, Any]:
        nonempty = [value for value in values if value]
        counter = Counter(nonempty)
        return {
            "nonempty_count": len(nonempty),
            "empty_count": max(total_rows - len(nonempty), 0),
            "unique_count": len(counter),
            "unique_ratio": _round_ratio(len(counter), len(nonempty) or total_rows),
            "duplicate_rows": _duplicate_rows(counter),
            "length": _length_stats(nonempty),
            "top_repeated": _top_repeats(counter, top_k),
        }

    @staticmethod
    def _allow_repeated_prompts(
        prompt_stats: Dict[str, Any],
        analyzed: PreferenceTechnique,
        result: Dict[str, Any],
    ) -> bool:
        if analyzed != "dpo":
            return False
        dpo_stats = result.get("dpo", {})
        return (
            prompt_stats.get("unique_count", 0) >= 20
            and dpo_stats.get("avg_rows_per_prompt", 0) >= 1.5
            and dpo_stats.get("multi_variant_prompt_ratio", 0) >= 0.5
        )

    def _analyze_dpo(
        self,
        rows: Sequence[Dict[str, Any]],
        warnings: List[str],
        recommendations: List[str],
        top_k: int,
    ) -> Dict[str, Any]:
        chosen_values = [_normalize_text(row.get("chosen")) for row in rows]
        rejected_values = [_normalize_text(row.get("rejected")) for row in rows]
        prompt_values = [_normalize_text(row.get("prompt")) for row in rows]
        pair_values = [
            f"{_normalize_text(row.get('prompt'))}\n<<<CHOSEN>>>\n{chosen}\n<<<REJECTED>>>\n{rejected}"
            for row, chosen, rejected in zip(rows, chosen_values, rejected_values)
        ]
        chosen_stats = self._build_text_stats(chosen_values, len(rows), top_k)
        rejected_stats = self._build_text_stats(rejected_values, len(rows), top_k)
        pair_counter = Counter(pair for pair in pair_values if pair)
        prompt_rejected_map: Dict[str, set[str]] = {}
        prompt_counter = Counter(prompt for prompt in prompt_values if prompt)
        for prompt, rejected in zip(prompt_values, rejected_values):
            if not prompt or not rejected:
                continue
            prompt_rejected_map.setdefault(prompt, set()).add(rejected)
        duplicate_pair_rows = _duplicate_rows(pair_counter)

        identical_rows = 0
        high_overlap_rows = 0
        hard_negative_rows = 0
        low_overlap_rows = 0
        overlaps: List[float] = []
        short_chosen_rows = 0
        short_rejected_rows = 0
        for chosen, rejected in zip(chosen_values, rejected_values):
            if chosen and rejected and chosen == rejected:
                identical_rows += 1
            if chosen and len(chosen) < _SHORT_TEXT_THRESHOLD:
                short_chosen_rows += 1
            if rejected and len(rejected) < _SHORT_TEXT_THRESHOLD:
                short_rejected_rows += 1
            overlap = _token_overlap(chosen, rejected)
            overlaps.append(overlap)
            if overlap >= 0.8:
                high_overlap_rows += 1
            if (
                chosen
                and rejected
                and chosen != rejected
                and len(rejected) >= _SHORT_TEXT_THRESHOLD
                and _HARD_NEGATIVE_MIN_OVERLAP <= overlap < _HARD_NEGATIVE_MAX_OVERLAP
            ):
                hard_negative_rows += 1
            elif chosen and rejected and overlap < _LOW_OVERLAP_THRESHOLD:
                low_overlap_rows += 1

        avg_overlap = round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0
        avg_rows_per_prompt = round(sum(prompt_counter.values()) / len(prompt_counter), 2) if prompt_counter else 0.0
        avg_rejected_variants_per_prompt = (
            round(sum(len(variants) for variants in prompt_rejected_map.values()) / len(prompt_rejected_map), 2)
            if prompt_rejected_map
            else 0.0
        )
        multi_variant_prompt_count = sum(1 for variants in prompt_rejected_map.values() if len(variants) > 1)
        nonempty_rejected_counter = Counter(rejected for rejected in rejected_values if rejected)
        dominant_rejected, dominant_rejected_count = ("", 0)
        if nonempty_rejected_counter:
            dominant_rejected, dominant_rejected_count = nonempty_rejected_counter.most_common(1)[0]

        if rejected_stats["unique_ratio"] < 0.7:
            warnings.append(
                "Rejected responses are highly repetitive, so DPO may just memorize one stock negative instead of broader failure modes."
            )
            recommendations.append(
                "Diversify rejected answers so each prompt shows a distinct mistake pattern, not the same generic bad response."
            )
        if _round_ratio(dominant_rejected_count, rejected_stats["nonempty_count"]) >= 0.25:
            warnings.append(
                "One rejected answer template dominates a large share of the dataset, so DPO may overfit that single failure mode."
            )
            recommendations.append(
                "Spread rejected supervision across multiple failure styles instead of repeating one stock rejection too often."
            )
        if duplicate_pair_rows > 0:
            warnings.append(
                "Some prompt/chosen/rejected triples are duplicated, which reduces the effective size of the preference dataset."
            )
            recommendations.append(
                "Deduplicate identical preference pairs before training to preserve useful signal per step."
            )
        if identical_rows > 0:
            warnings.append(
                "Some chosen and rejected answers are identical, which gives the trainer contradictory supervision."
            )
        if _round_ratio(high_overlap_rows, len(rows)) >= 0.25:
            warnings.append(
                "Many chosen and rejected answers are lexically very similar, so the preference gap is weak."
            )
            recommendations.append(
                "Make chosen answers clearly better than rejected ones on accuracy, specificity, or policy adherence."
            )
        if _round_ratio(hard_negative_rows, len(rows)) < 0.25:
            warnings.append(
                "Too few DPO pairs look like hard negatives, so the model mostly sees easy chosen-vs-rejected separations."
            )
            recommendations.append(
                "Add rejected answers that stay on-topic but fail on nuance, specificity, or policy constraints instead of only obvious bad replies."
            )
        if _round_ratio(low_overlap_rows, len(rows)) >= 0.35:
            warnings.append(
                "Many rejected answers barely overlap with the chosen answer, so DPO may learn off-topic filtering more than ranking subtle quality gaps."
            )
            recommendations.append(
                "Keep more rejected answers in-distribution for the same prompt so DPO learns finer-grained preference differences."
            )
        if _round_ratio(short_rejected_rows, len(rows)) >= 0.25:
            warnings.append(
                "A large share of rejected answers are very short, which can make the model learn to avoid brevity rather than real mistakes."
            )

        return {
            "chosen_stats": chosen_stats,
            "rejected_stats": rejected_stats,
            "duplicate_pair_rows": duplicate_pair_rows,
            "duplicate_pair_ratio": _round_ratio(duplicate_pair_rows, len(rows)),
            "identical_pair_rows": identical_rows,
            "identical_pair_ratio": _round_ratio(identical_rows, len(rows)),
            "avg_token_overlap": avg_overlap,
            "high_overlap_rows": high_overlap_rows,
            "high_overlap_ratio": _round_ratio(high_overlap_rows, len(rows)),
            "hard_negative_rows": hard_negative_rows,
            "hard_negative_ratio": _round_ratio(hard_negative_rows, len(rows)),
            "low_overlap_rows": low_overlap_rows,
            "low_overlap_ratio": _round_ratio(low_overlap_rows, len(rows)),
            "dominant_rejected_count": dominant_rejected_count,
            "dominant_rejected_ratio": _round_ratio(dominant_rejected_count, rejected_stats["nonempty_count"]),
            "dominant_rejected_preview": _preview_text(dominant_rejected) if dominant_rejected else "",
            "avg_rows_per_prompt": avg_rows_per_prompt,
            "avg_rejected_variants_per_prompt": avg_rejected_variants_per_prompt,
            "multi_variant_prompt_count": multi_variant_prompt_count,
            "multi_variant_prompt_ratio": _round_ratio(multi_variant_prompt_count, len(prompt_counter)),
            "short_chosen_rows": short_chosen_rows,
            "short_rejected_rows": short_rejected_rows,
        }

    def _analyze_grpo(
        self,
        rows: Sequence[Dict[str, Any]],
        warnings: List[str],
        recommendations: List[str],
        top_k: int,
    ) -> Dict[str, Any]:
        global_responses: List[str] = []
        response_counts: List[int] = []
        unique_response_counts: List[int] = []
        reward_values: List[float] = []
        invalid_rows = 0
        mismatched_rows = 0
        identical_response_rows = 0
        zero_variance_rows = 0
        single_response_rows = 0

        for row in rows:
            responses_raw = row.get("responses")
            rewards_raw = row.get("rewards")
            if not isinstance(responses_raw, list) or not isinstance(rewards_raw, list):
                invalid_rows += 1
                continue

            normalized_responses = [_normalize_text(response) for response in responses_raw]
            normalized_responses = [response for response in normalized_responses if response]
            numeric_rewards = [reward for reward in (_safe_float(value) for value in rewards_raw) if reward is not None]

            response_counts.append(len(normalized_responses))
            unique_response_counts.append(len(set(normalized_responses)))
            global_responses.extend(normalized_responses)
            reward_values.extend(numeric_rewards)

            if len(normalized_responses) <= 1:
                single_response_rows += 1
            if normalized_responses and len(set(normalized_responses)) == 1 and len(normalized_responses) > 1:
                identical_response_rows += 1
            if len(normalized_responses) != len(numeric_rewards):
                mismatched_rows += 1
            if len(numeric_rewards) > 1 and statistics.pvariance(numeric_rewards) == 0:
                zero_variance_rows += 1

        response_counter = Counter(global_responses)
        response_stats = self._build_text_stats(global_responses, len(global_responses), top_k)
        avg_responses = round(sum(response_counts) / len(response_counts), 2) if response_counts else 0.0
        avg_unique_responses = (
            round(sum(unique_response_counts) / len(unique_response_counts), 2)
            if unique_response_counts
            else 0.0
        )

        if invalid_rows > 0 or mismatched_rows > 0:
            warnings.append(
                "Some GRPO rows have malformed responses/rewards lists or mismatched lengths, so reward supervision is incomplete."
            )
            recommendations.append(
                "Ensure every GRPO row has the same number of responses and rewards, with numeric reward values for each response."
            )
        if _round_ratio(single_response_rows, len(rows)) >= 0.2:
            warnings.append(
                "Many GRPO rows contain fewer than two candidate responses, which weakens the ranking signal."
            )
        if _round_ratio(zero_variance_rows, len(rows)) >= 0.2:
            warnings.append(
                "A large share of GRPO rows have no reward variance, so the model sees weak preference ordering."
            )
            recommendations.append(
                "Increase reward separation within each prompt so better and worse candidates are clearly ranked."
            )
        if _round_ratio(identical_response_rows, len(rows)) >= 0.2:
            warnings.append(
                "Many GRPO rows reuse identical responses within the same prompt, which reduces response diversity."
            )
            recommendations.append(
                "Generate more diverse candidate responses per prompt before assigning rewards."
            )
        if response_stats["unique_ratio"] < 0.7:
            warnings.append(
                "Response diversity is low across analyzed GRPO candidates, so the trainer may mostly relearn the same few outputs."
            )

        reward_summary: Dict[str, float | int] = {
            "count": len(reward_values),
            "min": round(min(reward_values), 4) if reward_values else 0,
            "max": round(max(reward_values), 4) if reward_values else 0,
            "mean": round(sum(reward_values) / len(reward_values), 4) if reward_values else 0,
            "stdev": round(statistics.pstdev(reward_values), 4) if len(reward_values) > 1 else 0,
        }

        return {
            "response_stats": response_stats,
            "avg_responses_per_row": avg_responses,
            "avg_unique_responses_per_row": avg_unique_responses,
            "single_response_rows": single_response_rows,
            "invalid_rows": invalid_rows,
            "mismatched_rows": mismatched_rows,
            "identical_response_rows": identical_response_rows,
            "identical_response_ratio": _round_ratio(identical_response_rows, len(rows)),
            "zero_reward_variance_rows": zero_variance_rows,
            "zero_reward_variance_ratio": _round_ratio(zero_variance_rows, len(rows)),
            "reward_stats": reward_summary,
            "top_repeated_responses": _top_repeats(response_counter, top_k),
        }

    def _analyze_kto(
        self,
        rows: Sequence[Dict[str, Any]],
        warnings: List[str],
        recommendations: List[str],
        top_k: int,
    ) -> Dict[str, Any]:
        completion_values = [_normalize_text(row.get("completion")) for row in rows]
        completion_stats = self._build_text_stats(completion_values, len(rows), top_k)
        labels = [_safe_label(row.get("label")) for row in rows]
        valid_labels = [label for label in labels if label is not None]
        positive_count = sum(1 for label in valid_labels if label)
        negative_count = sum(1 for label in valid_labels if not label)
        invalid_label_rows = len(rows) - len(valid_labels)

        if positive_count == 0 or negative_count == 0:
            warnings.append(
                "KTO labels only cover one class, so the trainer cannot learn both preferred and rejected behavior."
            )
            recommendations.append(
                "Include both positive and negative KTO labels for the same task distribution."
            )
        elif min(positive_count, negative_count) / max(positive_count, negative_count) < 0.25:
            warnings.append(
                "KTO label balance is heavily skewed, so the trainer may underlearn the minority class."
            )
        if completion_stats["unique_ratio"] < 0.7:
            warnings.append(
                "Completion diversity is low, so KTO supervision may collapse onto a few repeated answers."
            )

        return {
            "completion_stats": completion_stats,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_ratio": _round_ratio(positive_count, len(valid_labels)),
            "negative_ratio": _round_ratio(negative_count, len(valid_labels)),
            "invalid_label_rows": invalid_label_rows,
        }
