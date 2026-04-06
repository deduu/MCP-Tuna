"""Helpers for deterministic fine-tuning benchmarks."""

from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Optional, Sequence


DEFAULT_BENCHMARK_SEEDS = [42, 43, 44]
DEFAULT_PRIMARY_METRIC = "avg_composite_score"
PRIMARY_PACK_PREFERENCE = ("hidden_holdout", "dev", "safety")


def normalize_seed_list(seeds: Optional[Sequence[int]]) -> List[int]:
    if seeds is None:
        return list(DEFAULT_BENCHMARK_SEEDS)

    normalized: List[int] = []
    seen = set()
    for seed in seeds:
        value = int(seed)
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized or list(DEFAULT_BENCHMARK_SEEDS)


def infer_primary_pack(
    available_packs: Iterable[str],
    requested_pack: Optional[str] = None,
) -> Optional[str]:
    pack_names = [str(name) for name in available_packs if str(name).strip()]
    if requested_pack:
        return requested_pack if requested_pack in pack_names else None
    for candidate in PRIMARY_PACK_PREFERENCE:
        if candidate in pack_names:
            return candidate
    return pack_names[0] if pack_names else None


def normalize_benchmark_cases(
    rows: Any,
    *,
    pack_name: str,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for index, raw_row in enumerate(_iter_rows(rows), start=1):
        prompt = _extract_prompt(raw_row)
        if not prompt:
            continue
        metadata = raw_row.get("metadata")
        cases.append(
            {
                "case_id": str(
                    raw_row.get("case_id")
                    or raw_row.get("id")
                    or f"{pack_name}_{index}"
                ),
                "pack_name": pack_name,
                "prompt": prompt,
                "reference": _extract_reference(raw_row),
                "system_prompt": str(raw_row.get("system") or "").strip() or None,
                "required_terms": _parse_string_list(
                    raw_row.get("required_terms")
                    or raw_row.get("must_mention_terms")
                    or raw_row.get("expected_terms")
                ),
                "required_any_terms": _parse_string_list(
                    raw_row.get("required_any_terms")
                    or raw_row.get("must_mention_any_terms")
                ),
                "required_patterns": _parse_string_list(
                    raw_row.get("required_patterns")
                    or raw_row.get("must_match_patterns")
                ),
                "required_any_patterns": _parse_string_list(
                    raw_row.get("required_any_patterns")
                    or raw_row.get("must_match_any_patterns")
                ),
                "forbidden_terms": _parse_string_list(
                    raw_row.get("forbidden_terms")
                    or raw_row.get("must_not_mention_terms")
                    or raw_row.get("banned_terms")
                ),
                "forbidden_patterns": _parse_string_list(
                    raw_row.get("forbidden_patterns")
                    or raw_row.get("must_not_match_patterns")
                    or raw_row.get("banned_patterns")
                ),
                "min_token_f1": _coerce_optional_float(raw_row.get("min_token_f1")),
                "label": raw_row.get("label"),
                "metadata": dict(metadata) if isinstance(metadata, dict) else {},
            }
        )
    return cases


def score_benchmark_case(case: Dict[str, Any], generated: str) -> Dict[str, Any]:
    reference = str(case.get("reference") or "")
    token_f1 = _token_f1(reference, generated) if reference else None
    exact_match = _normalized_text(reference) == _normalized_text(generated) if reference else None

    required_terms = list(case.get("required_terms") or [])
    required_any_terms = list(case.get("required_any_terms") or [])
    required_patterns = list(case.get("required_patterns") or [])
    required_any_patterns = list(case.get("required_any_patterns") or [])
    forbidden_terms = list(case.get("forbidden_terms") or [])
    forbidden_patterns = list(case.get("forbidden_patterns") or [])

    required_hits = [term for term in required_terms if _contains_term(generated, term)]
    required_any_hits = [term for term in required_any_terms if _contains_term(generated, term)]
    required_pattern_hits = [
        pattern for pattern in required_patterns if _contains_pattern(generated, pattern)
    ]
    required_any_pattern_hits = [
        pattern for pattern in required_any_patterns if _contains_pattern(generated, pattern)
    ]
    forbidden_hits = [term for term in forbidden_terms if _contains_term(generated, term)]
    forbidden_pattern_hits = [
        pattern for pattern in forbidden_patterns if _contains_pattern(generated, pattern)
    ]

    required_coverage = (
        len(required_hits) / len(required_terms) if required_terms else None
    )
    required_terms_pass = (
        len(required_hits) == len(required_terms) if required_terms else None
    )
    required_any_terms_pass = (
        bool(required_any_hits) if required_any_terms else None
    )
    required_patterns_coverage = (
        len(required_pattern_hits) / len(required_patterns)
        if required_patterns
        else None
    )
    required_patterns_pass = (
        len(required_pattern_hits) == len(required_patterns)
        if required_patterns
        else None
    )
    required_any_patterns_pass = (
        bool(required_any_pattern_hits) if required_any_patterns else None
    )
    forbidden_terms_pass = (
        not forbidden_hits if forbidden_terms else None
    )
    forbidden_patterns_pass = (
        not forbidden_pattern_hits if forbidden_patterns else None
    )

    min_token_f1 = _coerce_optional_float(case.get("min_token_f1"))
    min_token_f1_pass = (
        token_f1 is not None and token_f1 >= min_token_f1
        if min_token_f1 is not None
        else None
    )

    composite_components = [
        value
        for value in [
            token_f1,
            required_coverage,
            1.0 if required_any_terms_pass is True else 0.0 if required_any_terms else None,
            required_patterns_coverage,
            1.0 if required_any_patterns_pass is True else 0.0 if required_any_patterns else None,
            1.0 if forbidden_terms_pass is True else 0.0 if forbidden_terms else None,
            1.0 if forbidden_patterns_pass is True else 0.0 if forbidden_patterns else None,
        ]
        if value is not None
    ]
    composite_score = mean(composite_components) if composite_components else 0.0

    pass_checks = [
        value
        for value in [
            required_terms_pass,
            required_any_terms_pass,
            required_patterns_pass,
            required_any_patterns_pass,
            forbidden_terms_pass,
            forbidden_patterns_pass,
            min_token_f1_pass,
        ]
        if value is not None
    ]

    return {
        "case_id": case.get("case_id"),
        "prompt": case.get("prompt"),
        "reference": reference,
        "generated": generated,
        "label": case.get("label"),
        "token_f1": token_f1,
        "exact_match": exact_match,
        "required_terms_pass": required_terms_pass,
        "required_terms_coverage": required_coverage,
        "required_terms_hits": required_hits,
        "required_any_terms_pass": required_any_terms_pass,
        "required_any_terms_hits": required_any_hits,
        "required_patterns_pass": required_patterns_pass,
        "required_patterns_coverage": required_patterns_coverage,
        "required_patterns_hits": required_pattern_hits,
        "required_any_patterns_pass": required_any_patterns_pass,
        "required_any_patterns_hits": required_any_pattern_hits,
        "forbidden_terms_pass": forbidden_terms_pass,
        "forbidden_terms_hits": forbidden_hits,
        "forbidden_patterns_pass": forbidden_patterns_pass,
        "forbidden_patterns_hits": forbidden_pattern_hits,
        "min_token_f1": min_token_f1,
        "min_token_f1_pass": min_token_f1_pass,
        "composite_score": round(composite_score, 4),
        "case_pass": all(pass_checks) if pass_checks else None,
    }


def summarize_pack_scores(case_scores: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "num_cases": len(case_scores),
        "avg_composite_score": _rounded_average(
            score.get("composite_score") for score in case_scores
        ),
        "avg_token_f1": _rounded_average(score.get("token_f1") for score in case_scores),
        "exact_match_rate": _rounded_average(
            _bool_to_float(score.get("exact_match")) for score in case_scores
        ),
        "required_terms_pass_rate": _rounded_average(
            _bool_to_float(score.get("required_terms_pass")) for score in case_scores
        ),
        "required_terms_coverage": _rounded_average(
            score.get("required_terms_coverage") for score in case_scores
        ),
        "required_any_terms_pass_rate": _rounded_average(
            _bool_to_float(score.get("required_any_terms_pass")) for score in case_scores
        ),
        "required_patterns_pass_rate": _rounded_average(
            _bool_to_float(score.get("required_patterns_pass")) for score in case_scores
        ),
        "required_patterns_coverage": _rounded_average(
            score.get("required_patterns_coverage") for score in case_scores
        ),
        "required_any_patterns_pass_rate": _rounded_average(
            _bool_to_float(score.get("required_any_patterns_pass")) for score in case_scores
        ),
        "forbidden_terms_pass_rate": _rounded_average(
            _bool_to_float(score.get("forbidden_terms_pass")) for score in case_scores
        ),
        "forbidden_patterns_pass_rate": _rounded_average(
            _bool_to_float(score.get("forbidden_patterns_pass")) for score in case_scores
        ),
        "min_token_f1_pass_rate": _rounded_average(
            _bool_to_float(score.get("min_token_f1_pass")) for score in case_scores
        ),
        "pass_rate": _rounded_average(
            _bool_to_float(score.get("case_pass")) for score in case_scores
        ),
    }

    component_counts = Counter()
    for score in case_scores:
        for key in (
            "token_f1",
            "exact_match",
            "required_terms_pass",
            "required_any_terms_pass",
            "required_patterns_pass",
            "required_any_patterns_pass",
            "forbidden_terms_pass",
            "forbidden_patterns_pass",
            "min_token_f1_pass",
            "case_pass",
        ):
            if score.get(key) is not None:
                component_counts[key] += 1
    summary["component_case_counts"] = dict(component_counts)
    return summary


def aggregate_benchmark_runs(
    runs: Sequence[Dict[str, Any]],
    *,
    primary_pack: str,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
    benchmark_gates: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    method_runs: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        method = str(run.get("method") or "").strip()
        if not method:
            continue
        method_runs.setdefault(method, []).append(run)

    method_summaries: List[Dict[str, Any]] = []
    for method, method_entries in method_runs.items():
        packs: Dict[str, Dict[str, Any]] = {}
        pack_names = sorted(
            {
                pack_name
                for entry in method_entries
                for pack_name in (entry.get("evaluation") or {}).keys()
            }
        )
        for pack_name in pack_names:
            seed_scores = []
            numeric_metrics: Dict[str, List[float]] = {}
            for entry in method_entries:
                pack_result = (entry.get("evaluation") or {}).get(pack_name) or {}
                summary = pack_result.get("summary") or {}
                run_metric = summary.get(primary_metric)
                if run_metric is not None:
                    seed_scores.append(
                        {
                            "seed": entry.get("seed"),
                            primary_metric: run_metric,
                        }
                    )
                for key, value in summary.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        numeric_metrics.setdefault(key, []).append(float(value))

            pack_summary: Dict[str, Any] = {
                "run_count": len(seed_scores),
                "seed_scores": seed_scores,
            }
            for key, values in numeric_metrics.items():
                pack_summary[f"mean_{key}"] = round(mean(values), 4)
                pack_summary[f"stdev_{key}"] = round(stdev(values), 4) if len(values) > 1 else 0.0
            packs[pack_name] = pack_summary

        primary_pack_summary = packs.get(primary_pack) or {}
        method_summaries.append(
            {
                "method": method,
                "run_count": len(method_entries),
                "packs": packs,
                "primary_pack": primary_pack,
                "primary_metric": primary_metric,
                "primary_score_mean": primary_pack_summary.get(f"mean_{primary_metric}"),
                "primary_score_stdev": primary_pack_summary.get(f"stdev_{primary_metric}"),
            }
        )

    method_rankings = sorted(
        method_summaries,
        key=lambda item: (
            item.get("primary_score_mean") is None,
            -(
                item["primary_score_mean"]
                if item.get("primary_score_mean") is not None
                else -math.inf
            ),
        ),
    )
    best_method = method_rankings[0]["method"] if method_rankings else None
    best_run = select_best_benchmark_run(
        runs,
        primary_pack=primary_pack,
        primary_metric=primary_metric,
    )

    gates = _evaluate_benchmark_gates(
        method_rankings,
        primary_pack=primary_pack,
        primary_metric=primary_metric,
        benchmark_gates=benchmark_gates,
    )

    return {
        "primary_pack": primary_pack,
        "primary_metric": primary_metric,
        "method_rankings": method_rankings,
        "best_method": best_method,
        "best_run": best_run,
        "gates": gates,
    }


def select_best_benchmark_run(
    runs: Sequence[Dict[str, Any]],
    *,
    primary_pack: str,
    primary_metric: str = DEFAULT_PRIMARY_METRIC,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_score = -math.inf

    for run in runs:
        pack_result = ((run.get("evaluation") or {}).get(primary_pack) or {})
        summary = pack_result.get("summary") or {}
        score = summary.get(primary_metric)
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            continue
        numeric_score = float(score)
        if numeric_score <= best_score:
            continue
        best_score = numeric_score
        best = {
            "method": run.get("method"),
            "seed": run.get("seed"),
            "pack": primary_pack,
            "metric": primary_metric,
            "score": round(numeric_score, 4),
            "model_spec": run.get("model_spec"),
            "training": run.get("training"),
        }

    return best


def _evaluate_benchmark_gates(
    method_summaries: Sequence[Dict[str, Any]],
    *,
    primary_pack: str,
    primary_metric: str,
    benchmark_gates: Optional[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    by_method = {item.get("method"): item for item in method_summaries}
    configured_gates = list(benchmark_gates or [])
    if not configured_gates and {"flat_sft", "curriculum_sft"}.issubset(by_method):
        configured_gates.append(
            {
                "name": "curriculum_beats_flat",
                "candidate_method": "curriculum_sft",
                "baseline_method": "flat_sft",
                "pack": primary_pack,
                "metric": primary_metric,
                "min_delta": 0.0,
            }
        )

    results: List[Dict[str, Any]] = []
    for index, gate in enumerate(configured_gates, start=1):
        candidate_method = gate.get("candidate_method")
        baseline_method = gate.get("baseline_method")
        pack_name = gate.get("pack") or primary_pack
        metric_name = gate.get("metric") or primary_metric
        min_delta = float(gate.get("min_delta", 0.0))
        candidate_summary = by_method.get(candidate_method)
        baseline_summary = by_method.get(baseline_method)

        candidate_score = _method_metric(candidate_summary, pack_name, metric_name)
        baseline_score = _method_metric(baseline_summary, pack_name, metric_name)
        delta = (
            round(candidate_score - baseline_score, 4)
            if candidate_score is not None and baseline_score is not None
            else None
        )
        passed = delta is not None and delta >= min_delta

        results.append(
            {
                "name": gate.get("name") or f"gate_{index}",
                "candidate_method": candidate_method,
                "baseline_method": baseline_method,
                "pack": pack_name,
                "metric": metric_name,
                "min_delta": min_delta,
                "candidate_score": candidate_score,
                "baseline_score": baseline_score,
                "delta": delta,
                "passed": passed,
            }
        )
    return results


def _method_metric(
    method_summary: Optional[Dict[str, Any]],
    pack_name: str,
    metric_name: str,
) -> Optional[float]:
    if not method_summary:
        return None
    pack_summary = (method_summary.get("packs") or {}).get(pack_name) or {}
    return pack_summary.get(f"mean_{metric_name}")


def _iter_rows(rows: Any) -> Iterable[Dict[str, Any]]:
    if rows is None:
        return []
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    try:
        return [dict(row) for row in rows]
    except Exception:
        return []


def _extract_prompt(row: Dict[str, Any]) -> str:
    if isinstance(row.get("prompt"), str) and row["prompt"].strip():
        return row["prompt"].strip()
    if isinstance(row.get("user"), str) and row["user"].strip():
        return row["user"].strip()
    instruction = str(row.get("instruction") or "").strip()
    input_text = str(row.get("input") or "").strip()
    prompt = f"{instruction} {input_text}".strip()
    if prompt:
        return prompt
    if isinstance(row.get("question"), str) and row["question"].strip():
        return row["question"].strip()
    return ""


def _extract_reference(row: Dict[str, Any]) -> str:
    for key in ("assistant", "response", "output", "reference", "chosen", "answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        pieces = [part.strip() for part in value.split(",")]
        return [piece for piece in pieces if piece]
    if isinstance(value, (list, tuple, set)):
        normalized = []
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    normalized.append(item)
        return normalized
    return []


def _coerce_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _token_f1(reference: str, generated: str) -> float:
    ref_tokens = _tokenize(reference)
    gen_tokens = _tokenize(generated)
    if not ref_tokens or not gen_tokens:
        return 0.0
    ref_counts = Counter(ref_tokens)
    gen_counts = Counter(gen_tokens)
    overlap = sum(min(ref_counts[token], gen_counts[token]) for token in ref_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / len(gen_tokens)
    recall = overlap / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", _normalized_text(text))


def _normalized_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _contains_term(text: str, term: str) -> bool:
    haystack = _normalized_text(text)
    needle = _normalized_text(term)
    return bool(needle) and needle in haystack


def _contains_pattern(text: str, pattern: str) -> bool:
    try:
        return re.search(pattern, str(text or ""), flags=re.IGNORECASE) is not None
    except re.error:
        return False


def _rounded_average(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return round(mean(filtered), 4)


def _bool_to_float(value: Optional[bool]) -> Optional[float]:
    if value is None:
        return None
    return 1.0 if value else 0.0
