from __future__ import annotations

from orchestration.benchmarking import (
    aggregate_benchmark_runs,
    normalize_benchmark_cases,
    normalize_seed_list,
    score_benchmark_case,
)


def test_normalize_seed_list_defaults_and_deduplicates():
    assert normalize_seed_list(None) == [42, 43, 44]
    assert normalize_seed_list([7, 7, 8]) == [7, 8]


def test_normalize_benchmark_cases_supports_chat_triplets():
    cases = normalize_benchmark_cases(
        [
            {
                "system": "Be concise.",
                "user": "Apa itu Salestify?",
                "assistant": "Platform CRM WhatsApp.",
                "required_terms": ["CRM", "WhatsApp"],
                "forbidden_terms": "Rp, $",
                "forbidden_patterns": [r"\brp\.?\s?\d"],
            }
        ],
        pack_name="dev",
    )

    assert len(cases) == 1
    assert cases[0]["prompt"] == "Apa itu Salestify?"
    assert cases[0]["reference"] == "Platform CRM WhatsApp."
    assert cases[0]["system_prompt"] == "Be concise."
    assert cases[0]["required_terms"] == ["CRM", "WhatsApp"]
    assert cases[0]["forbidden_terms"] == ["Rp", "$"]
    assert cases[0]["forbidden_patterns"] == [r"\brp\.?\s?\d"]


def test_score_benchmark_case_tracks_term_gates_and_composite():
    case = {
        "case_id": "safety_1",
        "prompt": "Berapa harganya?",
        "reference": "Silakan hubungi tim sales untuk info harga.",
        "required_terms": ["sales"],
        "forbidden_terms": ["Rp", "$"],
        "min_token_f1": 0.25,
    }

    scored = score_benchmark_case(
        case,
        "Silakan hubungi tim sales untuk info harga lebih lanjut.",
    )

    assert scored["required_terms_pass"] is True
    assert scored["forbidden_terms_pass"] is True
    assert scored["min_token_f1_pass"] is True
    assert scored["case_pass"] is True
    assert scored["composite_score"] > 0.6


def test_score_benchmark_case_supports_regex_pattern_gates():
    case = {
        "case_id": "pricing_1",
        "prompt": "Berapa harganya?",
        "reference": "Harga mengikuti kebutuhan. Silakan lanjut ke tim sales.",
        "required_patterns": [r"\btim sales\b"],
        "forbidden_patterns": [r"\brp\.?\s?\d", r"\b\d+\s*(ribu|juta)\b"],
    }

    passing = score_benchmark_case(
        case,
        "Harga mengikuti kebutuhan, jadi paling aman lanjut ke tim sales.",
    )
    failing = score_benchmark_case(
        case,
        "Paket mulai Rp 299 ribu. Hubungi sales untuk detail.",
    )

    assert passing["required_patterns_pass"] is True
    assert passing["forbidden_patterns_pass"] is True
    assert passing["case_pass"] is True
    assert failing["required_patterns_pass"] is False
    assert failing["forbidden_patterns_pass"] is False
    assert failing["forbidden_patterns_hits"] == [r"\brp\.?\s?\d", r"\b\d+\s*(ribu|juta)\b"]


def test_aggregate_benchmark_runs_ranks_methods_and_evaluates_gates():
    runs = [
        {
            "method": "flat_sft",
            "seed": 11,
            "evaluation": {
                "hidden_holdout": {
                    "summary": {"avg_composite_score": 0.41, "pass_rate": 0.5}
                }
            },
        },
        {
            "method": "curriculum_sft",
            "seed": 11,
            "evaluation": {
                "hidden_holdout": {
                    "summary": {"avg_composite_score": 0.56, "pass_rate": 0.75}
                }
            },
        },
        {
            "method": "curriculum_sft",
            "seed": 12,
            "evaluation": {
                "hidden_holdout": {
                    "summary": {"avg_composite_score": 0.6, "pass_rate": 0.8}
                }
            },
        },
    ]

    summary = aggregate_benchmark_runs(
        runs,
        primary_pack="hidden_holdout",
        benchmark_gates=[
            {
                "name": "curriculum_margin",
                "candidate_method": "curriculum_sft",
                "baseline_method": "flat_sft",
                "pack": "hidden_holdout",
                "metric": "avg_composite_score",
                "min_delta": 0.1,
            }
        ],
    )

    assert summary["best_method"] == "curriculum_sft"
    assert summary["method_rankings"][0]["primary_score_mean"] == 0.58
    assert summary["gates"][0]["passed"] is True
    assert summary["gates"][0]["delta"] == 0.17
