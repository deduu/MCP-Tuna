from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.runtime_python import resolve_repo_python


COMPARE_SCRIPT = ROOT / "scripts" / "compare_wa_sales_preference.py"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] [wa_pref_sweep] {message}", flush=True)


def parse_args() -> Any:
    parser = ArgumentParser(description="Run a seeded DPO/GRPO compare sweep and aggregate the benchmark results.")
    parser.add_argument(
        "--seeds",
        default=os.getenv("SWEEP_SEEDS", "3407,3408,3409"),
        help="Comma-separated training seeds to evaluate.",
    )
    parser.add_argument(
        "--methods",
        default=os.getenv("METHODS", "dpo,grpo"),
        help="Comma-separated preference methods to train in each run.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "SWEEP_RUN_DIR",
            str(ROOT / "output" / f"wa_sales_preference_sweep_{time.strftime('%Y%m%d_%H%M%S')}"),
        ),
        help="Directory that will contain one compare run per seed plus an aggregate summary.",
    )
    return parser.parse_args()


def parse_seed_list(raw: str) -> List[int]:
    seeds = [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not seeds:
        raise ValueError("Provide at least one seed in --seeds.")
    return seeds


def repo_python() -> str:
    preferred = resolve_repo_python(ROOT)
    return str(preferred if preferred is not None else Path(sys.executable))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_compare(seed: int, methods: str, run_dir: Path) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "SEED": str(seed),
            "METHODS": methods,
            "RUN_DIR": str(run_dir),
        }
    )
    log(f"Running compare script for seed {seed} -> {run_dir}")
    subprocess.run(
        [repo_python(), str(COMPARE_SCRIPT)],
        cwd=str(ROOT),
        env=env,
        check=True,
    )
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Compare run did not produce results.json: {results_path}")
    return load_json(results_path)


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return round(sum(collected) / len(collected), 4) if collected else 0.0


def _stdev(values: Iterable[float]) -> float:
    collected = list(values)
    return round(statistics.pstdev(collected), 4) if len(collected) > 1 else 0.0


def build_summary(run_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful_runs = [record for record in run_records if record.get("success")]
    if not successful_runs:
        return {
            "successful_runs": 0,
            "failed_runs": len(run_records),
            "method_rankings": [],
            "pairwise_gaps": {},
        }

    score_map: Dict[str, List[Dict[str, Any]]] = {}
    primary_metric = "avg_composite_score"
    primary_pack = "dev"

    for record in successful_runs:
        benchmark = record["benchmark"]
        summary = benchmark.get("summary", {})
        primary_metric = summary.get("primary_metric", primary_metric)
        primary_pack = summary.get("primary_pack", primary_pack)
        for ranking in summary.get("method_rankings", []):
            score_map.setdefault(ranking["method"], []).append(
                {
                    "seed": record["seed"],
                    "score": float(ranking.get("primary_score_mean", 0.0)),
                    "run_dir": record["run_dir"],
                }
            )

    rankings = []
    sft_scores = {entry["seed"]: entry["score"] for entry in score_map.get("sft_init", [])}
    for method, entries in score_map.items():
        scores = [entry["score"] for entry in entries]
        mean_score = _mean(scores)
        method_summary: Dict[str, Any] = {
            "method": method,
            "run_count": len(entries),
            "seed_scores": entries,
            "mean_primary_score": mean_score,
            "stdev_primary_score": _stdev(scores),
        }
        comparable_gaps = [
            round(entry["score"] - sft_scores[entry["seed"]], 4)
            for entry in entries
            if entry["seed"] in sft_scores and method != "sft_init"
        ]
        if comparable_gaps:
            method_summary["mean_gap_vs_sft"] = _mean(comparable_gaps)
            method_summary["stdev_gap_vs_sft"] = _stdev(comparable_gaps)
            method_summary["beats_sft_count"] = sum(1 for gap in comparable_gaps if gap > 0)
        rankings.append(method_summary)

    rankings.sort(key=lambda item: item["mean_primary_score"], reverse=True)

    pairwise_gaps: Dict[str, Any] = {}
    for technique in ("dpo", "grpo"):
        mcp_method = f"mcp_{technique}"
        notebook_method = f"notebook_{technique}"
        if mcp_method not in score_map or notebook_method not in score_map:
            continue
        notebook_by_seed = {entry["seed"]: entry["score"] for entry in score_map[notebook_method]}
        seed_gaps = []
        for entry in score_map[mcp_method]:
            notebook_score = notebook_by_seed.get(entry["seed"])
            if notebook_score is None:
                continue
            seed_gaps.append(
                {
                    "seed": entry["seed"],
                    "mcp_minus_notebook": round(entry["score"] - notebook_score, 4),
                }
            )
        if seed_gaps:
            gap_values = [entry["mcp_minus_notebook"] for entry in seed_gaps]
            pairwise_gaps[technique] = {
                "seed_gaps": seed_gaps,
                "mean_mcp_minus_notebook": _mean(gap_values),
                "stdev_mcp_minus_notebook": _stdev(gap_values),
                "mcp_wins": sum(1 for gap in gap_values if gap > 0),
                "notebook_wins": sum(1 for gap in gap_values if gap < 0),
                "ties": sum(1 for gap in gap_values if gap == 0),
            }

    return {
        "successful_runs": len(successful_runs),
        "failed_runs": len(run_records) - len(successful_runs),
        "primary_pack": primary_pack,
        "primary_metric": primary_metric,
        "method_rankings": rankings,
        "pairwise_gaps": pairwise_gaps,
    }


def main() -> None:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_records: List[Dict[str, Any]] = []
    for seed in seeds:
        run_dir = output_dir / f"seed_{seed}"
        try:
            results = run_compare(seed, args.methods, run_dir)
            benchmark = results.get("benchmark") or {}
            benchmark_path = benchmark.get("results_path")
            run_records.append(
                {
                    "seed": seed,
                    "success": bool(results.get("success")),
                    "error": results.get("error"),
                    "run_dir": str(run_dir),
                    "results_path": str(run_dir / "results.json"),
                    "benchmark_results_path": benchmark_path,
                    "benchmark": benchmark,
                }
            )
        except Exception as exc:
            run_records.append(
                {
                    "seed": seed,
                    "success": False,
                    "error": str(exc),
                    "run_dir": str(run_dir),
                    "results_path": str(run_dir / "results.json"),
                    "benchmark_results_path": None,
                }
            )
            log(f"Seed {seed} failed: {exc}")

    summary = build_summary(run_records)
    summary_payload = {
        "success": summary["successful_runs"] > 0,
        "output_dir": str(output_dir),
        "compare_script": str(COMPARE_SCRIPT),
        "seeds": seeds,
        "methods": [method.strip() for method in args.methods.split(",") if method.strip()],
        "runs": run_records,
        "summary": summary,
    }
    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Saved sweep summary -> {summary_path}")

    if summary["successful_runs"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
