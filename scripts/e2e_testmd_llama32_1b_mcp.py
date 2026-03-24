from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from mcp_gateway import TunaGateway


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO_ROOT / "data" / "test.md"
BASE_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = REPO_ROOT / "output" / f"testmd_llama32_1b_4bit_{RUN_STAMP}"
DATA_DIR = RUN_DIR / "data"
MODEL_DIR = RUN_DIR / "adapter"
REPORT_PATH = RUN_DIR / "report.json"
DATASET_PATH = DATA_DIR / "testmd_sft_10.jsonl"
SPLIT_DIR = DATA_DIR / "splits"


def _hf_home() -> Path:
    return Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))


def resolve_latest_snapshot(model_id: str) -> str:
    snapshots_dir = _hf_home() / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    if not snapshots_dir.exists():
        return model_id
    snapshots = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
    if not snapshots:
        return model_id
    return str(max(snapshots, key=lambda entry: entry.stat().st_mtime))


async def call_tool(gateway: TunaGateway, tool_name: str, **kwargs: Any) -> Dict[str, Any]:
    raw = await gateway.mcp._tools[tool_name]["func"](**kwargs)
    return json.loads(raw)


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("â€¢", " ").replace("•", " ")
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def truncate_to_ten(data_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(data_points) < 10:
        raise RuntimeError(f"Expected at least 10 generated rows, got {len(data_points)}")
    trimmed = []
    seen = set()
    for row in data_points:
        key = (
            str(row.get("instruction", "")).strip(),
            str(row.get("input", "")).strip(),
            str(row.get("output", "")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        trimmed.append(
            {
                "instruction": key[0],
                "input": key[1],
                "output": key[2],
            }
        )
        if len(trimmed) == 10:
            return trimmed
    raise RuntimeError(f"Unable to keep 10 unique rows after deduplication; got {len(trimmed)}")


def build_eval_rows(test_rows: List[Dict[str, Any]], comparisons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for index, (row, comp) in enumerate(zip(test_rows, comparisons), start=1):
        prompt = f"{row.get('instruction', '')} {row.get('input', '')}".strip()
        reference = str(row.get("output", "")).strip()
        base_response = str(comp.get("base_response", "")).strip()
        tuned_response = str(comp.get("finetuned_response", "")).strip()
        base_score = token_f1(base_response, reference)
        tuned_score = token_f1(tuned_response, reference)
        if math.isclose(base_score, tuned_score, rel_tol=1e-9, abs_tol=1e-9):
            winner = "tie"
        elif tuned_score > base_score:
            winner = "finetuned"
        else:
            winner = "base"
        results.append(
            {
                "id": index,
                "prompt": prompt,
                "reference": reference,
                "base_response": base_response,
                "finetuned_response": tuned_response,
                "base_token_f1": round(base_score, 4),
                "finetuned_token_f1": round(tuned_score, 4),
                "winner": winner,
                "base_time_seconds": round(float(comp.get("base_time", 0.0)), 4),
                "finetuned_time_seconds": round(float(comp.get("finetuned_time", 0.0)), 4),
                "base_tps": round(float(comp.get("base_tps", 0.0)), 4),
                "finetuned_tps": round(float(comp.get("finetuned_tps", 0.0)), 4),
            }
        )
    return results


def summarize_eval(eval_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_wins = sum(1 for row in eval_rows if row["winner"] == "base")
    tuned_wins = sum(1 for row in eval_rows if row["winner"] == "finetuned")
    ties = sum(1 for row in eval_rows if row["winner"] == "tie")
    base_avg = sum(row["base_token_f1"] for row in eval_rows) / len(eval_rows)
    tuned_avg = sum(row["finetuned_token_f1"] for row in eval_rows) / len(eval_rows)
    return {
        "num_queries": len(eval_rows),
        "base_avg_token_f1": round(base_avg, 4),
        "finetuned_avg_token_f1": round(tuned_avg, 4),
        "absolute_gain": round(tuned_avg - base_avg, 4),
        "finetuned_wins": tuned_wins,
        "base_wins": base_wins,
        "ties": ties,
    }


async def generate_dataset(gateway: TunaGateway) -> List[Dict[str, Any]]:
    result = await call_tool(gateway, "generate.from_document", technique="sft", file_path=str(SOURCE_PATH))
    if not result.get("success"):
        raise RuntimeError(result.get("error") or "Dataset generation failed")

    cleaned = await call_tool(gateway, "clean.dataset", data_points=result.get("data_points", []))
    if not cleaned.get("success"):
        raise RuntimeError(cleaned.get("error") or "Dataset cleaning failed")

    normalized = await call_tool(
        gateway,
        "normalize.dataset",
        data_points=cleaned.get("data_points", []),
        target_format="sft",
    )
    if not normalized.get("success"):
        raise RuntimeError(normalized.get("error") or "Dataset normalization failed")

    return truncate_to_ten(normalized.get("data_points", []))


async def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base_model_path = resolve_latest_snapshot(BASE_MODEL_ID)
    gateway = TunaGateway()

    generated_rows = await generate_dataset(gateway)

    save_result = await call_tool(
        gateway,
        "dataset.save",
        data_points=generated_rows,
        output_path=str(DATASET_PATH),
        format="jsonl",
    )
    if not save_result.get("success"):
        raise RuntimeError(save_result.get("error") or "Dataset save failed")

    split_result = await call_tool(
        gateway,
        "dataset.split",
        file_path=str(DATASET_PATH),
        output_dir=str(SPLIT_DIR),
        train_ratio=0.7,
        val_ratio=0.0,
        test_ratio=0.3,
        seed=42,
    )
    if not split_result.get("success"):
        raise RuntimeError(split_result.get("error") or "Dataset split failed")

    train_path = split_result["splits"]["train"]["path"]
    test_path = split_result["splits"]["test"]["path"]

    preflight = await call_tool(
        gateway,
        "system.preflight_check",
        model_name=base_model_path,
        quantization="4bit",
        batch_size=1,
        max_seq_length=512,
        technique="sft",
        use_lora=True,
        lora_r=8,
        gradient_checkpointing=False,
    )

    train_result = await call_tool(
        gateway,
        "finetune.train",
        dataset_path=train_path,
        output_dir=str(MODEL_DIR),
        base_model=base_model_path,
        num_epochs=6,
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
        completion_only_loss=False,
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        max_seq_length=512,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=False,
        load_in_4bit=True,
    )
    if not train_result.get("success"):
        raise RuntimeError(train_result.get("error") or "Training failed")

    test_loaded = await call_tool(gateway, "dataset.load", file_path=test_path)
    if not test_loaded.get("success"):
        raise RuntimeError(test_loaded.get("error") or "Test split load failed")
    test_rows = test_loaded.get("data_points", [])
    prompts = [f"{row.get('instruction', '')} {row.get('input', '')}".strip() for row in test_rows]

    compare_result = await call_tool(
        gateway,
        "test.compare_models",
        prompts=prompts,
        base_model_path=base_model_path,
        finetuned_adapter_path=str(MODEL_DIR),
        max_new_tokens=192,
    )
    if not compare_result.get("success"):
        raise RuntimeError(compare_result.get("error") or "Comparison failed")

    eval_rows = build_eval_rows(test_rows, compare_result.get("comparisons", []))
    summary = summarize_eval(eval_rows)

    report = {
        "source_path": str(SOURCE_PATH),
        "base_model_id": BASE_MODEL_ID,
        "base_model_path": base_model_path,
        "run_dir": str(RUN_DIR),
        "dataset": {
            "saved_path": str(DATASET_PATH),
            "row_count": len(generated_rows),
            "train_path": train_path,
            "test_path": test_path,
            "split_counts": {
                "train": split_result["splits"]["train"]["count"],
                "val": split_result["splits"]["val"]["count"],
                "test": split_result["splits"]["test"]["count"],
            },
            "sample_rows": generated_rows[:2],
        },
        "preflight": preflight,
        "training": train_result,
        "comparison_summary": summary,
        "comparison_details": eval_rows,
    }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"report_path": str(REPORT_PATH), "comparison_summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
