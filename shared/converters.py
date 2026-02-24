"""Cross-pipeline data converters for bridging generator → evaluator → finetuning."""

import json
from dataclasses import asdict
from typing import Any, Dict, List

from .models import BaseDataPoint


def generator_to_evaluator(points: List[BaseDataPoint]) -> List[BaseDataPoint]:
    """Pass-through (same model now). Metadata is preserved."""
    return points


def evaluator_to_finetuning(
    points: List[BaseDataPoint],
    output_path: str,
    fmt: str = "jsonl",
) -> str:
    """Serialize filtered dataset to file for finetuning.

    Returns the output path written.
    """
    rows: List[Dict[str, Any]] = []
    for dp in points:
        row: Dict[str, Any] = {
            "instruction": dp.instruction,
            "input": dp.input,
            "output": dp.output,
        }
        if dp.metadata:
            row["metadata"] = dp.metadata
        rows.append(row)

    if fmt == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif fmt == "json":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return output_path


def finetuning_input_from_file(path: str) -> List[Dict[str, str]]:
    """Load JSONL/JSON and normalize to finetuning-ready dicts.

    Merges instruction+input → prompt, renames output → response.
    """
    records: List[Dict[str, Any]] = []
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    result: List[Dict[str, str]] = []
    for rec in records:
        instruction = rec.get("instruction", "")
        inp = rec.get("input", "")
        prompt = f"{instruction} {inp}".strip() if inp else instruction
        result.append({
            "prompt": prompt,
            "response": rec.get("output", ""),
        })
    return result
