import json
from ..core.data import DataPoint


def load_jsonl(path: str):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if not data.get("instruction") or not data.get("output"):
                continue

            dataset.append(
                DataPoint(
                    instruction=data.get("instruction", ""),
                    input=data.get("input", ""),
                    output=data.get("output", ""),
                    metadata=data.get("metadata"),
                )
            )
    return dataset
