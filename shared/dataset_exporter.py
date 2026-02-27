"""Dataset export utilities.

Exports lists of dicts to JSON, JSONL, CSV, Excel, and HuggingFace formats.
Used by orchestration and data_generator pipelines.
"""
from __future__ import annotations

import json
from typing import Dict, List


class DatasetExporter:
    """Export datasets to various formats."""

    @staticmethod
    def to_json(data: List[Dict], output_path: str) -> None:
        """Export to JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def to_jsonl(data: List[Dict], output_path: str) -> None:
        """Export to JSONL (one JSON object per line)."""
        with open(output_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def to_excel(data: List[Dict], output_path: str) -> None:
        """Export to Excel."""
        import pandas as pd

        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)

    @staticmethod
    def to_csv(data: List[Dict], output_path: str) -> None:
        """Export to CSV."""
        import pandas as pd

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    @staticmethod
    def to_huggingface(data: List[Dict], output_path: str) -> None:
        """Export in HuggingFace datasets format."""
        from datasets import Dataset

        dataset = Dataset.from_list(data)
        dataset.save_to_disk(output_path)
