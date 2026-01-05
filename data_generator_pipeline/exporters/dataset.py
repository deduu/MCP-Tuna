# ============================================================================
# FILE: src/finetuning/exporters/dataset.py
# ============================================================================

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict


class DatasetExporter:
    """Export datasets to various formats."""

    @staticmethod
    def to_json(data: List[Dict], output_path: str):
        """Export to JSON."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def to_jsonl(data: List[Dict], output_path: str):
        """Export to JSONL (one JSON object per line)."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    @staticmethod
    def to_excel(data: List[Dict], output_path: str):
        """Export to Excel."""
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)

    @staticmethod
    def to_csv(data: List[Dict], output_path: str):
        """Export to CSV."""
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)

    @staticmethod
    def to_huggingface(data: List[Dict], output_path: str):
        """Export in HuggingFace datasets format."""
        from datasets import Dataset

        dataset = Dataset.from_list(data)
        dataset.save_to_disk(output_path)
