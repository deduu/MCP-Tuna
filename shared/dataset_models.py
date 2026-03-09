"""Pydantic v2 models for dataset persistence operations."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class DatasetConfig(BaseModel):
    """Configuration for dataset operations."""

    model_config = {"frozen": True}

    default_output_dir: str = "./datasets"
    default_format: str = "jsonl"


class DatasetMetadata(BaseModel):
    """Metadata returned by dataset info and save operations."""

    dataset_id: str
    file_path: str
    format: str
    row_count: int
    columns: List[str]
    technique: Optional[str] = None  # sft | dpo | grpo | kto | None
    size_bytes: int = 0
    modified_at: Optional[str] = None


class SplitResult(BaseModel):
    """Result of a dataset split operation."""

    path: str
    count: int
    format: str
