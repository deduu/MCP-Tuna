"""Shared manifest model for profiled dataset composition outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from shared.composition_models import CanonicalSchemaKind, TrainerObjective, TuningMode


class CompositionManifest(BaseModel):
    """Sidecar metadata for a profiled dataset generation run."""

    manifest_version: int = 1
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    profile_name: str
    mode: TuningMode
    objective: TrainerObjective
    schema_adapter_name: str
    canonical_kind: CanonicalSchemaKind
    dataset_path: str
    dataset_format: str
    row_target: int
    row_count: int
    requested_mix: dict[str, int]
    resolved_mix: dict[str, int]
    achieved_mix: dict[str, int]
    row_plan: dict[str, int]
    source_paths: list[str]
    source_totals: dict[str, Any]
    source_summaries: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)
