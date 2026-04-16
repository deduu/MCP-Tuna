"""Shared models for mapping user schemas to canonical trainer shapes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from shared.composition_models import CanonicalSchemaKind


class SchemaAdapter(BaseModel):
    """Mapping between a user-facing row schema and a canonical internal schema."""

    name: str
    canonical_kind: CanonicalSchemaKind
    description: str = ""
    field_map: dict[str, str]
    defaults: dict[str, Any] = Field(default_factory=dict)
    strict: bool = True

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be empty.")
        return value

    @field_validator("description")
    @classmethod
    def _normalize_description(cls, value: str) -> str:
        return value.strip()

    @field_validator("field_map")
    @classmethod
    def _validate_field_map(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("field_map must not be empty.")

        normalized: dict[str, str] = {}
        for canonical_field, source_field in value.items():
            canonical_name = str(canonical_field).strip()
            source_name = str(source_field).strip()
            if not canonical_name or not source_name:
                raise ValueError("field_map keys and values must be non-empty strings.")
            normalized[canonical_name] = source_name
        return normalized
