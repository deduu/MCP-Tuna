"""Shared models for profiled dataset composition."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


TuningMode = Literal["general", "coding", "agent"]
TrainerObjective = Literal["sft", "dpo", "grpo", "kto", "vlm_sft"]
CanonicalSchemaKind = Literal[
    "text_sft",
    "preference_pair",
    "reward_group",
    "binary_label",
]


class CapabilityDefinition(BaseModel):
    """Capability metadata shared by composition profiles and evaluation."""

    name: str
    mode: TuningMode
    description: str
    supported_objectives: list[TrainerObjective]

    @field_validator("name", "description")
    @classmethod
    def _validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field must not be empty.")
        return value

    @field_validator("supported_objectives")
    @classmethod
    def _validate_supported_objectives(
        cls, value: list[TrainerObjective]
    ) -> list[TrainerObjective]:
        if not value:
            raise ValueError("supported_objectives must not be empty.")
        return value


class CapabilityTarget(BaseModel):
    """Weighted capability target for a composition profile."""

    capability: str
    weight_percent: int
    min_rows: int = 0
    enabled: bool = True

    @field_validator("capability")
    @classmethod
    def _validate_capability_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("capability must not be empty.")
        return value

    @field_validator("weight_percent", "min_rows")
    @classmethod
    def _validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Value must be non-negative.")
        return value


class CompositionProfile(BaseModel):
    """Default capability mix for a tuning mode."""

    name: str
    mode: TuningMode
    description: str
    default_objective: TrainerObjective
    allowed_objectives: list[TrainerObjective]
    capability_targets: list[CapabilityTarget]

    @field_validator("name", "description")
    @classmethod
    def _validate_profile_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field must not be empty.")
        return value

    @field_validator("allowed_objectives")
    @classmethod
    def _validate_allowed_objectives(
        cls, value: list[TrainerObjective]
    ) -> list[TrainerObjective]:
        if not value:
            raise ValueError("allowed_objectives must not be empty.")
        return value

    @model_validator(mode="after")
    def _validate_profile(self) -> "CompositionProfile":
        if self.default_objective not in self.allowed_objectives:
            raise ValueError("default_objective must exist in allowed_objectives.")

        if not self.capability_targets:
            raise ValueError("capability_targets must not be empty.")

        names = [target.capability for target in self.capability_targets]
        if len(set(names)) != len(names):
            raise ValueError("capability_targets must be unique by capability.")

        enabled_targets = [target for target in self.capability_targets if target.enabled]
        if not enabled_targets:
            raise ValueError("At least one capability target must be enabled.")

        enabled_weight_sum = sum(target.weight_percent for target in enabled_targets)
        if enabled_weight_sum != 100:
            raise ValueError("Enabled capability weights must sum to 100.")

        return self
