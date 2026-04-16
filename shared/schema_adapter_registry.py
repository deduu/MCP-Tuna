"""Built-in and runtime schema adapter registry."""

from __future__ import annotations

from typing import Any

from shared.composition_models import (
    CanonicalSchemaKind,
    TrainerObjective,
    TuningMode,
)
from shared.schema_adapter_models import SchemaAdapter


_SCHEMA_ADAPTERS: dict[str, SchemaAdapter] = {
    "instruction_input_output": SchemaAdapter(
        name="instruction_input_output",
        canonical_kind="text_sft",
        description="Canonical instruction/input/output SFT rows.",
        field_map={
            "instruction": "instruction",
            "input": "input",
            "output": "output",
        },
    ),
    "prompt_response": SchemaAdapter(
        name="prompt_response",
        canonical_kind="text_sft",
        description="Prompt/response style SFT rows.",
        field_map={
            "prompt": "prompt",
            "response": "response",
        },
    ),
    "text_messages": SchemaAdapter(
        name="text_messages",
        canonical_kind="text_sft",
        description=(
            "Structured text messages for chat SFT, including assistant tool_calls "
            "and tool result messages when present."
        ),
        field_map={
            "messages": "messages",
        },
    ),
    "prompt_chosen_rejected": SchemaAdapter(
        name="prompt_chosen_rejected",
        canonical_kind="preference_pair",
        description="Prompt/chosen/rejected preference rows.",
        field_map={
            "prompt": "prompt",
            "chosen": "chosen",
            "rejected": "rejected",
        },
    ),
    "prompt_responses_rewards": SchemaAdapter(
        name="prompt_responses_rewards",
        canonical_kind="reward_group",
        description="Prompt/responses/rewards grouped preference rows.",
        field_map={
            "prompt": "prompt",
            "responses": "responses",
            "rewards": "rewards",
        },
    ),
    "prompt_completion_label": SchemaAdapter(
        name="prompt_completion_label",
        canonical_kind="binary_label",
        description="Prompt/completion/label binary preference rows.",
        field_map={
            "prompt": "prompt",
            "completion": "completion",
            "label": "label",
        },
    ),
}

_OBJECTIVE_TO_SCHEMA_KIND: dict[TrainerObjective, CanonicalSchemaKind] = {
    "sft": "text_sft",
    "vlm_sft": "text_sft",
    "dpo": "preference_pair",
    "grpo": "reward_group",
    "kto": "binary_label",
}

_DEFAULT_OBJECTIVE_ADAPTERS: dict[TrainerObjective, str] = {
    "sft": "instruction_input_output",
    "vlm_sft": "instruction_input_output",
    "dpo": "prompt_chosen_rejected",
    "grpo": "prompt_responses_rewards",
    "kto": "prompt_completion_label",
}

_PROFILE_OBJECTIVE_DEFAULT_ADAPTERS: dict[tuple[TuningMode, TrainerObjective], str] = {
    ("agent", "sft"): "text_messages",
}


def _model_dump(adapter: SchemaAdapter) -> dict[str, Any]:
    return adapter.model_copy(deep=True).model_dump()


def list_schema_adapters(
    canonical_kind: CanonicalSchemaKind | None = None,
) -> list[dict[str, Any]]:
    adapters = _SCHEMA_ADAPTERS.values()
    if canonical_kind is not None:
        adapters = [adapter for adapter in adapters if adapter.canonical_kind == canonical_kind]
    return [_model_dump(adapter) for adapter in adapters]


def get_schema_adapter(name: str) -> dict[str, Any] | None:
    adapter = _SCHEMA_ADAPTERS.get(name)
    return _model_dump(adapter) if adapter is not None else None


def resolve_schema_adapter(name: str) -> SchemaAdapter | None:
    adapter = _SCHEMA_ADAPTERS.get(name)
    return adapter.model_copy(deep=True) if adapter is not None else None


def register_schema_adapter(
    adapter: SchemaAdapter,
    *,
    overwrite: bool = False,
) -> SchemaAdapter:
    existing = _SCHEMA_ADAPTERS.get(adapter.name)
    if existing is not None and not overwrite:
        raise ValueError(f"Schema adapter already exists: {adapter.name}")

    stored = adapter.model_copy(deep=True)
    _SCHEMA_ADAPTERS[stored.name] = stored
    return stored.model_copy(deep=True)


def objective_schema_kind(objective: TrainerObjective) -> CanonicalSchemaKind:
    return _OBJECTIVE_TO_SCHEMA_KIND[objective]


def default_schema_adapter_name(objective: TrainerObjective) -> str | None:
    return _DEFAULT_OBJECTIVE_ADAPTERS.get(objective)


def default_profile_schema_adapter_name(
    mode: TuningMode,
    objective: TrainerObjective,
) -> str | None:
    return _PROFILE_OBJECTIVE_DEFAULT_ADAPTERS.get(
        (mode, objective),
        _DEFAULT_OBJECTIVE_ADAPTERS.get(objective),
    )
