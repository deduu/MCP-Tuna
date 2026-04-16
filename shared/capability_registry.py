"""Built-in capability definitions and composition profiles."""

from __future__ import annotations

from typing import Any

from shared.composition_models import (
    CapabilityDefinition,
    CapabilityTarget,
    CompositionProfile,
    TuningMode,
)


def _capability(
    *,
    name: str,
    mode: TuningMode,
    description: str,
    supported_objectives: list[str],
) -> CapabilityDefinition:
    return CapabilityDefinition(
        name=name,
        mode=mode,
        description=description,
        supported_objectives=supported_objectives,
    )


_CAPABILITIES: dict[str, CapabilityDefinition] = {
    "instruction_following": _capability(
        name="instruction_following",
        mode="general",
        description="Follow the requested task, format, and constraints reliably.",
        supported_objectives=["sft", "dpo", "grpo", "kto"],
    ),
    "grounded_qa": _capability(
        name="grounded_qa",
        mode="general",
        description="Answer fact-seeking questions using only the supplied source.",
        supported_objectives=["sft", "dpo", "grpo", "kto"],
    ),
    "grounded_synthesis": _capability(
        name="grounded_synthesis",
        mode="general",
        description="Summarize or synthesize multiple grounded facts into one answer.",
        supported_objectives=["sft", "dpo", "grpo", "kto"],
    ),
    "reasoning": _capability(
        name="reasoning",
        mode="general",
        description="Combine grounded facts into a derived conclusion.",
        supported_objectives=["sft", "dpo", "grpo", "kto"],
    ),
    "multi_hop": _capability(
        name="multi_hop",
        mode="general",
        description="Require the answer to connect evidence across chunks or sections.",
        supported_objectives=["sft", "dpo", "grpo", "kto"],
    ),
    "unanswerable_or_refusal": _capability(
        name="unanswerable_or_refusal",
        mode="general",
        description="Refuse or state insufficiency when the source does not support an answer.",
        supported_objectives=["sft", "dpo", "grpo", "kto"],
    ),
    "repo_qa": _capability(
        name="repo_qa",
        mode="coding",
        description="Answer questions grounded in a repository or codebase.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "bug_localization": _capability(
        name="bug_localization",
        mode="coding",
        description="Find likely faulty files, functions, or lines from code/task context.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "patch_generation": _capability(
        name="patch_generation",
        mode="coding",
        description="Produce focused code changes for a requested task or fix.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "test_debug": _capability(
        name="test_debug",
        mode="coding",
        description="Interpret failing tests and derive the likely fix path.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "code_review": _capability(
        name="code_review",
        mode="coding",
        description="Identify implementation risks, regressions, and missing tests.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "tool_use_planning": _capability(
        name="tool_use_planning",
        mode="coding",
        description="Plan what repository reads, searches, or tests to run before editing.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "tool_selection": _capability(
        name="tool_selection",
        mode="agent",
        description="Choose the right tool instead of answering directly.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "argument_fidelity": _capability(
        name="argument_fidelity",
        mode="agent",
        description="Fill tool arguments accurately and schema-correctly.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "tool_result_grounding": _capability(
        name="tool_result_grounding",
        mode="agent",
        description="Use tool outputs as hard evidence for the final answer.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "multi_step_state": _capability(
        name="multi_step_state",
        mode="agent",
        description="Track intermediate state across several dependent actions.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "recovery": _capability(
        name="recovery",
        mode="agent",
        description="Recover from tool failures or partial outputs.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
    "stop_or_no_tool": _capability(
        name="stop_or_no_tool",
        mode="agent",
        description="Know when to stop calling tools or avoid calling one at all.",
        supported_objectives=["sft", "dpo", "grpo"],
    ),
}


def _target(capability: str, weight_percent: int) -> CapabilityTarget:
    return CapabilityTarget(capability=capability, weight_percent=weight_percent)


_PROFILES: dict[str, CompositionProfile] = {
    "general_instruction": CompositionProfile(
        name="general_instruction",
        mode="general",
        description=(
            "Balanced default for grounded instruction tuning with reasoning and "
            "modest multi-hop coverage."
        ),
        default_objective="sft",
        allowed_objectives=["sft", "dpo", "grpo", "kto"],
        capability_targets=[
            _target("grounded_qa", 25),
            _target("grounded_synthesis", 20),
            _target("instruction_following", 20),
            _target("reasoning", 15),
            _target("multi_hop", 10),
            _target("unanswerable_or_refusal", 10),
        ],
    ),
    "coding_assistant": CompositionProfile(
        name="coding_assistant",
        mode="coding",
        description=(
            "Default coding assistant mix for repo grounding, bug finding, patching, "
            "and debugging."
        ),
        default_objective="sft",
        allowed_objectives=["sft", "dpo", "grpo"],
        capability_targets=[
            _target("patch_generation", 20),
            _target("test_debug", 20),
            _target("repo_qa", 15),
            _target("bug_localization", 15),
            _target("code_review", 10),
            _target("tool_use_planning", 10),
            _target("instruction_following", 10),
        ],
    ),
    "agent_tool_calling": CompositionProfile(
        name="agent_tool_calling",
        mode="agent",
        description=(
            "Default tool-calling mix for tool choice, argument correctness, "
            "state tracking, and recovery."
        ),
        default_objective="sft",
        allowed_objectives=["sft", "dpo", "grpo"],
        capability_targets=[
            _target("tool_selection", 20),
            _target("argument_fidelity", 20),
            _target("multi_step_state", 20),
            _target("tool_result_grounding", 15),
            _target("recovery", 15),
            _target("stop_or_no_tool", 10),
        ],
    ),
}


def _model_dump(model: Any) -> dict[str, Any]:
    return model.model_copy(deep=True).model_dump()


def list_capabilities(mode: TuningMode | None = None) -> list[dict[str, Any]]:
    capabilities = _CAPABILITIES.values()
    if mode is not None:
        capabilities = [cap for cap in capabilities if cap.mode == mode]
    return [_model_dump(cap) for cap in capabilities]


def get_capability(name: str) -> dict[str, Any] | None:
    capability = _CAPABILITIES.get(name)
    return _model_dump(capability) if capability is not None else None


def resolve_capability(name: str) -> CapabilityDefinition | None:
    capability = _CAPABILITIES.get(name)
    return capability.model_copy(deep=True) if capability is not None else None


def list_profiles(mode: TuningMode | None = None) -> list[dict[str, Any]]:
    profiles = _PROFILES.values()
    if mode is not None:
        profiles = [profile for profile in profiles if profile.mode == mode]
    return [_model_dump(profile) for profile in profiles]


def get_profile(name: str) -> dict[str, Any] | None:
    profile = _PROFILES.get(name)
    return _model_dump(profile) if profile is not None else None


def resolve_profile(name: str) -> CompositionProfile | None:
    profile = _PROFILES.get(name)
    return profile.model_copy(deep=True) if profile is not None else None
