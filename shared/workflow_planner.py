"""Workflow Planner
===================

Pure planning engine for the ``workflow.guided_pipeline`` MCP tool.
Pattern-matches a natural language goal against known workflow templates
and returns a structured execution plan (tool names, params, descriptions).

No side effects — this module never calls any tool or service.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class PlannedStep:
    """A single step in a recommended workflow."""

    tool: str
    params: Dict[str, Any]
    description: str


@dataclass(frozen=True)
class WorkflowPlan:
    """Structured output returned by :class:`WorkflowPlanner`."""

    recommended_workflow: str
    steps: List[PlannedStep]
    estimated_tools_count: int
    servers_needed: List[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_workflow": self.recommended_workflow,
            "steps": [
                {"tool": s.tool, "params": s.params, "description": s.description}
                for s in self.steps
            ],
            "estimated_tools_count": self.estimated_tools_count,
            "servers_needed": self.servers_needed,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Keyword patterns → workflow names
# ---------------------------------------------------------------------------

_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    ("end_to_end", re.compile(
        r"end.to.end|full.pipeline|complete.pipeline|everything|start.to.finish",
        re.IGNORECASE,
    )),
    ("generate_data", re.compile(
        r"generate.*(data|dataset|training)|data.*from.*(doc|pdf|file|text|markdown)"
        r"|create.*(dataset|training.data)|extract.*generat",
        re.IGNORECASE,
    )),
    ("fine_tune", re.compile(
        r"fine.?tun|train|lora|qlora|sft\b|dpo\b|grpo\b|kto\b",
        re.IGNORECASE,
    )),
    ("evaluate", re.compile(
        r"evaluat|score|quality|filter|analyz|statistic|assess",
        re.IGNORECASE,
    )),
    ("deploy", re.compile(
        r"deploy|host|serve|launch.*model|publish",
        re.IGNORECASE,
    )),
    ("orchestration", re.compile(
        r"orchestrat|agent.trajector|tool.use.train|teach.*tool",
        re.IGNORECASE,
    )),
]

DEFAULT_BASE_MODEL = "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# Template builders
# ---------------------------------------------------------------------------

def _end_to_end_steps(
    file_path: str, base_model: str,
) -> List[PlannedStep]:
    return [
        PlannedStep("system.check_resources", {}, "Check GPU/RAM/disk before starting"),
        PlannedStep("system.preflight_check", {"model_name": base_model}, "Verify model fits available VRAM"),
        PlannedStep("extract.from_file", {"file_path": file_path}, "Extract text from document"),
        PlannedStep("generate.from_document", {"file_path": file_path, "technique": "sft"}, "Generate SFT training pairs"),
        PlannedStep("clean.dataset", {"data_points": "$prev.data_points"}, "Deduplicate and validate"),
        PlannedStep("normalize.dataset", {"data_points": "$prev.data_points", "target_format": "sft"}, "Normalize to SFT format"),
        PlannedStep("evaluate.dataset", {"data_points": "$prev.data_points"}, "Score quality of each pair"),
        PlannedStep("evaluate.filter", {"data_points": "$prev.data_points", "threshold": 0.7}, "Remove low-quality pairs"),
        PlannedStep("finetune.train", {"dataset_path": "$prev.output_path", "base_model": base_model}, "LoRA fine-tune"),
        PlannedStep("test.inference", {"model_path": "$prev.model_path", "prompt": "Hello, how can you help me?"}, "Smoke-test the model"),
    ]


def _generate_data_steps(
    file_path: str,
) -> List[PlannedStep]:
    return [
        PlannedStep("extract.from_file", {"file_path": file_path}, "Extract text from document"),
        PlannedStep("generate.from_document", {"file_path": file_path, "technique": "sft"}, "Generate SFT training pairs from document"),
        PlannedStep("clean.dataset", {"data_points": "$prev.data_points"}, "Deduplicate and validate"),
        PlannedStep("normalize.dataset", {"data_points": "$prev.data_points", "target_format": "sft"}, "Normalize to SFT format"),
        PlannedStep("evaluate.dataset", {"data_points": "$prev.data_points"}, "Score quality of each pair"),
        PlannedStep("evaluate.filter", {"data_points": "$prev.data_points", "threshold": 0.7}, "Remove low-quality pairs"),
        PlannedStep("evaluate.statistics", {"data_points": "$prev.data_points"}, "Dataset quality statistics"),
    ]


def _fine_tune_steps(
    file_path: str, base_model: str,
) -> List[PlannedStep]:
    return [
        PlannedStep("system.check_resources", {}, "Check GPU/RAM/disk before training"),
        PlannedStep("system.preflight_check", {"model_name": base_model}, "Verify model fits available VRAM"),
        PlannedStep("finetune.load_dataset", {"file_path": file_path}, "Load and validate training dataset"),
        PlannedStep("finetune.train", {"dataset_path": file_path, "base_model": base_model}, "LoRA fine-tune on dataset"),
        PlannedStep("test.inference", {"model_path": "$prev.model_path", "prompt": "Hello, how can you help me?"}, "Smoke-test the fine-tuned model"),
    ]


def _evaluate_steps() -> List[PlannedStep]:
    return [
        PlannedStep("evaluate.dataset", {"data_points": "<<provide data_points or file path>>"}, "Score quality of each data point"),
        PlannedStep("evaluate.filter", {"data_points": "$prev.data_points", "threshold": 0.7}, "Remove low-quality data points"),
        PlannedStep("evaluate.statistics", {"data_points": "$prev.data_points"}, "Compute dataset quality statistics"),
    ]


def _deploy_steps(base_model: str) -> List[PlannedStep]:
    return [
        PlannedStep("host.start", {"model_path": "<<provide model_path or adapter_path>>", "base_model": base_model}, "Deploy model as API endpoint"),
    ]


def _orchestration_steps() -> List[PlannedStep]:
    return [
        PlannedStep("orchestration.generate_problems", {"domain_description": "<<describe your domain>>", "num_problems": 50}, "Generate synthetic tasks for your domain"),
        PlannedStep("orchestration.collect_trajectories", {"problems": "$prev.problems", "n_per_problem": 4}, "Run agent on problems, record trajectories"),
        PlannedStep("orchestration.build_training_data", {"collected": "$prev.collected", "format": "sft"}, "Score trajectories and convert to training format"),
        PlannedStep("finetune.train", {"dataset_path": "$prev.output_path"}, "Fine-tune on orchestration data"),
    ]


# ---------------------------------------------------------------------------
# Workflow metadata
# ---------------------------------------------------------------------------

_SERVERS: Dict[str, List[str]] = {
    "end_to_end": ["transcendence-data", "transcendence-eval", "transcendence-train", "transcendence-host"],
    "generate_data": ["transcendence-data", "transcendence-eval"],
    "fine_tune": ["transcendence-train"],
    "evaluate": ["transcendence-eval"],
    "deploy": ["transcendence-host"],
    "orchestration": ["transcendence-orchestrate", "transcendence-train"],
}

_NOTES: Dict[str, List[str]] = {
    "end_to_end": [
        "Call system.check_resources first to verify GPU availability.",
        "Steps use $prev references — execute via workflow.run_pipeline or call sequentially.",
        "For 4 GB VRAM, use Qwen/Qwen3-1.7B with 4-bit QLoRA (default).",
    ],
    "generate_data": [
        "Requires OPENAI_API_KEY for LLM-based data generation.",
        "Adjust technique param to 'dpo', 'grpo', or 'kto' for preference data.",
    ],
    "fine_tune": [
        "Call system.check_resources first to verify GPU availability.",
        "For 4 GB VRAM, use Qwen/Qwen3-1.7B with 4-bit QLoRA (default).",
        "Provide a JSONL file with instruction/input/output fields.",
    ],
    "evaluate": [
        "Provide data_points as a list of dicts or a JSONL file path.",
        "Adjust threshold (default 0.7) to control filtering strictness.",
    ],
    "deploy": [
        "Provide the model_path from a previous training run.",
        "Default port is 8001. Set deploy_port to change.",
    ],
    "orchestration": [
        "Requires OPENAI_API_KEY for agent runs and trajectory collection.",
        "Generates training data by recording real agent tool-use trajectories.",
    ],
}


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class WorkflowPlanner:
    """Pattern-matches a goal string and returns a structured execution plan."""

    def plan(
        self,
        goal: str,
        file_path: Optional[str] = None,
        base_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a plan dict ready to be JSON-serialised as tool output."""
        base_model = base_model or DEFAULT_BASE_MODEL
        file_path = file_path or "<<provide file_path>>"

        workflow = self._match_workflow(goal)
        steps = self._build_steps(workflow, file_path, base_model)

        return WorkflowPlan(
            recommended_workflow=workflow,
            steps=steps,
            estimated_tools_count=len(steps),
            servers_needed=_SERVERS.get(workflow, []),
            notes=_NOTES.get(workflow, []),
        ).to_dict()

    # ------------------------------------------------------------------

    @staticmethod
    def _match_workflow(goal: str) -> str:
        for name, pattern in _PATTERNS:
            if pattern.search(goal):
                return name
        return "end_to_end"

    @staticmethod
    def _build_steps(
        workflow: str, file_path: str, base_model: str,
    ) -> List[PlannedStep]:
        builders = {
            "end_to_end": lambda: _end_to_end_steps(file_path, base_model),
            "generate_data": lambda: _generate_data_steps(file_path),
            "fine_tune": lambda: _fine_tune_steps(file_path, base_model),
            "evaluate": lambda: _evaluate_steps(),
            "deploy": lambda: _deploy_steps(base_model),
            "orchestration": lambda: _orchestration_steps(),
        }
        return builders.get(workflow, builders["end_to_end"])()
