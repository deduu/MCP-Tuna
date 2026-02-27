"""Pipeline Executor
====================

Server-side engine for executing a sequence of MCP tools in a single call.
Resolves ``$prev.key`` references so each step can consume the previous
step's output without client round-trips.
"""
from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_STEPS = 20


# ---------------------------------------------------------------------------
# Pydantic v2 models
# ---------------------------------------------------------------------------

class PipelineStep(BaseModel):
    """A single step in a pipeline: which tool to call and with what params."""

    model_config = {"frozen": True}

    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)


class StepResult(BaseModel):
    """Outcome of a single pipeline step."""

    step: int
    tool: str
    success: bool
    result: Dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    """Aggregate outcome of a full pipeline execution."""

    success: bool
    steps_completed: int
    total_steps: int
    error: Optional[str] = None
    results: List[StepResult] = Field(default_factory=list)
    final_result: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class PipelineExecutor:
    """Execute a list of MCP tool calls sequentially, chaining results.

    Parameters
    ----------
    tools:
        The ``gateway.mcp._tools`` dict. Each key is a tool name, each value
        is ``{"func": <async callable>, "description": str, "schema": dict}``.
    """

    def __init__(self, tools: Dict[str, Any]) -> None:
        self._tools = tools

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def execute(
        self,
        steps: List[PipelineStep],
        dry_run: bool = False,
    ) -> PipelineResult:
        """Run *steps* sequentially.  Stop on first failure.

        If *dry_run* is ``True``, validate only — no tools are called.
        """
        total = len(steps)

        # --- Validation ---
        err = self._validate(steps)
        if err is not None:
            return PipelineResult(
                success=False,
                steps_completed=0,
                total_steps=total,
                error=err,
                results=[],
            )

        if dry_run:
            return PipelineResult(
                success=True,
                steps_completed=0,
                total_steps=total,
                results=[],
            )

        # --- Execution ---
        results: List[StepResult] = []
        prev_result: Dict[str, Any] = {}

        for idx, step in enumerate(steps):
            # Resolve $prev references
            try:
                resolved_params = self._resolve_refs(step.params, prev_result, idx)
            except _RefError as exc:
                results.append(StepResult(
                    step=idx, tool=step.tool, success=False,
                    result={"error": str(exc)},
                ))
                return PipelineResult(
                    success=False,
                    steps_completed=idx,
                    total_steps=total,
                    error=f"Step {idx} ({step.tool}) ref error: {exc}",
                    results=results,
                )

            # Call the tool
            try:
                raw = await self._tools[step.tool]["func"](**resolved_params)
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                results.append(StepResult(
                    step=idx, tool=step.tool, success=False,
                    result={"error": "Tool returned invalid JSON"},
                ))
                return PipelineResult(
                    success=False,
                    steps_completed=idx,
                    total_steps=total,
                    error=f"Step {idx} ({step.tool}) returned invalid JSON",
                    results=results,
                )
            except Exception as exc:
                results.append(StepResult(
                    step=idx, tool=step.tool, success=False,
                    result={"error": str(exc)},
                ))
                return PipelineResult(
                    success=False,
                    steps_completed=idx,
                    total_steps=total,
                    error=f"Step {idx} ({step.tool}) exception: {exc}",
                    results=results,
                )

            step_ok = parsed.get("success", True)
            results.append(StepResult(
                step=idx, tool=step.tool, success=step_ok, result=parsed,
            ))

            if not step_ok:
                return PipelineResult(
                    success=False,
                    steps_completed=idx,
                    total_steps=total,
                    error=f"Step {idx} ({step.tool}) failed: {parsed.get('error', 'unknown')}",
                    results=results,
                )

            prev_result = parsed
            logger.info("Pipeline step %d/%d (%s) succeeded", idx + 1, total, step.tool)

        return PipelineResult(
            success=True,
            steps_completed=total,
            total_steps=total,
            results=results,
            final_result=prev_result,
        )

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #

    def _validate(self, steps: List[PipelineStep]) -> Optional[str]:
        """Return an error message if *steps* are invalid, else ``None``."""
        if not steps:
            return "Pipeline must contain at least 1 step"

        if len(steps) > MAX_STEPS:
            return f"Pipeline exceeds maximum of {MAX_STEPS} steps ({len(steps)} given)"

        for idx, step in enumerate(steps):
            if step.tool not in self._tools:
                return f"Unknown tool '{step.tool}' in step {idx}"
            if idx == 0 and self._has_prev_refs(step.params):
                return f"Step 0 cannot use $prev references (found in params of '{step.tool}')"

        return None

    # ------------------------------------------------------------------ #
    # Reference resolution
    # ------------------------------------------------------------------ #

    def _resolve_refs(
        self,
        params: Dict[str, Any],
        prev_result: Dict[str, Any],
        step_idx: int,
    ) -> Dict[str, Any]:
        """Deep-copy *params* and replace ``$prev.key`` strings."""
        resolved = deepcopy(params)
        return self._resolve_value(resolved, prev_result, step_idx)

    def _resolve_value(self, value: Any, prev: Dict[str, Any], step_idx: int) -> Any:
        if isinstance(value, str) and value.startswith("$prev."):
            key = value[len("$prev."):]
            if key not in prev:
                raise _RefError(
                    f"$prev.{key} referenced in step {step_idx} "
                    f"but previous result has no key '{key}'"
                )
            return prev[key]
        if isinstance(value, dict):
            return {k: self._resolve_value(v, prev, step_idx) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_value(item, prev, step_idx) for item in value]
        return value

    @staticmethod
    def _has_prev_refs(value: Any) -> bool:
        """Return True if *value* contains any ``$prev.*`` string."""
        if isinstance(value, str):
            return value.startswith("$prev.")
        if isinstance(value, dict):
            return any(PipelineExecutor._has_prev_refs(v) for v in value.values())
        if isinstance(value, list):
            return any(PipelineExecutor._has_prev_refs(item) for item in value)
        return False


class _RefError(Exception):
    """Raised when a ``$prev.key`` reference cannot be resolved."""
