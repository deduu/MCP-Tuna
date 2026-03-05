"""Confirmation gate for critical tool execution.

Returns an async callback that AgentSoul's ``tool_pre_execute`` hook calls
before running each tool.  When confirmation is required the callback returns
a dict that the agent yields as a ``confirmation_needed`` event and then stops.
"""
from __future__ import annotations

from typing import Any, Callable, Coroutine, Dict, Optional

# Tools whose execution should be gated when require_approval == "critical"
CRITICAL_TOOLS = {
    "finetune.train",
    "finetune.train_dpo",
    "finetune.train_grpo",
    "finetune.train_kto",
    "finetune.train_curriculum",
    "finetune.sequential_train",
    "finetune.train_async",
    "host.deploy_mcp",
    "host.deploy_api",
    "clean.dataset",
    "workflow.full_pipeline",
}

ToolPreExecute = Callable[
    [str, Dict[str, Any]],
    Coroutine[Any, Any, Optional[Dict[str, Any]]],
]


def build_confirmation_gate(
    require_approval: str = "never",
) -> Optional[ToolPreExecute]:
    """Return a ``tool_pre_execute`` callback or *None* if no gating is needed."""
    if require_approval == "never":
        return None

    async def _gate(
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        needs_confirmation = (
            require_approval == "always"
            or (require_approval == "critical" and tool_name in CRITICAL_TOOLS)
        )
        if needs_confirmation:
            return {
                "type": "confirmation_needed",
                "tool": tool_name,
                "arguments": arguments,
                "message": f"About to execute '{tool_name}'. Proceed?",
            }
        return None

    return _gate
