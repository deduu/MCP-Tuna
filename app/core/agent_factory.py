# app/core/agent_factory.py
from __future__ import annotations

from typing import Optional, List, Dict
from agentsoul.core.agent import AgentSoul
from app.generation.registry import get_registry
from app.core.config import settings
from app.core.confirmation_gate import build_confirmation_gate

TUNA_SYSTEM_PROMPT = """\
You are a helpful AI assistant with access to the MCP Tuna fine-tuning platform.

When working with fine-tuning workflows, follow these chaining patterns:
0. CHECK: Before training, call system.preflight_check to verify hardware can handle the model + config. Use system.check_resources to see current GPU/RAM/disk status.
1. PREPARE: Use generate.from_document -> clean.dataset -> normalize.dataset -> evaluate.dataset to build training data
2. TRAIN: Use finetune.train (SFT), finetune.train_dpo (DPO), finetune.train_grpo (GRPO), or finetune.train_kto (KTO)
3. CHAIN: Each training tool returns model_path — use it as base_model for subsequent training stages (e.g., SFT -> DPO -> GRPO)
4. EVALUATE: Use test.compare_models or evaluate_model.batch with the trained model_path
5. DEPLOY: Use host.deploy_mcp with the final model_path to serve the model

Key chaining rules:
- finetune.train* returns {model_path, success} — use model_path as the next stage's base_model
- finetune.sequential_train chains multiple techniques (SFT->DPO->GRPO) in a single call
- workflow.full_pipeline handles the entire pipeline end-to-end with optional deploy=True
- DPO requires prompt/chosen/rejected data; GRPO requires prompt/responses/rewards data
- Use generate.from_document(technique="dpo"/"grpo") to create technique-specific datasets
- workflow.run_pipeline executes multiple tools in one call — pass steps as JSON array with $prev.key refs to chain results (e.g., $prev.model_path)

CLARIFICATION RULES:
- If the user's request is ambiguous about which model to train, which dataset to use, or which technique to apply, ASK a clarification question before proceeding.
- Do NOT guess parameters for destructive operations (training, deployment, deletion) — always confirm with the user first.
- When multiple approaches are possible (e.g., SFT vs DPO, 1B vs 7B model), briefly present the trade-offs and ask the user to choose.
- After receiving clarification, proceed with the confirmed parameters.
- If the user does not specify a model, use system.auto_prescribe to recommend one based on their hardware and dataset.
"""


async def create_agent(
    model_name: str,
    mcp_servers: Optional[List[Dict]] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_turns: int = 10,
    system_prompt: Optional[str] = None,
) -> AgentSoul:
    """
    Create a AgentSoul instance, optionally wired to MCP tool servers.

    If mcp_servers is provided, those are used directly.
    Otherwise, auto-connects to servers defined in settings.mcp when auto_connect=True.
    """
    registry = get_registry()
    llm_provider = await registry.get(model_name, api_key=api_key, base_url=base_url)

    tools = _build_tool_configs(mcp_servers)

    # Use MCP Tuna-aware prompt when connected to the gateway and no custom prompt
    if system_prompt is None and tools:
        server_configs = mcp_servers or [
            {"server_label": s.server_label} for s in settings.mcp.servers
        ]
        labels = {s.get("server_label", "").lower() for s in server_configs}
        if any("mcp-tuna" in label or "tuna" in label for label in labels):
            system_prompt = TUNA_SYSTEM_PROMPT

    # Build confirmation gate from require_approval config
    approval_mode = _get_approval_mode(mcp_servers)
    gate = build_confirmation_gate(approval_mode)

    if tools:
        agent = await AgentSoul.create(
            llm_provider=llm_provider,
            tools=tools,
            system_prompt=system_prompt,
            max_turns=max_turns,
            tool_pre_execute=gate,
        )
    else:
        agent = AgentSoul(
            llm_provider=llm_provider,
            system_prompt=system_prompt,
            max_turns=max_turns,
        )

    return agent


def _build_tool_configs(
    mcp_servers: Optional[List[Dict]] = None,
) -> Optional[List[Dict]]:
    """Convert MCP server configs into tool config dicts for AgentSoul.create()."""
    tools = []

    # Use explicit MCP servers if provided
    if mcp_servers:
        for server in mcp_servers:
            tools.append({
                "type": "mcp",
                "server_label": server["server_label"],
                "server_url": server["server_url"],
                "server_description": server.get("server_description", ""),
                "require_approval": server.get("require_approval", "never"),
            })
        return tools

    # Auto-connect to configured servers
    if settings.mcp.auto_connect:
        for server_cfg in settings.mcp.servers:
            tools.append({
                "type": "mcp",
                "server_label": server_cfg.server_label,
                "server_url": server_cfg.server_url,
                "server_description": server_cfg.server_description,
                "require_approval": server_cfg.require_approval,
            })

    return tools if tools else None


def _get_approval_mode(mcp_servers: Optional[List[Dict]] = None) -> str:
    """Extract the most restrictive require_approval value from MCP server configs."""
    priority = {"always": 3, "critical": 2, "never": 1}

    if mcp_servers:
        configs = mcp_servers
    elif settings.mcp.auto_connect:
        configs = [
            {"require_approval": s.require_approval} for s in settings.mcp.servers
        ]
    else:
        return "never"

    modes = [c.get("require_approval", "never") for c in configs]
    return max(modes, key=lambda m: priority.get(m, 0), default="never")
