"""Tests for MCP tool descriptions and agent chaining awareness (Feature 4)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ----------------------------------------------------------------
# Tool description tests
# ----------------------------------------------------------------

TRAINING_TOOLS = [
    "finetune.train",
    "finetune.train_dpo",
    "finetune.train_grpo",
    "finetune.train_kto",
]

HOSTING_TOOLS = [
    "host.deploy_mcp",
    "host.deploy_api",
]


def _get_gateway_tool_descriptions() -> dict[str, str]:
    """Import the gateway and extract tool name → description mapping.

    The MCPServer stores registered tools internally; we inspect the
    gateway's registration methods by checking the decorated functions'
    docstrings / tool metadata.

    Simpler approach: import the module and read the source decorators.
    """
    import ast
    from pathlib import Path

    gateway_path = Path(__file__).resolve().parent.parent / "mcp_gateway.py"
    source = gateway_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    descriptions: dict[str, str] = {}

    class _ToolVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            # Look for @self.mcp.tool(name=..., description=...)
            for kw in node.keywords:
                if kw.arg == "name":
                    try:
                        name = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        self.generic_visit(node)
                        return
                    desc_kw = next(
                        (k for k in node.keywords if k.arg == "description"), None
                    )
                    if desc_kw is not None:
                        try:
                            desc = ast.literal_eval(desc_kw.value)
                        except (ValueError, TypeError):
                            desc = ""
                        descriptions[name] = desc
            self.generic_visit(node)

    _ToolVisitor().visit(tree)
    return descriptions


@pytest.fixture(scope="module")
def tool_descriptions() -> dict[str, str]:
    return _get_gateway_tool_descriptions()


@pytest.mark.parametrize("tool_name", TRAINING_TOOLS)
def test_training_tool_description_mentions_model_path(
    tool_descriptions, tool_name
):
    """Every training tool should mention model_path in its description."""
    desc = tool_descriptions.get(tool_name, "")
    assert "model_path" in desc.lower() or "model_path" in desc, (
        f"{tool_name} description does not mention model_path: {desc!r}"
    )


@pytest.mark.parametrize("tool_name", ["finetune.train_dpo", "finetune.train_grpo", "finetune.train_kto"])
def test_chaining_tools_mention_base_model_input(
    tool_descriptions, tool_name
):
    """DPO/GRPO/KTO should mention accepting base_model from previous training."""
    desc = tool_descriptions.get(tool_name, "")
    assert "base_model" in desc.lower() or "base_model" in desc, (
        f"{tool_name} description does not mention base_model input: {desc!r}"
    )


def test_sft_description_mentions_downstream_tools(tool_descriptions):
    """SFT tool should mention it can chain to DPO/GRPO/KTO."""
    desc = tool_descriptions.get("finetune.train", "")
    assert "train_dpo" in desc or "dpo" in desc.lower(), (
        f"finetune.train description does not mention DPO chaining: {desc!r}"
    )


@pytest.mark.parametrize("tool_name", HOSTING_TOOLS)
def test_hosting_tool_description_mentions_model_path_input(
    tool_descriptions, tool_name
):
    """Hosting tools should mention using model_path from training results."""
    desc = tool_descriptions.get(tool_name, "")
    assert "model_path" in desc.lower() or "finetune" in desc.lower(), (
        f"{tool_name} description does not reference finetune results: {desc!r}"
    )


# ----------------------------------------------------------------
# Agent factory system prompt tests
# ----------------------------------------------------------------


@pytest.mark.asyncio
@patch("app.core.agent_factory.get_registry")
async def test_agenty_system_prompt_used_when_gateway_connected(mock_registry):
    """When MCP servers include a MCP Tuna gateway, the system prompt should be set."""
    from app.core.agent_factory import TUNA_SYSTEM_PROMPT

    mock_provider = AsyncMock()
    mock_registry.return_value.get = AsyncMock(return_value=mock_provider)

    with patch("app.core.agent_factory.AgentSoul") as MockAgent:
        MockAgent.create = AsyncMock(return_value=MagicMock())

        from app.core.agent_factory import create_agent

        await create_agent(
            model_name="test-model",
            mcp_servers=[
                {
                    "server_label": "mcp-tuna-gateway",
                    "server_url": "http://localhost:8002",
                }
            ],
            system_prompt=None,
        )

        # AgentSoul.create should have been called with the MCP Tuna prompt
        MockAgent.create.assert_called_once()
        call_kwargs = MockAgent.create.call_args
        assert call_kwargs.kwargs.get("system_prompt") == TUNA_SYSTEM_PROMPT


@pytest.mark.asyncio
@patch("app.core.agent_factory.get_registry")
async def test_custom_system_prompt_overrides_default(mock_registry):
    """An explicit system_prompt should NOT be replaced by TUNA_SYSTEM_PROMPT."""
    mock_provider = AsyncMock()
    mock_registry.return_value.get = AsyncMock(return_value=mock_provider)

    with patch("app.core.agent_factory.AgentSoul") as MockAgent:
        MockAgent.create = AsyncMock(return_value=MagicMock())

        from app.core.agent_factory import create_agent

        custom = "You are a custom assistant."
        await create_agent(
            model_name="test-model",
            mcp_servers=[
                {
                    "server_label": "mcp-tuna-gateway",
                    "server_url": "http://localhost:8002",
                }
            ],
            system_prompt=custom,
        )

        MockAgent.create.assert_called_once()
        call_kwargs = MockAgent.create.call_args
        assert call_kwargs.kwargs.get("system_prompt") == custom


@pytest.mark.asyncio
@patch("app.core.agent_factory.get_registry")
async def test_no_agenty_prompt_when_unrelated_server(mock_registry):
    """When MCP servers don't include MCP Tuna gateway, default prompt should be None."""
    mock_provider = AsyncMock()
    mock_registry.return_value.get = AsyncMock(return_value=mock_provider)

    with patch("app.core.agent_factory.AgentSoul") as MockAgent:
        MockAgent.create = AsyncMock(return_value=MagicMock())

        from app.core.agent_factory import create_agent

        await create_agent(
            model_name="test-model",
            mcp_servers=[
                {
                    "server_label": "some-other-server",
                    "server_url": "http://localhost:9000",
                }
            ],
            system_prompt=None,
        )

        MockAgent.create.assert_called_once()
        call_kwargs = MockAgent.create.call_args
        assert call_kwargs.kwargs.get("system_prompt") is None
