from __future__ import annotations

from typing import TYPE_CHECKING

from agentsoul.tools.service import ToolService

if TYPE_CHECKING:
    from agentsoul.core.agent import AgentSoul


class AgentTool:
    """Wraps a GenericAgent (AgentSoul) as a callable tool.

    The wrapped agent appears as a regular tool in the ToolService,
    with a single "task" parameter. When called, it delegates to the
    agent's invoke() method and returns the string result.
    """

    def __init__(self, agent: "AgentSoul"):
        if not agent.name:
            raise ValueError(
                "Agent must have a 'name' to be wrapped as a tool. "
                "Pass name= to AgentSoul()."
            )
        self.agent = agent
        self.tool_name = agent.name
        self.tool_description = agent.description or f"Delegate tasks to the {agent.name} agent."

    async def _execute(self, task: str) -> str:
        return await self.agent.invoke(task)

    def get_tool_description(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "The natural language task to delegate to this agent.",
                        }
                    },
                    "required": ["task"],
                },
            },
        }

    def register(self, tool_service: ToolService) -> None:
        """Register this agent-tool into an existing ToolService."""
        tool_service.register_tool(
            name=self.tool_name,
            func=self._execute,
            description=self.get_tool_description(),
        )

    @staticmethod
    def register_agent(agent: "AgentSoul", tool_service: ToolService) -> "AgentTool":
        """One-liner convenience: wrap agent and register into tool_service."""
        at = AgentTool(agent)
        at.register(tool_service)
        return at
