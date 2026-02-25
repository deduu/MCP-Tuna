from __future__ import annotations

from typing import Optional, List, TYPE_CHECKING

from agentsoul.tools.service import ToolService
from agentsoul.tools.composite import CompositeToolService
from agentsoul.multi.agent_tool import AgentTool

if TYPE_CHECKING:
    from agentsoul.core.agent import AgentSoul
    from agentsoul.providers.base import BaseLLM
    from agentsoul.memory.base import BaseMemory


class AgentTeam:
    """Builder that constructs a supervisor AgentSoul whose tools are sub-agents.

    Routing is entirely LLM-based: the supervisor sees sub-agents as tools,
    reasons about which to call, delegates via tool calls, and synthesizes results.

    Usage:
        team = AgentTeam(
            name="supervisor",
            supervisor_llm=my_llm,
            supervisor_prompt="You are a project manager...",
        )
        team.add_agent(researcher, name="researcher", description="Does web research")
        team.add_agent(coder, name="coder", description="Writes code")
        supervisor = team.build()

        async for event in supervisor.run("Build me a web scraper"):
            ...
    """

    def __init__(
        self,
        name: str,
        supervisor_llm: "BaseLLM",
        supervisor_prompt: Optional[str] = None,
        max_turns: int = 15,
        supervisor_memory: Optional["BaseMemory"] = None,
    ):
        self.name = name
        self.supervisor_llm = supervisor_llm
        self.supervisor_prompt = supervisor_prompt
        self.max_turns = max_turns
        self.supervisor_memory = supervisor_memory

        self._agents: List[dict] = []
        self._extra_tool_services: List[ToolService] = []

    def add_agent(
        self,
        agent: "AgentSoul",
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "AgentTeam":
        """Add a specialist agent to the team. Chainable."""
        if name:
            agent.name = name
        if description:
            agent.description = description

        self._agents.append({
            "agent": agent,
            "name": agent.name,
            "description": agent.description or f"Agent: {agent.name}",
        })
        return self

    def add_tools(self, tool_service: ToolService) -> "AgentTeam":
        """Add non-agent tools that the supervisor can also use. Chainable."""
        self._extra_tool_services.append(tool_service)
        return self

    def build(self) -> "AgentSoul":
        """Construct the supervisor agent with sub-agents registered as tools."""
        from agentsoul.core.agent import AgentSoul

        # Create a ToolService for agent tools
        agent_tool_service = ToolService()
        for entry in self._agents:
            AgentTool.register_agent(entry["agent"], agent_tool_service)

        # Build final tool service
        if self._extra_tool_services:
            composite = CompositeToolService()
            composite.register_service("agents", agent_tool_service)
            for i, svc in enumerate(self._extra_tool_services):
                composite.register_service(f"tools_{i}", svc)
            final_tool_service = composite
        else:
            final_tool_service = agent_tool_service

        # Build system prompt with agent roster
        roster_lines = []
        for entry in self._agents:
            roster_lines.append(f"- **{entry['name']}**: {entry['description']}")
        roster = "\n".join(roster_lines)

        base_prompt = self.supervisor_prompt or (
            "You are a supervisor agent that coordinates specialist agents to complete tasks."
        )

        system_prompt = (
            f"{base_prompt}\n\n"
            f"## Available Specialist Agents\n"
            f"You can delegate tasks to these agents by calling them as tools:\n"
            f"{roster}\n\n"
            f"Break the user's request into sub-tasks, delegate each to the appropriate agent, "
            f"and synthesize their results into a final answer."
        )

        return AgentSoul(
            llm_provider=self.supervisor_llm,
            tool_service=final_tool_service,
            max_turns=self.max_turns,
            system_prompt=system_prompt,
            name=self.name,
            description=f"Supervisor agent coordinating: {', '.join(e['name'] for e in self._agents)}",
            memory=self.supervisor_memory,
        )
