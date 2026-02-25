from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from agentsoul.core.models import ToolCall


class ToolCallingStrategy(ABC):
    """
    Abstracts how tools are presented to the LLM and how tool calls
    are extracted from the LLM's output.

    Implementations:
    - JsonSchemaStrategy: default, passes tools via provider's native API
    - CodeActStrategy: LLM generates Python code, parsed via AST
    - CodeActExecStrategy: LLM generates Python code, executed in sandbox
    """

    @abstractmethod
    def format_tools_for_prompt(
        self, tool_descriptions: List[Dict[str, Any]]
    ) -> str:
        """
        Convert tool descriptions into a string for the system prompt.
        Returns empty string if tools should be passed via API parameter.
        """
        ...

    @abstractmethod
    def should_pass_tools_to_api(self) -> bool:
        """
        Whether tool descriptions should be passed to the LLM provider's
        `tools=` parameter (True for JSON schema) or injected into the
        prompt (False for code-act).
        """
        ...

    @abstractmethod
    def parse_tool_calls(
        self, content: str, registered_tool_names: set
    ) -> Optional[List[ToolCall]]:
        """
        Parse tool calls from the LLM's text output.
        Returns None if no tool calls were found.
        """
        ...

    def get_system_prompt_addition(self) -> str:
        """Extra system prompt text for tool calling instructions."""
        return ""
