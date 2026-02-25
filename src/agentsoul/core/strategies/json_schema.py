from typing import List, Dict, Any, Optional
from agentsoul.core.tool_strategy import ToolCallingStrategy
from agentsoul.core.models import ToolCall


class JsonSchemaStrategy(ToolCallingStrategy):
    """
    Default strategy: tools are passed via the provider's native API parameter.
    Tool calls are extracted by the provider itself (OpenAI native tool_calls,
    HuggingFace <tool_call> XML parsing).
    """

    def format_tools_for_prompt(self, tool_descriptions: List[Dict[str, Any]]) -> str:
        return ""

    def should_pass_tools_to_api(self) -> bool:
        return True

    def parse_tool_calls(
        self, content: str, registered_tool_names: set
    ) -> Optional[List[ToolCall]]:
        # Provider handles parsing in JSON schema mode
        return None

    def get_system_prompt_addition(self) -> str:
        return ""
