import json
from typing import List, Any, Optional

from agentsoul.providers.base import BaseLLM
from agentsoul.core.models import ToolCall, MessageRole, Message


class MessageFormatter:
    """Handles provider-specific message formatting"""

    def __init__(self, provider: BaseLLM):
        self.provider = provider

    def create_assistant_message(
        self,
        content: str,
        tool_calls: Optional[List[ToolCall]] = None
    ) -> Message:
        """Create an assistant message based on provider capabilities"""
        format_config = self.provider.get_message_format_config()

        if format_config.get('supports_tool_calls_in_assistant', True) and tool_calls:
            # OpenAI/Anthropic style: separate tool_calls field
            return Message(
                MessageRole.ASSISTANT,
                content=content or "",
                tool_calls=tool_calls
            )
        else:
            # HuggingFace style: tool calls are already serialized in content
            # The model returns: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
            if tool_calls:
                tool_calls_text = self._serialize_tool_calls_to_text(
                    tool_calls)
                content = f"{content}{tool_calls_text}"

            return Message(MessageRole.ASSISTANT, content or "")

    def create_tool_result_message(
        self,
        tool_call: ToolCall,
        result: Any,
        metadata: Optional[str] = None,
        warnings: str = None
    ) -> Message:
        """Create a tool result message based on provider capabilities"""
        format_config = self.provider.get_message_format_config()

        # --- Normalize result safely for all providers ---
        # Prefer JSON-safe string result first, use metadata only if result is empty
        if result is not None and result != "":
            content = result
        elif metadata is not None:
            if isinstance(metadata, (dict, list)):
                content = json.dumps(metadata, ensure_ascii=False)
            else:
                content = str(metadata)
        else:
            content = ""

        # ----------------------------------------------------

        tool_format = format_config.get('tool_message_format', 'full')

        if tool_format == 'full':
            # OpenAI-style: role, content, name, tool_call_id
            return Message(
                MessageRole.TOOL,
                content,
                name=tool_call.name,
                tool_call_id=tool_call.id,
                warnings=warnings
            )
        elif tool_format == 'name_content':
            # HuggingFace-style: role, name, content (no tool_call_id)
            return Message(
                MessageRole.TOOL,
                content,
                name=tool_call.name,
                warnings=warnings
            )
        elif tool_format == 'code_act':
            # Code-act style: show result as a comment with function call
            args_str = ", ".join(
                f"{k}={v!r}" for k, v in (tool_call.arguments or {}).items()
            )
            formatted_content = (
                f"# Result of {tool_call.name}({args_str})\n{content}"
            )
            return Message(MessageRole.USER, formatted_content)
        elif tool_format == 'user_content':
            # Some providers treat tool results as user messages
            formatted_content = f"Tool '{tool_call.name}' returned: {content}"
            return Message(MessageRole.USER, formatted_content)
        else:
            # Fallback to full format
            return Message(
                MessageRole.TOOL,
                content,
                name=tool_call.name,
                tool_call_id=tool_call.id,
                warnings=warnings
            )
