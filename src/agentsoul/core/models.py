import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from enum import Enum


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    warnings: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role.value}

        # Handle assistant with tool calls
        if self.role == MessageRole.ASSISTANT and self.tool_calls:
            d["content"] = self.content or ""
            d["tool_calls"] = []
            for tc in self.tool_calls:
                # tc could be a dict or a ToolCall dataclass
                name = tc.get("name") if isinstance(tc, dict) else tc.name
                args = tc.get("arguments") if isinstance(
                    tc, dict) else tc.arguments
                if isinstance(args, dict):
                    args = json.dumps(args)
                d["tool_calls"].append({
                    "id": tc.get("id") if isinstance(tc, dict) else tc.id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                })
            return d

        # Handle tool response
        if self.role == MessageRole.TOOL:
            # For 'full' format: role, content, tool_call_id, name
            if self.tool_call_id:
                d["content"] = self.content
                d["tool_call_id"] = self.tool_call_id
                if self.name:
                    d["name"] = self.name
                if self.warnings:
                    d["warnings"] = self.warnings
            # For 'name_content' format: role, name, content
            else:
                if self.name:
                    d["name"] = self.name
                d["content"] = self.content
                if self.warnings:
                    d["warnings"] = self.warnings
            return d

        # Default case (system, user, plain assistant)
        d["content"] = self.content
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    function_call: Optional[Dict] = None
    thinking: Optional[str] = None
    perplexity: Optional[float] = None
    confidence_level: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


@dataclass
class StreamChunk:
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    perplexity: Optional[float] = None
    confidence_level: Optional[str] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


class ReflectionMode(str, Enum):
    NEVER = "never"
    BEFORE_FINAL = "before_final"
    EVERY_N_TURNS = "every_n_turns"
    ON_LOW_CONFIDENCE = "on_low_confidence"


@dataclass
class ReflectionPolicy:
    mode: ReflectionMode = ReflectionMode.NEVER
    every_n_turns: int = 3
    confidence_threshold: float = 0.7
    max_reflections: int = 2
    custom_prompt: Optional[str] = None
