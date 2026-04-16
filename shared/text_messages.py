"""Helpers for text-only chat messages used by SFT and agent traces."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from shared.multimodal_models import content_has_image, extract_text_from_content


_VALID_MESSAGE_ROLES = {"system", "user", "assistant", "tool"}


def normalize_text_message_content(content: Any) -> str:
    """Return a plain-text representation for text-only message content."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (int, float, bool)):
        return str(content)
    if isinstance(content, list):
        return extract_text_from_content(content).strip()
    if isinstance(content, Mapping):
        return json.dumps(dict(content), ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        return json.dumps(list(content), ensure_ascii=False, default=str)
    return str(content).strip()


def is_text_message(message: Any) -> bool:
    """Return True when a message follows the text-chat contract."""
    if not isinstance(message, Mapping):
        return False

    role = str(message.get("role") or "").strip().lower()
    if role not in _VALID_MESSAGE_ROLES:
        return False

    content = message.get("content")
    if content_has_image(content):
        return False

    if role == "assistant" and _extract_tool_calls(message):
        return True

    return bool(normalize_text_message_content(content))


def is_text_messages_sample(row: Any) -> bool:
    """Detect a text-only structured messages SFT row."""
    if not isinstance(row, Mapping):
        return False

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    if not all(is_text_message(message) for message in messages):
        return False

    assistant_turns = 0
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        text = normalize_text_message_content(message.get("content"))
        if text or _extract_tool_calls(message):
            assistant_turns += 1

    return assistant_turns > 0


def serialize_tool_calls(tool_calls: Any) -> str:
    """Serialize OpenAI-style tool calls to inline XML blocks."""
    serialized: list[str] = []
    for tool_call in _extract_tool_calls(tool_calls):
        name = str(tool_call.get("name") or "").strip()
        if not name:
            continue
        arguments = tool_call.get("arguments", {})
        if not isinstance(arguments, Mapping):
            arguments = {"_raw": arguments}
        payload = {"name": name, "arguments": dict(arguments)}
        serialized.append(
            f"<tool_call>{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}</tool_call>"
        )
    return "\n".join(serialized)


def flatten_messages_for_training(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Normalize rich text messages to chat-template-safe role/content pairs."""
    normalized: list[dict[str, str]] = []

    for raw_message in messages:
        role = str(raw_message.get("role") or "").strip().lower()
        if role not in _VALID_MESSAGE_ROLES:
            continue

        content = normalize_text_message_content(raw_message.get("content"))
        if role == "assistant":
            tool_call_text = serialize_tool_calls(raw_message.get("tool_calls"))
            content = "\n".join(part for part in (content, tool_call_text) if part).strip()
            if content:
                normalized.append({"role": "assistant", "content": content})
            continue

        if role == "tool":
            prefix = f"Tool '{str(raw_message.get('name') or 'tool').strip() or 'tool'}' returned"
            tool_call_id = str(raw_message.get("tool_call_id") or "").strip()
            if tool_call_id:
                prefix = f"{prefix} (tool_call_id={tool_call_id})"
            if not content:
                content = "No tool result content provided."
            normalized.append({"role": "user", "content": f"{prefix}: {content}"})
            continue

        if content:
            normalized.append({"role": role, "content": content})

    return normalized


def messages_prompt_completion(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str, str, list[dict[str, str]]]:
    """Split structured messages into prompt/completion text plus normalized messages."""
    normalized = flatten_messages_for_training(messages)
    last_assistant_index = -1
    for index in range(len(normalized) - 1, -1, -1):
        if normalized[index]["role"] == "assistant":
            last_assistant_index = index
            break

    if last_assistant_index < 0:
        transcript = render_text_message_transcript(normalized)
        return transcript, "", normalized

    prompt_messages = normalized[:last_assistant_index]
    completion = normalized[last_assistant_index]["content"]
    prompt = render_text_message_transcript(prompt_messages)
    return prompt, completion, normalized[: last_assistant_index + 1]


def render_text_message_transcript(messages: Sequence[Mapping[str, Any]]) -> str:
    """Render normalized text messages to a readable prompt transcript."""
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if not role or not content:
            continue
        lines.append(f"{role.capitalize()}: {content}")
    return "\n".join(lines).strip()


def count_tool_calls(messages: Sequence[Mapping[str, Any]]) -> int:
    """Count assistant tool calls across a message sequence."""
    total = 0
    for message in messages:
        if str(message.get("role") or "").strip().lower() != "assistant":
            continue
        total += len(_extract_tool_calls(message))
    return total


def _extract_tool_calls(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        value = value.get("tool_calls")

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            continue
        function = item.get("function")
        if isinstance(function, Mapping):
            name = str(function.get("name") or item.get("name") or "").strip()
            arguments = function.get("arguments", {})
        else:
            name = str(item.get("name") or "").strip()
            arguments = item.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments}
        if not isinstance(arguments, Mapping):
            arguments = {"_raw": arguments}
        if not name:
            continue
        normalized.append(
            {
                "id": str(item.get("id") or f"call_{index}"),
                "name": name,
                "arguments": dict(arguments),
            }
        )
    return normalized
