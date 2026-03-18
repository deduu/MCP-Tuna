"""Shared multimodal message helpers used by VLM training and inference paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class ContentBlock:
    """Provider-agnostic content block."""

    type: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[Dict[str, Any] | str] = None
    image_base64: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class MultimodalMessage:
    """A chat message containing text and/or image blocks."""

    role: str
    content: List[ContentBlock] = field(default_factory=list)


@dataclass
class VLMDataPoint:
    """Supervised multimodal sample for VLM fine-tuning."""

    messages: List[MultimodalMessage] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = field(default=None)


def normalize_content_blocks(content: Any) -> List[Dict[str, Any]]:
    """Normalize flexible message content into a list of block dictionaries."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, (int, float, bool)):
        return [{"type": "text", "text": str(content)}]
    if not isinstance(content, list):
        return []

    blocks: List[Dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        block_type = str(item.get("type", "")).strip()
        if not block_type:
            continue
        block = {"type": block_type}
        for key in ("text", "image_path", "image_url", "image_base64", "mime_type", "path", "url"):
            if key in item:
                block[key] = item[key]
        if block_type == "image_path" and "image_path" not in block and "path" in block:
            block["image_path"] = block["path"]
        if block_type == "image_url" and "image_url" not in block and "url" in block:
            block["image_url"] = {"url": block["url"]}
        blocks.append(block)
    return blocks


def extract_text_from_content(content: Any) -> str:
    """Extract concatenated text content from mixed text/image blocks."""
    texts = []
    for block in normalize_content_blocks(content):
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text = block["text"].strip()
            if text:
                texts.append(text)
    return "\n".join(texts)


def content_has_image(content: Any) -> bool:
    """Return True when a message content payload includes any image block."""
    for block in normalize_content_blocks(content):
        if block.get("type") in {"image_path", "image_url", "image_base64"}:
            return True
    return False


def is_multimodal_message(message: Any) -> bool:
    """Return True when a message follows the multimodal message contract."""
    if not isinstance(message, dict):
        return False
    role = message.get("role")
    if not isinstance(role, str) or not role.strip():
        return False
    return bool(normalize_content_blocks(message.get("content")))


def is_vlm_sample(row: Any) -> bool:
    """Detect the canonical VLM SFT sample shape."""
    if not isinstance(row, dict):
        return False

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    if not all(is_multimodal_message(message) for message in messages):
        return False

    has_user_or_system_text = False
    has_assistant_text = False
    has_image = False

    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        text = extract_text_from_content(message.get("content"))
        if role in {"system", "user"} and text:
            has_user_or_system_text = True
        if role == "assistant" and text:
            has_assistant_text = True
        if content_has_image(message.get("content")):
            has_image = True

    return has_image and has_user_or_system_text and has_assistant_text


def count_image_blocks(messages: Iterable[Dict[str, Any]]) -> int:
    """Count image blocks across a sequence of messages."""
    total = 0
    for message in messages:
        total += sum(
            1
            for block in normalize_content_blocks(message.get("content"))
            if block.get("type") in {"image_path", "image_url", "image_base64"}
        )
    return total
