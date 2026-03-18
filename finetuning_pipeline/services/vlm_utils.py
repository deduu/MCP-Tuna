"""Utilities shared by VLM training and inference code paths."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from shared.multimodal_models import normalize_content_blocks


def resolve_dataset_base_dir(dataset_path: Optional[str]) -> Path:
    """Resolve the directory used for relative image paths."""
    if not dataset_path:
        return Path.cwd()
    return Path(dataset_path).resolve().parent


def _decode_data_url(data_url: str) -> bytes:
    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def load_image_from_block(block: Dict[str, Any], base_dir: Path):
    """Load a PIL image from a normalized content block."""
    from PIL import Image

    block_type = str(block.get("type", "")).strip()
    if block_type == "image_path":
        raw_path = str(block.get("image_path") or block.get("path") or "").strip()
        if not raw_path:
            raise ValueError("image_path block is missing a path")
        image_path = Path(raw_path)
        if not image_path.is_absolute():
            image_path = (base_dir / image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image path not found: {image_path}")
        return Image.open(image_path).convert("RGB")

    if block_type == "image_base64":
        encoded = block.get("image_base64")
        if not encoded:
            raise ValueError("image_base64 block is missing payload")
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

    if block_type == "image_url":
        image_url = block.get("image_url")
        if isinstance(image_url, dict):
            url = str(image_url.get("url", "")).strip()
        else:
            url = str(image_url or block.get("url") or "").strip()
        if url.startswith("data:"):
            return Image.open(io.BytesIO(_decode_data_url(url))).convert("RGB")
        raise ValueError("Only local paths and data URLs are supported for VLM execution")

    raise ValueError(f"Unsupported image block type: {block_type}")


def to_hf_messages_and_images(
    messages: Iterable[Dict[str, Any]],
    *,
    base_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Any]]:
    """Convert provider-agnostic messages into HF-compatible chat messages."""
    hf_messages: List[Dict[str, Any]] = []
    images: List[Any] = []

    for message in messages:
        role = str(message.get("role", "user")).strip() or "user"
        normalized_blocks = normalize_content_blocks(message.get("content"))
        if not normalized_blocks:
            continue

        hf_blocks: List[Dict[str, Any]] = []
        for block in normalized_blocks:
            block_type = block.get("type")
            if block_type == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    hf_blocks.append({"type": "text", "text": text})
            elif block_type in {"image_path", "image_url", "image_base64"}:
                images.append(load_image_from_block(block, base_dir))
                hf_blocks.append({"type": "image"})

        if hf_blocks:
            hf_messages.append({"role": role, "content": hf_blocks})

    return hf_messages, images


def build_vlm_prompt_and_images(
    messages: Iterable[Dict[str, Any]],
    *,
    processor: Any,
    base_dir: Path,
    add_generation_prompt: bool,
) -> Tuple[str, List[Any]]:
    """Render chat-template text and load companion images for a conversation."""
    hf_messages, images = to_hf_messages_and_images(messages, base_dir=base_dir)
    if not hf_messages:
        raise ValueError("Messages must include at least one text block")

    if hasattr(processor, "apply_chat_template"):
        prompt = processor.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    elif hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "apply_chat_template"):
        prompt = processor.tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        prompt = "\n".join(
            f"{message['role'].capitalize()}: "
            + " ".join(block.get("text", "[image]") for block in message["content"])
            for message in hf_messages
        )

    return prompt, images


def get_processor_tokenizer(processor: Any) -> Any:
    """Return the tokenizer-like object used for padding/token IDs."""
    return getattr(processor, "tokenizer", processor)
