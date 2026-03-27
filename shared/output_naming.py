"""Helpers for compact, readable output names."""

from __future__ import annotations

import re


_DISALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")
_UUID_PREFIX_RE = re.compile(r"^[0-9a-fA-F-]{36}_")


def _to_base36(value: int) -> str:
    if value == 0:
        return "0"

    digits = []
    alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
    while value > 0:
        value, remainder = divmod(value, 36)
        digits.append(alphabet[remainder])
    return "".join(reversed(digits))


def _hash_token(value: str) -> str:
    hash_value = 2166136261
    for char in value:
        hash_value ^= ord(char)
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return _to_base36(hash_value)[:6]


def sanitize_name_segment(value: str, fallback: str = "item") -> str:
    sanitized = (
        value.strip()
        .replace("\\", "/")
        .split("/")[-1]
    )
    sanitized = re.sub(r"\.[^.]+$", "", sanitized)
    sanitized = _DISALLOWED_CHARS_RE.sub("_", sanitized).strip("_")
    return sanitized or fallback


def compact_name_segment(value: str, max_length: int = 24, fallback: str = "item") -> str:
    sanitized = sanitize_name_segment(value, fallback)
    if len(sanitized) <= max_length:
        return sanitized

    fingerprint = _hash_token(sanitized)
    head_length = max(8, max_length - len(fingerprint) - 1)
    return f"{sanitized[:head_length]}_{fingerprint}"


def compact_source_hint(value: str, max_length: int = 24, fallback: str = "dataset") -> str:
    without_uuid = _UUID_PREFIX_RE.sub("", sanitize_name_segment(value, fallback))
    return compact_name_segment(without_uuid, max_length, fallback)
