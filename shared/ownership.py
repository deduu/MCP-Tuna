from __future__ import annotations

import os
from contextvars import ContextVar, Token
from typing import Any, Mapping, Optional

from pydantic import BaseModel

from shared.workspace_paths import get_workspace_root


class OwnershipContext(BaseModel):
    """Minimal owner/workspace snapshot for alpha-era job and artifact records."""

    workspace_id: str
    user_id: Optional[str] = None


_ownership_context_var: ContextVar[OwnershipContext | None] = ContextVar(
    "ownership_context",
    default=None,
)


def default_ownership_context() -> OwnershipContext:
    workspace_id = str(os.getenv("MCP_TUNA_WORKSPACE_ID", "") or "").strip()
    if not workspace_id:
        workspace_id = get_workspace_root().name or "workspace"

    user_id = str(os.getenv("MCP_TUNA_USER_ID", "") or "").strip() or None
    return OwnershipContext(workspace_id=workspace_id, user_id=user_id)


def normalize_ownership_context(value: Any) -> OwnershipContext:
    """Normalize ownership input while preserving a safe local fallback."""

    if isinstance(value, OwnershipContext):
        return value

    if isinstance(value, Mapping):
        payload = {
            "workspace_id": str(value.get("workspace_id", "") or "").strip(),
            "user_id": str(value.get("user_id", "") or "").strip() or None,
        }
        if payload["workspace_id"]:
            return OwnershipContext(**payload)

    if isinstance(value, str) and value.strip():
        return OwnershipContext(workspace_id=value.strip())

    return default_ownership_context()


def get_current_ownership_context() -> OwnershipContext:
    return _ownership_context_var.get() or default_ownership_context()


def set_current_ownership_context(value: Any) -> Token:
    return _ownership_context_var.set(normalize_ownership_context(value))


def reset_current_ownership_context(token: Token) -> None:
    _ownership_context_var.reset(token)


def effective_ownership_context(value: Any = None) -> OwnershipContext:
    if value is not None:
        return normalize_ownership_context(value)
    return get_current_ownership_context()


def ownership_from_headers(headers: Mapping[str, Any]) -> OwnershipContext:
    default = default_ownership_context()
    workspace_id = str(headers.get("X-Workspace-Id", "") or "").strip()
    user_id = str(headers.get("X-User-Id", "") or "").strip() or None
    return normalize_ownership_context(
        {
            "workspace_id": workspace_id or default.workspace_id,
            "user_id": user_id if user_id is not None else default.user_id,
        }
    )
