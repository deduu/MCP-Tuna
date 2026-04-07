from __future__ import annotations

from shared.ownership import (
    default_ownership_context,
    get_current_ownership_context,
    normalize_ownership_context,
    ownership_from_headers,
    reset_current_ownership_context,
    set_current_ownership_context,
)


def test_normalize_ownership_context_prefers_explicit_mapping():
    ownership = normalize_ownership_context(
        {"workspace_id": "alpha-ws", "user_id": "user-1"}
    )

    assert ownership.workspace_id == "alpha-ws"
    assert ownership.user_id == "user-1"


def test_normalize_ownership_context_accepts_workspace_string():
    ownership = normalize_ownership_context("workspace-42")

    assert ownership.workspace_id == "workspace-42"
    assert ownership.user_id is None


def test_default_ownership_context_returns_workspace_id():
    ownership = default_ownership_context()

    assert ownership.workspace_id


def test_set_current_ownership_context_updates_contextvar():
    token = set_current_ownership_context({"workspace_id": "ctx-ws", "user_id": "user-1"})
    try:
        ownership = get_current_ownership_context()
        assert ownership.workspace_id == "ctx-ws"
        assert ownership.user_id == "user-1"
    finally:
        reset_current_ownership_context(token)


def test_ownership_from_headers_prefers_request_headers():
    ownership = ownership_from_headers(
        {
            "X-Workspace-Id": "header-ws",
            "X-User-Id": "header-user",
        }
    )

    assert ownership.workspace_id == "header-ws"
    assert ownership.user_id == "header-user"
