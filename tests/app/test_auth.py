from __future__ import annotations

from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient

from app.core.auth import require_request_context
from app.core.config import settings
from shared.ownership import get_current_ownership_context


def _make_app() -> FastAPI:
    app = FastAPI()

    @app.get("/probe")
    async def probe(
        request: Request,
        _ownership=Depends(require_request_context),
    ):
        current = get_current_ownership_context()
        return {
            "workspace_id": request.state.ownership.workspace_id,
            "user_id": request.state.ownership.user_id,
            "current_workspace_id": current.workspace_id,
            "current_user_id": current.user_id,
        }

    return app


def test_request_context_uses_default_workspace_when_auth_disabled(monkeypatch):
    monkeypatch.setattr(settings.auth, "enabled", False)

    client = TestClient(_make_app())
    response = client.get("/probe")

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace_id"]
    assert payload["workspace_id"] == payload["current_workspace_id"]


def test_request_context_requires_api_key_when_enabled(monkeypatch):
    monkeypatch.setattr(settings.auth, "enabled", True)
    monkeypatch.setattr(settings.auth, "api_key", "secret-key")
    monkeypatch.setattr(settings.auth, "api_key_header_name", "X-API-Key")

    client = TestClient(_make_app())
    response = client.get("/probe")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key."


def test_request_context_accepts_valid_key_and_headers(monkeypatch):
    monkeypatch.setattr(settings.auth, "enabled", True)
    monkeypatch.setattr(settings.auth, "api_key", "secret-key")
    monkeypatch.setattr(settings.auth, "api_key_header_name", "X-API-Key")

    client = TestClient(_make_app())
    response = client.get(
        "/probe",
        headers={
            "X-API-Key": "secret-key",
            "X-Workspace-Id": "alpha-ws",
            "X-User-Id": "user-7",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workspace_id"] == "alpha-ws"
    assert payload["user_id"] == "user-7"
    assert payload["workspace_id"] == payload["current_workspace_id"]
