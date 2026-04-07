from __future__ import annotations

import hmac
from collections.abc import AsyncIterator

from fastapi import HTTPException, Request, status

from app.core.config import settings
from shared.ownership import (
    OwnershipContext,
    ownership_from_headers,
    reset_current_ownership_context,
    set_current_ownership_context,
)


def _validate_api_key(request: Request) -> None:
    auth_settings = settings.auth
    if not auth_settings.enabled:
        return

    expected = auth_settings.api_key
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API key auth is enabled but no MCP_TUNA_API_KEY is configured.",
        )

    provided = request.headers.get(auth_settings.api_key_header_name)
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


async def require_request_context(request: Request) -> AsyncIterator[OwnershipContext]:
    """Validate optional alpha auth and attach normalized ownership context."""

    _validate_api_key(request)
    ownership = ownership_from_headers(request.headers)
    request.state.ownership = ownership
    token = set_current_ownership_context(ownership)
    try:
        yield ownership
    finally:
        reset_current_ownership_context(token)
