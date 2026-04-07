from fastapi import APIRouter, Depends, Request
from ..core.auth import require_request_context
from ..utils.api.deck_api_generator import DeckApiGenerator


router = APIRouter()
orchestrator = DeckApiGenerator()


@router.post("/generate-deck")
async def generate_deck(
    form_data: Request,
    _ownership=Depends(require_request_context),
):
    return await orchestrator.generate_deck(form_data)
