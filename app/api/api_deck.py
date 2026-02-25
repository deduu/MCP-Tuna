from fastapi import APIRouter, HTTPException, status, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import json
from ..utils.api.deck_api_generator import DeckApiGenerator


router = APIRouter()
orchestrator = DeckApiGenerator()


@router.post("/generate-deck")
async def generate_deck(form_data: Request):
    return await orchestrator.generate_deck(form_data)
