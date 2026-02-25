from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.security import APIKeyHeader
from fastapi.security.api_key import APIKey
from ..utils.api.chat_api_orchestrator import ChatAPIOrchestrator
import openai
import os

router = APIRouter(tags=["Chat Completions"])

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")


def get_api_key():
    api_key = APIKeyHeader(name="X-API-Key", auto_error=False)
    return api_key


orchestrator = ChatAPIOrchestrator()


@router.post("/v1/chat/completions", response_model=dict, status_code=status.HTTP_200_OK)
async def chat_completions(request: Request):
    """
    OpenAI-compatible /v1/chat/completions endpoint with detailed logging.
    Compatible with OpenWebUI, supports streaming and non-streaming modes.
    """

    return await orchestrator.handle_request(request)
