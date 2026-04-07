from fastapi import APIRouter, Depends, Request, status
from ..utils.api.chat_api_orchestrator import ChatAPIOrchestrator
from ..core.auth import require_request_context
import openai
import os

router = APIRouter(tags=["Chat Completions"])

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_version = os.getenv("OPENAI_API_VERSION")


orchestrator = ChatAPIOrchestrator()


@router.post("/v1/chat/completions", response_model=dict, status_code=status.HTTP_200_OK)
async def chat_completions(
    request: Request,
    _ownership=Depends(require_request_context),
):
    """
    OpenAI-compatible /v1/chat/completions endpoint with detailed logging.
    Compatible with OpenWebUI, supports streaming and non-streaming modes.
    """

    return await orchestrator.handle_request(request)
