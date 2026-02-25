from typing import Dict, Any, Optional
import time
import traceback
from fastapi.responses import JSONResponse


class ResponseBuilder:
    """Builds OpenAI-compatible response objects."""

    @staticmethod
    def build_openai_chat_completion(
        content: str,
        model_name: str,
        usage: Optional[Dict[str, int]] = None
    ) -> Dict[str, Any]:
        """Build a non-streaming chat completion response."""
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

    @staticmethod
    def build_openai_chat_chunk(
        content: str,
        model_name: str,
        finish_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build a streaming chunk response."""
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "delta": {
                    "role": "assistant",
                    "content": content
                },
                "index": 0,
                "finish_reason": finish_reason,
            }],
        }

    @staticmethod
    def build_openai_error_response(error: Exception, status_code: int = 500) -> JSONResponse:
        """Build an error response."""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": str(error),
                "traceback": traceback.format_exc(),
                "hint": "Check logs for detailed trace.",
            },
        )
