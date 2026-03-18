import logging
import json
from fastapi import Request
from ..models.request import ChatRequest
from shared.multimodal_models import extract_text_from_content

logger = logging.getLogger(__name__)


class RequestParser:

    @staticmethod
    def _extract_user_prompt(messages: list[dict]) -> str:
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            return extract_text_from_content(content)
        return ""

    @staticmethod
    async def _parse_common_request(body: dict) -> dict:
        """Parse common request data."""
        return {
            "stream": body.get("stream", False),
            "model_name": body.get("model", "Auto"),
            "query_mode": body.get("query_mode", "split_query"),
            "selected_tools": body.get("selected_tools", ["get_vector_context"]),
            "do_rerank": body.get("do_rerank", True),
            "api_key": body.get("api_key"),
            "base_url": body.get("base_url"),

            # ---- Generation parameters with defaults ----
            "temperature": float(body.get("temperature", 0.7)),
            "top_p": float(body.get("top_p", 0.95)),
            # None = not provided
            "top_k": body.get("top_k"),
            "max_tokens": body.get("max_tokens"),
            "presence_penalty": body.get("presence_penalty"),
            "frequency_penalty": body.get("frequency_penalty"),
            "repetition_penalty": body.get("repetition_penalty"),
        }

    @staticmethod
    async def parse_openai_chat_completion(request: Request) -> ChatRequest:
        """Parse OpenAI chat completion request."""
        body = await request.json()
        logger.info(f"Received OpenAI chat completion request: {body}")
        common = await RequestParser._parse_common_request(body)
        messages = body.get("messages", [])
        truncated_messages = messages[-10:] if len(messages) > 10 else messages
        user_prompt = RequestParser._extract_user_prompt(truncated_messages)

        return ChatRequest(
            messages=messages,
            truncated_messages=truncated_messages,
            user_prompt=user_prompt,
            request_type="chat_completion",
            **common
        )

      # ----------------------------
    # 2️⃣ /v1/responses
    # ----------------------------
    @staticmethod
    async def parse_openai_response_api(request: Request) -> ChatRequest:
        body = await request.json()
        logger.info(f"📥 Response API Request: {json.dumps(body)}")

        common = await RequestParser._parse_common_request(body)

        # Response API uses "input" instead of "messages"
        inputs = body.get("input", [])
        messages = []

        # Convert response-api format → Chat-like normalized structure
        for item in inputs:
            role = item.get("role", "user")
            content = item.get("content", [])
            if isinstance(content, list):
                messages.append({"role": role, "content": content})
            else:
                messages.append({"role": role, "content": content})

        truncated_messages = messages[-10:] if len(messages) > 10 else messages
        user_prompt = RequestParser._extract_user_prompt(truncated_messages)

        return ChatRequest(
            messages=messages,
            truncated_messages=truncated_messages,
            user_prompt=user_prompt,
            request_type="response_api",
            **common
        )
