from __future__ import annotations

import json

import pytest
from starlette.requests import Request

from app.utils.api.handlers.request_parser import RequestParser


def _make_request(payload: dict) -> Request:
    body = json.dumps(payload).encode("utf-8")
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            return {"type": "http.request", "body": b"", "more_body": False}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request({"type": "http", "method": "POST", "headers": []}, receive)


@pytest.mark.asyncio
async def test_parse_openai_chat_completion_preserves_multimodal_messages():
    request = _make_request(
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_path", "image_path": "uploads/images/example.png"},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
            ],
        }
    )

    parsed = await RequestParser.parse_openai_chat_completion(request)

    assert isinstance(parsed.messages[1]["content"], list)
    assert parsed.messages[1]["content"][0]["type"] == "image_path"
    assert parsed.user_prompt == "Describe this image."


@pytest.mark.asyncio
async def test_parse_openai_response_api_extracts_text_from_multimodal_input():
    request = _make_request(
        {
            "model": "gpt-4o",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_path", "image_path": "uploads/images/example.png"},
                        {"type": "text", "text": "What is in this image?"},
                    ],
                }
            ],
        }
    )

    parsed = await RequestParser.parse_openai_response_api(request)

    assert isinstance(parsed.messages[0]["content"], list)
    assert parsed.user_prompt == "What is in this image?"
