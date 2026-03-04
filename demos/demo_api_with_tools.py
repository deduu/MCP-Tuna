"""
Demo: API Request with MCP-Wired Agent
=======================================

Shows that regular OpenAI-compatible API requests automatically get
tool-equipped agents when the MCP Tuna gateway is running.

Prerequisites:
    - MCP Tuna Gateway running: python scripts/run_gateway.py http 8000
    - App server running: uvicorn app.api.api_chat:app --port 8080
    - OPENAI_API_KEY set in environment

Usage:
    python demos/demo_api_with_tools.py
    python demos/demo_api_with_tools.py --stream
    python demos/demo_api_with_tools.py "Your custom prompt here"
"""

import asyncio
import json
import sys
import os

APP_URL = os.getenv("APP_URL", "http://localhost:8080")
DEFAULT_PROMPT = "What fine-tuning techniques are available? List them."


async def run_non_streaming(prompt: str):
    """Send a non-streaming request to the API."""
    import httpx

    print(f"Sending non-streaming request to {APP_URL}/v1/chat/completions...")
    print(f"Prompt: {prompt}\n")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{APP_URL}/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120.0,
        )

    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} - {resp.text}")
        return

    data = resp.json()
    print("Response:")
    print("-" * 40)
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    print(content)
    print("-" * 40)
    usage = data.get("usage", {})
    if usage:
        print(f"Tokens: prompt={usage.get('prompt_tokens', '?')}, "
              f"completion={usage.get('completion_tokens', '?')}")


async def run_streaming(prompt: str):
    """Send a streaming request to the API."""
    import httpx

    print(f"Sending streaming request to {APP_URL}/v1/chat/completions...")
    print(f"Prompt: {prompt}\n")
    print("-" * 40)

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{APP_URL}/v1/chat/completions",
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=120.0,
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        print(content, end="", flush=True)
                except json.JSONDecodeError:
                    pass

    print("\n" + "-" * 40)
    print("Stream complete.")


async def main():
    stream = "--stream" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--stream"]
    prompt = args[0] if args else DEFAULT_PROMPT

    if stream:
        await run_streaming(prompt)
    else:
        await run_non_streaming(prompt)


if __name__ == "__main__":
    asyncio.run(main())
