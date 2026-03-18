"""Chat service — manages interactive conversations with deployed or local models."""
from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional

from shared.config import ChatConfig
from shared.multimodal_models import extract_text_from_content, normalize_content_blocks


class ChatSession:
    """Manages conversation state and generation for a single chat session.

    Supports two modes:
    - **API mode** (``config.endpoint`` set): connects to a deployed model's
      ``/generate`` endpoint via httpx.
    - **Direct mode** (``config.model_path`` set): loads the model in-process
      using ``HuggingFaceProvider`` from agentsoul.
    """

    def __init__(
        self,
        config: ChatConfig,
        provider: Any = None,
        inference_service: Any = None,
    ) -> None:
        self._config = config
        self._history: List[Dict[str, Any]] = []
        self._mode: Optional[str] = None
        self._initialized: bool = False

        # API mode
        self._http_client: Any = None

        # Direct mode
        self._provider: Any = provider
        self._owns_provider = provider is None
        self._inference_service = inference_service
        self._owns_inference_service = inference_service is None

        # Seed system prompt into history
        if config.system_prompt:
            self._history.append({"role": "system", "content": config.system_prompt})

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def initialize(self) -> Dict[str, Any]:
        """Load model or validate endpoint. Returns session info dict."""
        if self._config.endpoint:
            return await self._init_api()
        if self._config.model_path:
            return await self._init_direct()
        raise ValueError("ChatConfig must have either 'endpoint' or 'model_path' set")

    async def _init_api(self) -> Dict[str, Any]:
        import httpx

        self._mode = "api"
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Validate the endpoint is reachable
        resp = await self._http_client.get(f"{self._config.endpoint}/health")
        health = resp.json() if resp.status_code == 200 else {}

        self._initialized = True
        return {
            "mode": "api",
            "endpoint": self._config.endpoint,
            "model": health.get("model", "unknown"),
        }

    async def _init_direct(self) -> Dict[str, Any]:
        self._mode = "direct"
        if self._config.modality == "vision-language":
            if self._inference_service is None:
                from finetuning_pipeline.services.inference_service import InferenceService

                self._inference_service = InferenceService()
        elif self._provider is None:
            from agentsoul.providers.hf import HuggingFaceProvider

            self._provider = HuggingFaceProvider(
                model_path=self._config.model_path,
                lora_adapter_path=self._config.adapter_path,
            )

        self._initialized = True
        return {
            "mode": "direct",
            "model_path": self._config.model_path,
            "adapter_path": self._config.adapter_path,
            "shared_provider": not self._owns_provider,
            "modality": self._config.modality,
        }

    async def shutdown(self) -> None:
        """Release resources: close HTTP client or unload model."""
        if self._mode == "api" and self._http_client is not None:
            await self._http_client.aclose()
        elif self._mode == "direct":
            if self._provider is not None and self._owns_provider:
                del self._provider
                self._provider = None
            if self._owns_inference_service:
                self._inference_service = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        self._initialized = False

    # ------------------------------------------------------------------ #
    # Messaging
    # ------------------------------------------------------------------ #

    async def send_message(self, user_input: str) -> str:
        """Send a message and get the full response. Updates conversation history."""
        if self._config.modality == "vision-language":
            raise ValueError("send_message only supports text chats; use send_messages for VLM")

        self._history.append({"role": "user", "content": user_input})

        if self._mode == "api":
            response = await self._send_api(user_input)
        else:
            response = await self._send_direct()

        self._history.append({"role": "assistant", "content": response})
        return response

    async def send_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Send structured multimodal messages and get the full response."""
        if self._config.modality != "vision-language":
            raise ValueError("send_messages is only available for VLM chat sessions")

        normalized_messages = [self._normalize_message(message) for message in messages]
        self._history.extend(normalized_messages)

        if self._mode == "api":
            response = await self._send_api_messages()
        else:
            response = await self._send_direct_messages()

        self._history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            }
        )
        return response

    async def stream_message(self, user_input: str) -> AsyncGenerator[str, None]:
        """Stream response tokens. Updates history on completion."""
        self._history.append({"role": "user", "content": user_input})

        if self._mode == "api":
            # Current /generate endpoint does not support SSE — fall back
            response = await self._send_api(user_input)
            self._history.append({"role": "assistant", "content": response})
            yield response
            return

        # Direct mode — stream from provider
        full_response = ""
        async for token in self._provider.stream(
            messages=list(self._history),
            max_new_tokens=self._config.max_new_tokens,
        ):
            full_response += token
            yield token

        self._history.append({"role": "assistant", "content": full_response})

    # ------------------------------------------------------------------ #
    # Internal send helpers
    # ------------------------------------------------------------------ #

    async def _send_api(self, user_input: str) -> str:
        """POST to the deployed /generate endpoint."""
        prompt = self._format_history_as_prompt()
        resp = await self._http_client.post(
            f"{self._config.endpoint}{self._config.api_path or '/generate'}",
            params={"prompt": prompt, "max_new_tokens": self._config.max_new_tokens},
        )
        data = resp.json()
        return data.get("response", "")

    async def _send_direct(self) -> str:
        """Generate via the in-process HuggingFaceProvider."""
        resp = await self._provider.chat(
            messages=list(self._history),
            max_new_tokens=self._config.max_new_tokens,
        )
        return resp.content or ""

    async def _send_api_messages(self) -> str:
        """POST structured multimodal messages to a hosted VLM endpoint."""
        resp = await self._http_client.post(
            f"{self._config.endpoint}{self._config.api_path or '/generate_vlm'}",
            json={
                "messages": list(self._history),
                "max_new_tokens": self._config.max_new_tokens,
                "temperature": self._config.temperature,
            },
        )
        data = resp.json()
        return data.get("response", "")

    async def _send_direct_messages(self) -> str:
        """Generate a response using the in-process VLM inference service."""
        result = await self._inference_service.run_vlm_inference(
            messages=list(self._history),
            model_path=self._config.model_path,
            adapter_path=self._config.adapter_path,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
        )
        if not result.get("success"):
            raise RuntimeError(result.get("error", "VLM generation failed"))
        return result.get("response", "")

    def _format_history_as_prompt(self) -> str:
        """Format conversation history as a flat prompt string for API mode."""
        parts: List[str] = []
        for msg in self._history:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {extract_text_from_content(msg.get('content')) or msg.get('content', '')}")
        return "\n".join(parts)

    @staticmethod
    def _normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": message.get("role", "user"),
            "content": normalize_content_blocks(message.get("content")),
        }

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def clear_history(self) -> None:
        """Reset conversation history (preserves system prompt if set)."""
        if self._config.system_prompt:
            self._history = [{"role": "system", "content": self._config.system_prompt}]
        else:
            self._history = []

    def get_info(self) -> Dict[str, Any]:
        """Return session metadata."""
        # Count turns: each user+assistant pair is one turn
        user_turns = sum(1 for m in self._history if m["role"] == "user")
        info: Dict[str, Any] = {
            "mode": self._mode,
            "initialized": self._initialized,
            "turns": user_turns,
            "max_new_tokens": self._config.max_new_tokens,
            "temperature": self._config.temperature,
            "streaming": self._config.streaming,
            "modality": self._config.modality,
        }
        if self._mode == "api":
            info["endpoint"] = self._config.endpoint
        elif self._mode == "direct":
            info["model_path"] = self._config.model_path
            info["adapter_path"] = self._config.adapter_path
            info["shared_provider"] = not self._owns_provider
        if self._config.system_prompt:
            info["system_prompt"] = self._config.system_prompt
        return info
