"""Chat service that manages interactive conversations with deployed or local models."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from shared.config import ChatConfig
from shared.multimodal_models import extract_text_from_content, normalize_content_blocks


def _resolve_hf_cache_path(model_path: str | None) -> str | None:
    if not model_path:
        return model_path
    path = Path(model_path)
    if not path.is_dir():
        return model_path
    snapshots = path / "snapshots"
    if not snapshots.is_dir() or (path / "config.json").exists():
        return model_path
    snapshot_dirs = [item for item in snapshots.iterdir() if item.is_dir()]
    if not snapshot_dirs:
        return model_path
    latest = max(snapshot_dirs, key=lambda item: item.stat().st_mtime)
    return str(latest)


class ChatSession:
    """Manages conversation state and generation for a single chat session."""

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

        self._http_client: Any = None
        self._provider: Any = provider
        self._owns_provider = provider is None
        self._inference_service = inference_service
        self._owns_inference_service = inference_service is None
        self._last_result: Optional[Dict[str, Any]] = None

        if config.system_prompt:
            self._history.append({"role": "system", "content": config.system_prompt})

    async def initialize(self) -> Dict[str, Any]:
        """Load model or validate endpoint. Returns session info."""
        if self._config.endpoint:
            return await self._init_api()
        if self._config.model_path:
            return await self._init_direct()
        raise ValueError("ChatConfig must have either 'endpoint' or 'model_path' set")

    async def _init_api(self) -> Dict[str, Any]:
        import httpx

        self._mode = "api"
        self._http_client = httpx.AsyncClient(timeout=30.0)

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
        resolved_model_path = _resolve_hf_cache_path(self._config.model_path)
        if self._config.modality == "vision-language":
            if self._inference_service is None:
                from finetuning_pipeline.services.inference_service import InferenceService

                self._inference_service = InferenceService()
        elif self._config.use_tokenizer_chat_template:
            if self._inference_service is None:
                from finetuning_pipeline.services.inference_service import InferenceService

                self._inference_service = InferenceService()
        elif self._provider is None:
            from agentsoul.providers.hf import HuggingFaceProvider

            self._provider = HuggingFaceProvider(
                model_path=resolved_model_path or self._config.model_path,
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

    async def send_message(self, user_input: str) -> str:
        result = await self.send_message_result(user_input)
        return result["response"]

    async def send_message_result(self, user_input: str) -> Dict[str, Any]:
        """Send a message and get the full response."""
        if self._config.modality == "vision-language":
            raise ValueError("send_message only supports text chats; use send_messages for VLM")

        self._history.append({"role": "user", "content": user_input})
        result = await (self._send_api(user_input) if self._mode == "api" else self._send_direct())
        self._finalize_text_result(result)
        return result

    async def send_messages(self, messages: List[Dict[str, Any]]) -> str:
        result = await self.send_messages_result(messages)
        return result["response"]

    async def send_messages_result(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Send structured multimodal messages and get the full response."""
        if self._config.modality != "vision-language":
            raise ValueError("send_messages is only available for VLM chat sessions")

        normalized_messages = [self._normalize_message(message) for message in messages]
        self._history.extend(normalized_messages)

        result = await (
            self._send_api_messages() if self._mode == "api" else self._send_direct_messages()
        )

        self._history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": result["response"]}],
            }
        )
        self._last_result = result
        return result

    async def stream_message(self, user_input: str) -> AsyncGenerator[str, None]:
        """Stream response tokens only."""
        async for event in self.stream_message_events(user_input):
            if event.get("type") == "token" and event.get("content"):
                yield str(event["content"])

    async def stream_message_events(
        self,
        user_input: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream structured text chat events."""
        if self._config.modality == "vision-language":
            raise ValueError("stream_message_events only supports text chats")

        self._history.append({"role": "user", "content": user_input})

        if self._mode == "api":
            async for event in self._stream_api_events(user_input):
                yield event
            return

        async for event in self._stream_direct_events():
            yield event

    async def _send_api(self, user_input: str) -> Dict[str, Any]:
        """POST to the deployed /generate endpoint."""
        prompt = self._format_history_as_prompt()
        start_time = time.perf_counter()
        resp = await self._http_client.post(
            f"{self._config.endpoint}{self._config.api_path or '/generate'}",
            params={
                "prompt": prompt,
                "max_new_tokens": self._config.max_new_tokens,
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k,
            },
        )
        data = resp.json()
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        metrics = data.get("metrics") if isinstance(data.get("metrics"), dict) else {}
        return {
            "response": data.get("response", ""),
            "usage": usage,
            "metrics": {
                **metrics,
                "latency_ms": metrics.get("latency_ms", latency_ms),
            },
            "model_id": data.get("model_id") or data.get("model") or self._config.model_path,
        }

    async def _send_direct(self) -> Dict[str, Any]:
        """Generate via the in-process HuggingFace provider."""
        if self._config.use_tokenizer_chat_template:
            return await self._generate_direct_text_result()

        resp = await self._provider.chat(
            messages=list(self._history),
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            do_sample=self._config.temperature > 0,
        )
        provider_metrics = getattr(self._provider, "last_metrics", None) or {}
        usage = self._normalize_usage(resp.usage, provider_metrics)
        return {
            "response": resp.content or "",
            "usage": usage,
            "metrics": self._build_text_metrics(provider_metrics, usage),
            "model_id": getattr(self._provider, "model_id", None) or self._config.model_path,
        }

    async def _generate_direct_text_result(self) -> Dict[str, Any]:
        """Generate text with the local tokenizer chat-template runtime."""
        result = await self._inference_service.run_text_messages_inference(
            messages=list(self._history),
            model_path=self._config.model_path,
            adapter_path=self._config.adapter_path,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            do_sample=self._config.temperature > 0,
            quantization=self._config.quantization or "4bit",
        )
        if not result.get("success"):
            raise RuntimeError(result.get("error", "Text generation failed"))

        usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
        return {
            "response": result.get("response", ""),
            "usage": usage,
            "metrics": {
                "latency_ms": self._to_ms(result.get("generation_time_seconds")),
                "ttft_ms": None,
                "output_tokens_per_second": self._to_float(result.get("tokens_per_second")),
                "perplexity": None,
                "confidence": None,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
            },
            "model_id": self._config.model_path,
        }

    async def _send_api_messages(self) -> Dict[str, Any]:
        """POST structured multimodal messages to a hosted VLM endpoint."""
        start_time = time.perf_counter()
        resp = await self._http_client.post(
            f"{self._config.endpoint}{self._config.api_path or '/generate_vlm'}",
            json={
                "messages": list(self._history),
                "max_new_tokens": self._config.max_new_tokens,
                "temperature": self._config.temperature,
                "top_p": self._config.top_p,
                "top_k": self._config.top_k,
            },
        )
        data = resp.json()
        latency_ms = round((time.perf_counter() - start_time) * 1000, 1)
        details = data.get("details", {}) if isinstance(data.get("details"), dict) else {}
        completion_tokens = details.get("tokens_generated")
        return {
            "response": data.get("response", ""),
            "usage": None,
            "metrics": {
                "latency_ms": latency_ms,
                "output_tokens_per_second": self._safe_rate(
                    completion_tokens,
                    details.get("generation_time_seconds"),
                ),
                "completion_tokens": completion_tokens,
            },
            "model_id": data.get("model") or self._config.model_path,
        }

    async def _send_direct_messages(self) -> Dict[str, Any]:
        """Generate a response using the in-process VLM inference service."""
        result = await self._inference_service.run_vlm_inference(
            messages=list(self._history),
            model_path=self._config.model_path,
            adapter_path=self._config.adapter_path,
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            do_sample=self._config.temperature > 0,
        )
        if not result.get("success"):
            raise RuntimeError(result.get("error", "VLM generation failed"))
        completion_tokens = result.get("tokens_generated")
        latency_ms = self._to_ms(result.get("generation_time_seconds"))
        return {
            "response": result.get("response", ""),
            "usage": None,
            "metrics": {
                "latency_ms": latency_ms,
                "output_tokens_per_second": self._safe_rate(
                    completion_tokens,
                    result.get("generation_time_seconds"),
                ),
                "completion_tokens": completion_tokens,
            },
            "model_id": self._config.model_path,
        }

    async def _stream_api_events(
        self,
        user_input: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        prompt = self._format_history_as_prompt()
        stream_path = self._stream_api_path()

        try:
            async with self._http_client.stream(
                "POST",
                f"{self._config.endpoint}{stream_path}",
                json={
                    "prompt": prompt,
                    "max_new_tokens": self._config.max_new_tokens,
                    "temperature": self._config.temperature,
                    "top_p": self._config.top_p,
                    "top_k": self._config.top_k,
                },
            ) as response:
                content_type = response.headers.get("content-type", "").lower()
                if response.status_code != 200 or "text/event-stream" not in content_type:
                    result = await self._send_api(user_input)
                    if result["response"]:
                        yield {"type": "token", "content": result["response"]}
                    self._finalize_text_result(result)
                    yield {"type": "complete", **result}
                    return

                full_response = ""
                async for event in self._iter_sse_events(response):
                    event_type = event.get("event") or "message"
                    payload = event.get("data")
                    if not isinstance(payload, dict):
                        continue

                    if event_type == "token":
                        content = payload.get("content")
                        if isinstance(content, str) and content:
                            full_response += content
                            yield {"type": "token", "content": content}
                        continue

                    if event_type == "complete":
                        result = {
                            "response": payload.get("response", full_response),
                            "usage": payload.get("usage"),
                            "metrics": payload.get("metrics"),
                            "model_id": payload.get("model_id") or payload.get("model"),
                        }
                        self._finalize_text_result(result)
                        yield {"type": "complete", **result}
                        return

                    if event_type == "error":
                        raise RuntimeError(str(payload.get("error", "Streaming request failed")))

                result = {
                    "response": full_response,
                    "usage": None,
                    "metrics": None,
                    "model_id": self._config.model_path,
                }
                self._finalize_text_result(result)
                yield {"type": "complete", **result}
        except Exception:
            result = await self._send_api(user_input)
            if result["response"]:
                yield {"type": "token", "content": result["response"]}
            self._finalize_text_result(result)
            yield {"type": "complete", **result}

    async def _stream_direct_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        if self._config.use_tokenizer_chat_template:
            result = await self._generate_direct_text_result()
            if result["response"]:
                yield {"type": "token", "content": result["response"]}
            self._finalize_text_result(result)
            yield {"type": "complete", **result}
            return

        full_response = ""
        streamed_usage: Optional[Dict[str, Any]] = None

        async for chunk in self._provider.stream(
            messages=list(self._history),
            max_new_tokens=self._config.max_new_tokens,
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            top_k=self._config.top_k,
            do_sample=self._config.temperature > 0,
        ):
            content = chunk if isinstance(chunk, str) else getattr(chunk, "content", None)
            if isinstance(content, str) and content:
                full_response += content
                yield {"type": "token", "content": content}

            chunk_usage = None if isinstance(chunk, str) else getattr(chunk, "usage", None)
            if isinstance(chunk_usage, dict):
                streamed_usage = chunk_usage

        provider_metrics = getattr(self._provider, "last_metrics", None) or {}
        usage = self._normalize_usage(streamed_usage, provider_metrics)
        result = {
            "response": full_response,
            "usage": usage,
            "metrics": self._build_text_metrics(provider_metrics, usage),
            "model_id": getattr(self._provider, "model_id", None) or self._config.model_path,
        }
        self._finalize_text_result(result)
        yield {"type": "complete", **result}

    def _finalize_text_result(self, result: Dict[str, Any]) -> None:
        self._history.append({"role": "assistant", "content": result["response"]})
        self._last_result = result

    def _build_text_metrics(
        self,
        provider_metrics: Dict[str, Any],
        usage: Dict[str, int],
    ) -> Dict[str, Any]:
        return {
            "latency_ms": self._to_ms(provider_metrics.get("time_total_s")),
            "ttft_ms": self._to_ms(provider_metrics.get("first_token_latency_s")),
            "output_tokens_per_second": self._to_float(
                provider_metrics.get("decode_tps") or provider_metrics.get("total_tps")
            ),
            "perplexity": self._to_float(provider_metrics.get("perplexity")),
            "confidence": provider_metrics.get("confidence_level"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    def _stream_api_path(self) -> str:
        api_path = (self._config.api_path or "/generate").rstrip("/")
        if api_path.endswith("_stream"):
            return api_path
        return f"{api_path}_stream"

    def _format_history_as_prompt(self) -> str:
        """Format conversation history as a flat prompt string for API mode."""
        parts: List[str] = []
        for msg in self._history:
            role = msg["role"].capitalize()
            content = extract_text_from_content(msg.get("content")) or msg.get("content", "")
            parts.append(f"{role}: {content}")
        return "\n".join(parts)

    @staticmethod
    def _normalize_message(message: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": message.get("role", "user"),
            "content": normalize_content_blocks(message.get("content")),
        }

    @staticmethod
    async def _iter_sse_events(response: Any) -> AsyncGenerator[Dict[str, Any], None]:
        event_type = "message"
        data_lines: List[str] = []

        async for raw_line in response.aiter_lines():
            line = raw_line.strip()
            if not line:
                if data_lines:
                    payload = "\n".join(data_lines)
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        data = {"raw": payload}
                    yield {"event": event_type, "data": data}
                event_type = "message"
                data_lines = []
                continue

            if line.startswith("event:"):
                event_type = line[6:].strip() or "message"
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())

        if data_lines:
            payload = "\n".join(data_lines)
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                data = {"raw": payload}
            yield {"event": event_type, "data": data}

    def clear_history(self) -> None:
        """Reset conversation history and preserve the system prompt."""
        if self._config.system_prompt:
            self._history = [{"role": "system", "content": self._config.system_prompt}]
        else:
            self._history = []
        self._last_result = None

    def update_generation_config(
        self,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Apply per-request generation overrides without resetting history."""
        if max_new_tokens is not None:
            self._config.max_new_tokens = max_new_tokens
        if temperature is not None:
            self._config.temperature = temperature
        if top_p is not None:
            self._config.top_p = top_p
        if top_k is not None:
            self._config.top_k = top_k

    def restore_history(self, messages: List[Dict[str, Any]]) -> None:
        """Restore a previously persisted conversation history."""
        if self._config.system_prompt:
            self._history = [{"role": "system", "content": self._config.system_prompt}]
        else:
            self._history = []

        for message in messages:
            role = message.get("role", "user")
            if role == "system":
                continue
            self._history.append(
                {
                    "role": role,
                    "content": message.get("content"),
                }
            )

    def get_info(self) -> Dict[str, Any]:
        """Return session metadata."""
        user_turns = sum(1 for message in self._history if message["role"] == "user")
        info: Dict[str, Any] = {
            "mode": self._mode,
            "initialized": self._initialized,
            "turns": user_turns,
            "max_new_tokens": self._config.max_new_tokens,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "top_k": self._config.top_k,
            "streaming": self._config.streaming,
            "modality": self._config.modality,
        }
        if self._mode == "api":
            info["endpoint"] = self._config.endpoint
        elif self._mode == "direct":
            info["model_path"] = self._config.model_path
            info["adapter_path"] = self._config.adapter_path
            info["shared_provider"] = not self._owns_provider
            info["use_tokenizer_chat_template"] = self._config.use_tokenizer_chat_template
            if self._config.quantization:
                info["quantization"] = self._config.quantization
        if self._config.system_prompt:
            info["system_prompt"] = self._config.system_prompt
        if self._last_result:
            info["last_metrics"] = self._last_result.get("metrics")
        return info

    @staticmethod
    def _to_ms(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return round(float(value) * 1000, 1)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_rate(tokens: Any, seconds: Any) -> Optional[float]:
        try:
            token_count = float(tokens)
            elapsed = float(seconds)
        except (TypeError, ValueError):
            return None
        if elapsed <= 0:
            return None
        return round(token_count / elapsed, 3)

    @staticmethod
    def _normalize_usage(
        usage: Optional[Dict[str, Any]],
        provider_metrics: Dict[str, Any],
    ) -> Dict[str, int]:
        normalized: Dict[str, int] = {}
        if isinstance(usage, dict):
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if value is None:
                    continue
                try:
                    normalized[key] = int(value)
                except (TypeError, ValueError):
                    continue

        if "prompt_tokens" not in normalized and provider_metrics.get("prompt_tokens") is not None:
            normalized["prompt_tokens"] = int(provider_metrics["prompt_tokens"])
        if "completion_tokens" not in normalized and provider_metrics.get("generated_tokens") is not None:
            normalized["completion_tokens"] = int(provider_metrics["generated_tokens"])
        if "total_tokens" not in normalized:
            if provider_metrics.get("total_tokens") is not None:
                normalized["total_tokens"] = int(provider_metrics["total_tokens"])
            else:
                normalized["total_tokens"] = (
                    normalized.get("prompt_tokens", 0) + normalized.get("completion_tokens", 0)
                )
        return normalized
