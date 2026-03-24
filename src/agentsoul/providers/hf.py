# src/agentsoul/providers/hf.py
"""
HuggingFace local model provider for the agent framework.

Requires optional dependencies: torch, transformers, accelerate, bitsandbytes.
Install with: pip install agentsoul[local]
"""
from __future__ import annotations

import asyncio
import gc
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
)

from agentsoul.core.models import LLMResponse, StreamChunk, ToolCall
from agentsoul.providers.base import BaseLLM
from agentsoul.utils.logger import get_logger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports for torch / transformers so the framework can be imported
# on machines that don't have them installed.
# ---------------------------------------------------------------------------
def _import_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "torch is required for HuggingFaceProvider. "
            "Install with: pip install agentsoul[local]"
        )


def _import_transformers():
    try:
        import transformers
        return transformers
    except ImportError:
        raise ImportError(
            "transformers is required for HuggingFaceProvider. "
            "Install with: pip install agentsoul[local]"
        )


def _load_tokenizer_with_fallback(
    AutoTokenizer,
    model_path: str,
    logger: logging.Logger,
    **kwargs: Any,
):
    try:
        return AutoTokenizer.from_pretrained(model_path, use_fast=True, **kwargs)
    except Exception as exc:
        message = str(exc).lower()
        fallback_markers = (
            "backend tokenizer",
            "convert a slow tokenizer to a fast one",
            "sentencepiece",
            "tiktoken",
            "'nonetype' object has no attribute 'endswith'",
        )
        if not any(marker in message for marker in fallback_markers):
            raise

        logger.warning(
            "Fast tokenizer unavailable for %s; falling back to slow tokenizer: %s",
            model_path,
            exc,
        )
        return AutoTokenizer.from_pretrained(model_path, use_fast=False, **kwargs)


# ---------------------------------------------------------------------------
# Performance dataclass
# ---------------------------------------------------------------------------
@dataclass
class TokenPerf:
    run_id: str
    model_path: str
    device: str
    torch_dtype: str
    attn_impl: str
    quantization: Optional[str]
    gpu_name: Optional[str]
    torch_version: str
    transformers_version: str

    prompt_tokens: int
    generated_tokens: int
    total_tokens: int

    perplexity: float
    confidence_level: str

    time_total_s: float
    time_prefill_s: Optional[float]
    time_decode_s: Optional[float]

    prefill_tps: Optional[float]
    decode_tps: Optional[float]
    total_tps: float

    first_token_latency_s: Optional[float]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ---------------------------------------------------------------------------
# SDPA context helper
# ---------------------------------------------------------------------------
def _sdpa_context():
    """
    Prefer torch.nn.attention.sdpa_kernel() (PyTorch >= 2.4),
    fall back to torch.backends.cuda.sdp_kernel() on older versions,
    otherwise no-op.
    """
    try:
        from torch.nn.attention import sdpa_kernel
        return sdpa_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
    except Exception:
        try:
            from torch.backends.cuda import sdp_kernel
            return sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True)
        except Exception:
            return nullcontext()


# ===========================================================================
# HuggingFaceProvider
# ===========================================================================
class HuggingFaceProvider(BaseLLM):
    """
    Local HuggingFace text-generation provider.

    Supports:
    - 4-bit quantization via BitsAndBytes
    - Thinking/reasoning tokens (e.g. Qwen's <think> blocks)
    - Tool call parsing from <tool_call> XML blocks
    - Streaming via TextIteratorStreamer
    - Perplexity / confidence metrics
    - LoRA adapter loading via PEFT
    - Multi-GPU via device_map="auto"
    """

    supports_tool_calls_in_messages = False

    def __init__(
        self,
        model_path: str,
        lora_adapter_path: Optional[str] = None,
        device: str = "cuda:0",
        dtype: str = "float16",
        quantization: Optional[str] = None,
        attn_implementation: Optional[str] = None,
        verbose: bool = True,
        thinking_token_id: Optional[int] = 151668,
        device_map: Optional[Union[str, Dict]] = None,
        max_memory: Optional[Dict[int, str]] = None,
        offload_folder: Optional[str] = None,
        compute_perplexity: bool = False,
    ):
        """
        Args:
            model_path: HuggingFace model ID or local path.
            lora_adapter_path: Optional path to a LoRA adapter (via PEFT).
            device: Target device, e.g. "cuda:0" or "cpu".
            dtype: Torch dtype name: "float16", "bfloat16", "float32".
            quantization: "bitsandbytes" for 4-bit NF4 quantization, or None.
            attn_implementation: "flash_attention_2", "sdpa", or None (auto).
            verbose: Enable debug-level logging.
            thinking_token_id: Token ID that delimits thinking blocks (151668 for Qwen3).
            device_map: "auto", "balanced", or a custom dict. Overrides `device`.
            max_memory: Per-device memory limits, e.g. {0: "20GB", "cpu": "100GB"}.
            offload_folder: Folder for CPU/disk offloading when using device_map.
            compute_perplexity: Whether to compute perplexity after each generation (expensive). Default False.
        """
        torch = _import_torch()
        transformers = _import_transformers()
        from transformers import AutoTokenizer, AutoModelForCausalLM

        super().__init__(model_id=model_path)

        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__, verbose=verbose)
        self.quantization = quantization
        self.thinking_token_id = thinking_token_id
        self.last_metrics: Optional[Dict[str, Any]] = None
        self._compute_perplexity = compute_perplexity
        self._concurrency = asyncio.Semaphore(
            int(os.getenv("LOCAL_LLM_MAX_CONCURRENCY", "16"))
        )

        # Resolve device / device_map
        if device_map is not None:
            self.device = device  # primary device for tokenizer outputs
            effective_device_map = device_map
        else:
            self.device = device
            effective_device_map = {"": device}

        # Torch dtype
        self.torch_dtype = getattr(torch, dtype)

        # Quantization config
        quant_kwargs: Dict[str, Any] = {}
        if quantization == "bitsandbytes":
            from transformers import BitsAndBytesConfig
            quant_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=(
                    torch.bfloat16 if dtype.lower() in ("bf16", "bfloat16")
                    else torch.float16
                ),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # Attention implementation
        attn_impl = attn_implementation or "flash_attention_2"
        if attn_impl == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except Exception:
                attn_impl = "sdpa"
        self._attn_impl = attn_impl

        # Tokenizer
        self.tokenizer = _load_tokenizer_with_fallback(
            AutoTokenizer,
            model_path,
            self.logger,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
            "device_map": effective_device_map,
            "low_cpu_mem_usage": True,
            "attn_implementation": attn_impl,
            "trust_remote_code": True,
            **quant_kwargs,
        }
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
        if offload_folder is not None:
            load_kwargs["offload_folder"] = offload_folder

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, **load_kwargs
        )

        # LoRA adapter
        if lora_adapter_path is not None:
            from peft import PeftModel
            self.logger.info("Applying LoRA adapter from %s", lora_adapter_path)
            self.model = PeftModel.from_pretrained(
                self.model,
                lora_adapter_path,
                torch_dtype=self.torch_dtype,
                device_map=effective_device_map,
                trust_remote_code=True,
            )

        self.model.eval()

        # Perf-friendly defaults (set only when torch is loaded)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
            os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

        # Static metadata for logging
        self._gpu_name = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() and "cuda" in self.device
            else None
        )
        self._torch_version = torch.__version__
        self._tf_version = transformers.__version__

        self.logger.info(
            "Loaded %s | dtype=%s | attn=%s | quant=%s | device_map=%s",
            model_path, dtype, attn_impl, quantization, effective_device_map,
        )

    # ------------------------------------------------------------------
    # Message conversion helper
    # ------------------------------------------------------------------
    @staticmethod
    def _to_chat_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for msg in messages:
            if isinstance(msg, dict):
                out.append(msg)
            elif hasattr(msg, "to_dict"):
                out.append(msg.to_dict())
            else:
                out.append({
                    "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                    "content": msg.content,
                })
        return out

    # ------------------------------------------------------------------
    # chat()
    # ------------------------------------------------------------------
    async def chat(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False,
        **kwargs: Any,
    ) -> LLMResponse:
        torch = _import_torch()
        run_id = str(uuid.uuid4())
        chat_messages = self._to_chat_messages(messages)

        # Build prompt via chat template
        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            template_kwargs["tools"] = tools
        if enable_thinking:
            template_kwargs["enable_thinking"] = True

        prompt = self.tokenizer.apply_chat_template(chat_messages, **template_kwargs)

        # Tokenize
        model_inputs = self.tokenizer(
            [prompt], return_tensors="pt"
        ).to(self.device)
        prompt_tokens = model_inputs.input_ids.shape[1]

        gcfg = getattr(self.model, "generation_config", None)
        max_new = kwargs.get(
            "max_new_tokens",
            kwargs.get("max_tokens", 5120 if enable_thinking else 1024),
        )
        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 50),
            "pad_token_id": getattr(gcfg, "pad_token_id", self.tokenizer.pad_token_id),
            "eos_token_id": getattr(gcfg, "eos_token_id", self.tokenizer.eos_token_id),
            "use_cache": True,
        }

        async with self._concurrency:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with _sdpa_context(), torch.inference_mode():
                generated_ids = self.model.generate(**model_inputs, **gen_kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()

        # Decode only the newly generated tokens
        output_ids = generated_ids[0][prompt_tokens:].tolist()

        # Parse thinking tokens
        thinking = None
        if enable_thinking and self.thinking_token_id:
            try:
                index = len(output_ids) - output_ids[::-1].index(self.thinking_token_id)
                thinking = self.tokenizer.decode(
                    output_ids[:index], skip_special_tokens=True
                ).strip("\n")
                decoded = self.tokenizer.decode(
                    output_ids[index:], skip_special_tokens=True
                ).strip("\n")
            except ValueError:
                decoded = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip("\n")
        else:
            decoded = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip("\n")

        # Parse tool calls
        tool_calls = self._parse_tool_calls(decoded)

        # Metrics
        generated_tokens = len(output_ids)
        metrics = self._build_metrics(
            run_id=run_id,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            generated_texts=f"{thinking}\n\n{decoded}" if thinking else decoded,
            time_total=t1 - t0,
            time_prefill=None,
            time_decode=None,
        )
        self.last_metrics = metrics
        self._log_metrics(metrics)

        cb = kwargs.get("on_metrics")
        if callable(cb):
            cb(metrics)

        # Cleanup tensors
        del generated_ids, model_inputs

        return LLMResponse(
            content=decoded,
            tool_calls=tool_calls,
            thinking=thinking,
            perplexity=metrics.get("perplexity"),
            confidence_level=metrics.get("confidence_level"),
        )

    # ------------------------------------------------------------------
    # stream()
    # ------------------------------------------------------------------
    async def stream(
        self,
        messages: Iterable[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = False,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamChunk, None]:
        torch = _import_torch()
        from transformers import TextIteratorStreamer

        run_id = str(uuid.uuid4())
        chat_messages = self._to_chat_messages(messages)

        template_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            template_kwargs["tools"] = tools
        if enable_thinking:
            template_kwargs["enable_thinking"] = True

        prompt = self.tokenizer.apply_chat_template(chat_messages, **template_kwargs)

        model_inputs = self.tokenizer(
            [prompt], return_tensors="pt"
        ).to(self.device)
        prompt_tokens = model_inputs.input_ids.shape[1]

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        gcfg = getattr(self.model, "generation_config", None)
        max_new = kwargs.get(
            "max_new_tokens",
            kwargs.get("max_tokens", 5120 if enable_thinking else 2048),
        )
        gen_kwargs = {
            "max_new_tokens": max_new,
            "do_sample": kwargs.get("do_sample", True),
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.95),
            "top_k": kwargs.get("top_k", 50),
            "streamer": streamer,
            "pad_token_id": getattr(gcfg, "pad_token_id", self.tokenizer.pad_token_id),
            "eos_token_id": getattr(gcfg, "eos_token_id", self.tokenizer.eos_token_id),
            "use_cache": True,
        }

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()

        t_start: Optional[float] = None
        t_first: Optional[float] = None

        def _generate():
            nonlocal t_start
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            with _sdpa_context(), torch.inference_mode():
                self.model.generate(**model_inputs, **gen_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def _drain_streamer():
            nonlocal t_first
            for chunk in streamer:
                if t_first is None:
                    t_first = time.perf_counter()
                asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        collected: list[str] = []
        # Simple inline tool-call buffering: hold back text once we see <tool_call>
        tool_buffer: list[str] = []
        inside_tool_tag = False

        async with self._concurrency:
            threading.Thread(target=_generate, daemon=True).start()
            threading.Thread(target=_drain_streamer, daemon=True).start()

            while True:
                chunk = await queue.get()
                if chunk is None:
                    # End of generation
                    t_end = time.perf_counter()
                    generated_text = "".join(collected).strip()
                    generated_tokens = self._count_tokens(generated_text)

                    time_total = (t_end - t_start) if (t_end and t_start) else 0.0
                    time_prefill = (t_first - t_start) if (t_first and t_start) else None
                    time_decode = (t_end - t_first) if (t_end and t_first) else None

                    metrics = self._build_metrics(
                        run_id=run_id,
                        prompt_tokens=prompt_tokens,
                        generated_tokens=generated_tokens,
                        generated_texts=generated_text,
                        time_total=time_total,
                        time_prefill=time_prefill,
                        time_decode=time_decode,
                    )
                    self.last_metrics = metrics
                    self._log_metrics(metrics)
                    cb = kwargs.get("on_metrics")
                    if callable(cb):
                        cb(metrics)

                    del model_inputs

                    # Final tool call parse
                    tool_calls = self._parse_tool_calls(generated_text)
                    if tool_calls:
                        yield StreamChunk(
                            content=None,
                            tool_calls=tool_calls,
                            perplexity=metrics.get("perplexity"),
                            confidence_level=metrics.get("confidence_level"),
                            finish_reason="tool_calls",
                        )
                    else:
                        yield StreamChunk(
                            content=None,
                            tool_calls=None,
                            perplexity=metrics.get("perplexity"),
                            confidence_level=metrics.get("confidence_level"),
                            finish_reason="stop",
                        )
                    break

                collected.append(chunk)

                # Inline tool-call detection: buffer chunks inside <tool_call>...</tool_call>
                if inside_tool_tag:
                    tool_buffer.append(chunk)
                    if "</tool_call>" in "".join(tool_buffer):
                        inside_tool_tag = False
                        tool_buffer.clear()
                    continue

                if "<tool_call>" in chunk:
                    inside_tool_tag = True
                    # Yield any text before the tag
                    before = chunk.split("<tool_call>")[0]
                    if before:
                        yield StreamChunk(content=before, tool_calls=None, finish_reason=None)
                    tool_buffer.append(chunk)
                    continue

                yield StreamChunk(content=chunk, tool_calls=None, finish_reason=None)

    # ------------------------------------------------------------------
    # Tool call parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_tool_calls(response: str) -> Optional[List[ToolCall]]:
        """
        Parse <tool_call>{"name": "...", "arguments": {...}}</tool_call> blocks.
        """
        if not response or "<tool_call>" not in response:
            return None

        cleaned = response.strip()
        cleaned = re.sub(r"^```[\s\S]*?\n", "", cleaned)
        cleaned = re.sub(r"\n```$", "", cleaned)

        blocks = re.findall(
            r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>",
            cleaned,
            flags=re.IGNORECASE,
        )

        tool_calls: List[ToolCall] = []
        for block in blocks:
            try:
                data = json.loads(block)
                args = data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass

                name = data.get("name", "")
                if not isinstance(name, str) or not name:
                    continue

                tool_calls.append(ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=name,
                    arguments=args if isinstance(args, dict) else {"_raw": args},
                ))
            except json.JSONDecodeError:
                continue

        return tool_calls or None

    # ------------------------------------------------------------------
    # BaseLLM contract
    # ------------------------------------------------------------------
    def supports_tools(self) -> bool:
        return True

    def supports_thinking(self) -> bool:
        return self.thinking_token_id is not None

    def get_message_format_config(self) -> Dict[str, Any]:
        return {
            "supports_tool_calls_in_assistant": False,
            "tool_message_format": "name_content",
        }

    def get_specs(self) -> Dict[str, Any]:
        config = self.model.config
        num_heads = getattr(config, "num_attention_heads", None)
        num_kv = getattr(config, "num_key_value_heads", num_heads)
        head_dim = getattr(
            config, "head_dim",
            config.hidden_size // num_heads if num_heads else None,
        )
        return {
            "provider": "huggingface",
            "model_name": config._name_or_path,
            "architecture": config.__class__.__name__,
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "context_length": getattr(config, "max_position_embeddings", None),
            "num_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": num_heads,
            "num_key_value_heads": num_kv,
            "head_dim": head_dim,
        }

    # ------------------------------------------------------------------
    # Perplexity / confidence
    # ------------------------------------------------------------------
    def compute_perplexity(self, text: str) -> Dict[str, Any]:
        torch = _import_torch()
        encodings = self.tokenizer(text, return_tensors="pt")
        input_ids = encodings["input_ids"].to(self.model.device)
        attention_mask = encodings["attention_mask"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )

        score = float(math.exp(outputs.loss.item()))

        confidence_map = {
            (0, 1): "Extremely High Confidence",
            (1, 5): "Very High Confidence",
            (5, 15): "High Confidence",
            (15, 30): "Moderate Confidence",
            (30, float("inf")): "Low Confidence",
        }
        level = next(
            desc for (low, high), desc in confidence_map.items()
            if low <= score < high
        )
        return {"perplexity": score, "confidence_level": level}

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def _build_metrics(
        self,
        *,
        run_id: str,
        prompt_tokens: int,
        generated_tokens: int,
        generated_texts: str,
        time_total: float,
        time_prefill: Optional[float],
        time_decode: Optional[float],
    ) -> Dict[str, Any]:
        total_tokens = prompt_tokens + generated_tokens
        total_tps = (total_tokens / time_total) if time_total > 0 else 0.0
        prefill_tps = (
            prompt_tokens / time_prefill
        ) if (time_prefill and time_prefill > 0) else None
        decode_tps = (
            generated_tokens / time_decode
        ) if (time_decode and time_decode > 0) else None

        if self._compute_perplexity:
            perplexity_info = self.compute_perplexity(generated_texts)
        else:
            perplexity_info = {"perplexity": None, "confidence_level": None}

        perf = TokenPerf(
            run_id=run_id,
            model_path=self.model_id,
            device=self.device,
            torch_dtype=str(self.torch_dtype).replace("torch.", ""),
            attn_impl=self._attn_impl,
            quantization=self.quantization,
            gpu_name=self._gpu_name,
            torch_version=self._torch_version,
            transformers_version=self._tf_version,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            total_tokens=total_tokens,
            perplexity=perplexity_info["perplexity"],
            confidence_level=perplexity_info["confidence_level"],
            time_total_s=time_total,
            time_prefill_s=time_prefill,
            time_decode_s=time_decode,
            prefill_tps=prefill_tps,
            decode_tps=decode_tps,
            total_tps=total_tps,
            first_token_latency_s=time_prefill,
        )
        return asdict(perf)

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        logger.info("[llm_perf] %s", TokenPerf(**metrics).to_json())

    # ------------------------------------------------------------------
    # Memory info
    # ------------------------------------------------------------------
    def get_memory_usage(self) -> Dict[str, Any]:
        torch = _import_torch()
        if not torch.cuda.is_available():
            return {"type": "cpu"}

        device_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3

        return {
            "type": "cuda",
            "device_id": device_id,
            "allocated_gb": f"{allocated:.2f}",
            "reserved_gb": f"{reserved:.2f}",
            "total_gb": f"{total:.2f}",
            "utilization_pct": f"{(allocated / total) * 100:.1f}%",
        }

    # ------------------------------------------------------------------
    # Unload
    # ------------------------------------------------------------------
    def unload(self):
        """Release GPU memory held by this provider."""
        torch = _import_torch()
        try:
            if hasattr(self, "model") and self.model is not None:
                try:
                    self.model.to("cpu")
                except Exception:
                    pass
            for attr in ("model", "tokenizer"):
                try:
                    if hasattr(self, attr):
                        delattr(self, attr)
                except Exception:
                    pass
        finally:
            gc.collect()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
