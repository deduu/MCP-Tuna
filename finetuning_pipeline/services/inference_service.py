"""Inference and model comparison operations."""
from __future__ import annotations

import logging
import time
import torch
from pathlib import Path
from typing import Any, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from shared.gpu_lock import GPULock
from .gpu_service import GPUService
from .vlm_utils import build_vlm_prompt_and_images, get_processor_tokenizer

log = logging.getLogger(__name__)


class InferenceService:
    """Runs inference on local models (base or LoRA-adapted)."""

    def __init__(self, gpu: GPUService = None):
        self.gpu = gpu or GPUService()

    @staticmethod
    def _resolve_model_path(model_path: str) -> str:
        """Resolve HF cache wrapper dirs to a concrete snapshot path when needed."""
        path = Path(model_path)
        if not path.is_dir():
            return model_path

        snapshots_dir = path / "snapshots"
        if not snapshots_dir.is_dir() or (path / "config.json").exists():
            return model_path

        snapshot_dirs = [entry for entry in snapshots_dir.iterdir() if entry.is_dir()]
        if not snapshot_dirs:
            return model_path

        return str(max(snapshot_dirs, key=lambda entry: entry.stat().st_mtime))

    @staticmethod
    def _load_vlm_model_and_processor(model_path: str):
        from transformers import AutoProcessor

        resolved_model_path = InferenceService._resolve_model_path(model_path)
        processor = AutoProcessor.from_pretrained(resolved_model_path)

        errors: List[str] = []
        model = None
        for loader_name in ("AutoModelForVision2Seq", "AutoModelForImageTextToText"):
            try:
                transformers_mod = __import__("transformers", fromlist=[loader_name])
                loader = getattr(transformers_mod, loader_name)
                model = loader.from_pretrained(
                    resolved_model_path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                break
            except Exception as exc:
                errors.append(f"{loader_name}: {exc}")

        if model is None:
            raise RuntimeError(
                "Unable to load a VLM-compatible model class. "
                + " | ".join(errors[-2:])
            )

        tokenizer_like = get_processor_tokenizer(processor)
        if getattr(tokenizer_like, "pad_token", None) is None and getattr(tokenizer_like, "eos_token", None) is not None:
            tokenizer_like.pad_token = tokenizer_like.eos_token
        return model, processor

    @staticmethod
    def _load_text_tokenizer(model_path: str):
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        except Exception as exc:
            message = str(exc).lower()
            fallback_markers = (
                "backend tokenizer",
                "convert a slow tokenizer to a fast one",
                "sentencepiece",
                "tiktoken",
            )
            if not any(marker in message for marker in fallback_markers):
                raise

            log.warning(
                "Fast tokenizer unavailable for %s; falling back to slow tokenizer: %s",
                model_path,
                exc,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    @staticmethod
    def _text_inference_dtype() -> Any:
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32

    @classmethod
    def _build_text_quantization_config(cls, quantization: Optional[str]) -> Any:
        if not quantization or quantization == "none":
            return None

        try:
            from transformers import BitsAndBytesConfig
        except Exception:
            return None

        if quantization in {"4bit", "bitsandbytes"}:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=cls._text_inference_dtype(),
            )
        if quantization == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _load_text_model_and_tokenizer(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        quantization: Optional[str] = "4bit",
    ) -> tuple[Any, Any]:
        resolved_model_path = self._resolve_model_path(model_path)
        tokenizer = self._load_text_tokenizer(resolved_model_path)

        quantization_config = self._build_text_quantization_config(quantization)
        model = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            device_map="auto",
            quantization_config=quantization_config,
            max_memory=self.gpu.max_memory,
            torch_dtype=self._text_inference_dtype(),
        )
        if adapter_path:
            model = PeftModel.from_pretrained(model, adapter_path)
        if hasattr(model, "eval"):
            model.eval()
        return model, tokenizer

    async def run_text_messages_inference(
        self,
        messages: List[Dict[str, Any]],
        model_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        quantization: Optional[str] = "4bit",
    ) -> Dict[str, Any]:
        """Run a single text generation request from structured chat messages."""
        model = None
        tokenizer = None
        gpu_lock = GPULock.get()
        await gpu_lock.acquire("inference")
        try:
            model, tokenizer = self._load_text_model_and_tokenizer(
                model_path=model_path,
                adapter_path=adapter_path,
                quantization=quantization,
            )

            start_time = time.time()
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            prompt_len = model_inputs.input_ids.shape[-1]
            new_ids = generated_ids[0][prompt_len:]
            response_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            gen_time = time.time() - start_time
            completion_tokens = len(new_ids)
            prompt_tokens = int(prompt_len)
            total_tokens = prompt_tokens + completion_tokens

            return {
                "success": True,
                "response": response_text,
                "generation_time_seconds": gen_time,
                "tokens_generated": completion_tokens,
                "tokens_per_second": completion_tokens / gen_time if gen_time > 0 else 0,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "model_path": model_path,
                "adapter_path": adapter_path,
                "quantization": quantization,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "model_path": model_path}
        finally:
            del model, tokenizer
            self.gpu.clear_gpu_memory()
            gpu_lock.release()

    async def run_inference(
        self,
        prompts: List[str],
        model_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        system_prompt: Optional[str] = None,
        quantization: Optional[str] = "4bit",
    ) -> Dict[str, Any]:
        """Run inference on a list of prompts."""
        results = []
        for prompt_text in prompts:
            messages: List[Dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_text})
            result = await self.run_text_messages_inference(
                messages=messages,
                model_path=model_path,
                adapter_path=adapter_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                quantization=quantization,
            )
            if not result.get("success"):
                return result
            results.append({
                "prompt": prompt_text,
                "response": result["response"],
                "generation_time_seconds": result["generation_time_seconds"],
                "tokens_generated": result["tokens_generated"],
                "tokens_per_second": result["tokens_per_second"],
                "usage": result.get("usage"),
            })

        return {
            "success": True,
            "results": results,
            "model_path": model_path,
            "adapter_path": adapter_path,
            "num_prompts": len(prompts),
            "quantization": quantization,
        }

    async def compare_models(
        self,
        prompts: List[str],
        base_model_path: str,
        finetuned_adapter_path: str,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Compare base model vs fine-tuned model on the same prompts."""
        try:
            base_results = await self.run_inference(
                prompts=prompts,
                model_path=base_model_path,
                adapter_path=None,
                max_new_tokens=max_new_tokens,
            )
            if not base_results["success"]:
                return base_results

            finetuned_results = await self.run_inference(
                prompts=prompts,
                model_path=base_model_path,
                adapter_path=finetuned_adapter_path,
                max_new_tokens=max_new_tokens,
            )
            if not finetuned_results["success"]:
                return finetuned_results

            comparisons = []
            for base, ft in zip(base_results["results"], finetuned_results["results"]):
                comparisons.append({
                    "prompt": base["prompt"],
                    "base_response": base["response"],
                    "finetuned_response": ft["response"],
                    "base_time": base["generation_time_seconds"],
                    "finetuned_time": ft["generation_time_seconds"],
                    "base_tps": base["tokens_per_second"],
                    "finetuned_tps": ft["tokens_per_second"],
                })

            return {
                "success": True,
                "comparisons": comparisons,
                "base_model": base_model_path,
                "finetuned_adapter": finetuned_adapter_path,
                "num_prompts": len(prompts),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def run_vlm_inference(
        self,
        messages: List[Dict[str, Any]],
        model_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> Dict[str, Any]:
        """Run multimodal inference on a structured chat message payload."""
        model = None
        processor = None
        gpu_lock = GPULock.get()
        await gpu_lock.acquire("vlm_inference")
        try:
            resolved_model_path = self._resolve_model_path(model_path)
            model, processor = self._load_vlm_model_and_processor(resolved_model_path)

            if adapter_path:
                model = PeftModel.from_pretrained(model, adapter_path)

            prompt_text, images = build_vlm_prompt_and_images(
                messages,
                processor=processor,
                base_dir=Path.cwd(),
                add_generation_prompt=True,
            )
            processor_kwargs: Dict[str, Any] = {"text": prompt_text, "return_tensors": "pt"}
            if images:
                processor_kwargs["images"] = images if len(images) != 1 else images[0]

            start_time = time.time()
            model_inputs = processor(**processor_kwargs).to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    pad_token_id=getattr(get_processor_tokenizer(processor), "pad_token_id", None),
                )

            input_ids = model_inputs.get("input_ids")
            prompt_len = input_ids.shape[-1] if input_ids is not None else 0
            new_ids = generated_ids[0][prompt_len:]
            response_text = get_processor_tokenizer(processor).decode(new_ids, skip_special_tokens=True)
            gen_time = time.time() - start_time

            return {
                "success": True,
                "response": response_text,
                "model_path": model_path,
                "adapter_path": adapter_path,
                "image_count": len(images),
                "generation_time_seconds": gen_time,
                "tokens_generated": len(new_ids),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "model_path": model_path}
        finally:
            del model, processor
            self.gpu.clear_gpu_memory()
            gpu_lock.release()
