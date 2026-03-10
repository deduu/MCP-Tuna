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

    async def run_inference(
        self,
        prompts: List[str],
        model_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> Dict[str, Any]:
        """Run inference on a list of prompts."""
        model = None
        tokenizer = None
        gpu_lock = GPULock.get()
        await gpu_lock.acquire("inference")
        try:
            resolved_model_path = self._resolve_model_path(model_path)
            try:
                tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, use_fast=True)
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
                    resolved_model_path,
                    exc,
                )
                tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(
                resolved_model_path,
                device_map="auto",
                quantization_config=self.gpu.bnb_config,
                max_memory=self.gpu.max_memory,
                torch_dtype=torch.bfloat16,
            )

            if adapter_path:
                model = PeftModel.from_pretrained(model, adapter_path)

            results = []
            for prompt_text in prompts:
                start_time = time.time()

                messages = [{"role": "user", "content": prompt_text}]
                input_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                prompt_len = model_inputs.input_ids.shape[-1]
                new_ids = generated_ids[0][prompt_len:]
                response_text = tokenizer.decode(new_ids, skip_special_tokens=True)

                gen_time = time.time() - start_time
                tokens_generated = len(new_ids)
                tps = tokens_generated / gen_time if gen_time > 0 else 0

                results.append({
                    "prompt": prompt_text,
                    "response": response_text,
                    "generation_time_seconds": gen_time,
                    "tokens_generated": tokens_generated,
                    "tokens_per_second": tps,
                })

            return {
                "success": True,
                "results": results,
                "model_path": model_path,
                "adapter_path": adapter_path,
                "num_prompts": len(prompts),
            }
        except Exception as e:
            return {"success": False, "error": str(e), "model_path": model_path}
        finally:
            del model, tokenizer
            self.gpu.clear_gpu_memory()
            gpu_lock.release()

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
