"""LoRA adapter merging and GGUF export operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from shared.async_utils import run_sync


class AdapterService:
    """Merge LoRA adapters into base models and export to GGUF format."""

    _SUPPORTED_QUANTIZATIONS = {"q4_0", "q4_k_m", "q5_k_m", "q8_0", "f16"}

    async def merge_adapter(
        self,
        base_model: str,
        adapter_path: str,
        output_path: str,
        push_to_hub: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Merge a LoRA adapter into the base model weights.

        Produces a standalone model directory at *output_path*.
        Optionally pushes the merged model to HuggingFace Hub.
        """
        return await run_sync(
            self._merge_adapter_sync,
            base_model,
            adapter_path,
            output_path,
            push_to_hub,
        )

    def _merge_adapter_sync(
        self,
        base_model: str,
        adapter_path: str,
        output_path: str,
        push_to_hub: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            import torch

            model = AutoModelForCausalLM.from_pretrained(
                base_model, torch_dtype=torch.float16, device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(base_model)

            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload()

            out = Path(output_path)
            out.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(out)
            tokenizer.save_pretrained(out)

            size_bytes = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())

            if push_to_hub:
                model.push_to_hub(push_to_hub)
                tokenizer.push_to_hub(push_to_hub)

            return {
                "success": True,
                "model_path": str(out.resolve()),
                "model_size_gb": round(size_bytes / (1024 ** 3), 2),
                "pushed_to_hub": push_to_hub,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def export_gguf(
        self,
        model_path: str,
        output_path: str,
        quantization: str = "q4_k_m",
    ) -> Dict[str, Any]:
        """Export a model to GGUF format for llama.cpp / Ollama.

        Requires ``llama-cpp-python`` (install via ``pip install mcp-tuna[export]``).
        """
        return await run_sync(
            self._export_gguf_sync, model_path, output_path, quantization,
        )

    def _export_gguf_sync(
        self,
        model_path: str,
        output_path: str,
        quantization: str = "q4_k_m",
    ) -> Dict[str, Any]:
        if quantization not in self._SUPPORTED_QUANTIZATIONS:
            return {
                "success": False,
                "error": (
                    f"Unsupported quantization: {quantization}. "
                    f"Use one of {sorted(self._SUPPORTED_QUANTIZATIONS)}"
                ),
            }

        try:
            from llama_cpp import llama_model_quantize  # noqa: F401
        except ImportError:
            return {
                "success": False,
                "error": (
                    "llama-cpp-python is not installed. "
                    "Install with: pip install mcp-tuna[export]"
                ),
            }

        try:
            from llama_cpp.llama import Llama  # noqa: F401
            import subprocess
            import sys

            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            # Use the convert script bundled with llama-cpp-python
            cmd = [
                sys.executable, "-m", "llama_cpp.convert",
                "--outfile", str(out),
                "--outtype", quantization,
                str(model_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if proc.returncode != 0:
                return {
                    "success": False,
                    "error": f"GGUF conversion failed: {proc.stderr[:500]}",
                }

            size_bytes = out.stat().st_size if out.exists() else 0
            return {
                "success": True,
                "gguf_path": str(out.resolve()),
                "quantization": quantization,
                "size_gb": round(size_bytes / (1024 ** 3), 2),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
