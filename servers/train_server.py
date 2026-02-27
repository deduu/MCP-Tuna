"""Training MCP server.

Combines: system, finetune, test, validate tools.
Requires GPU and PyTorch.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional

from agentsoul.server import MCPServer


class TrainServer:
    """All training-related tools in a single MCP server."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._finetuning_svc = None
        self._job_manager_instance = None
        self.mcp = MCPServer("transcendence-train", "1.0.0")
        self._register_tools()

    @property
    def finetuner(self):
        if self._finetuning_svc is None:
            from finetuning_pipeline.services.pipeline_service import FineTuningService
            self._finetuning_svc = FineTuningService(
                default_base_model=self._config.get("finetuning", {}).get(
                    "base_model", "meta-llama/Llama-3.2-3B-Instruct"
                )
            )
        return self._finetuning_svc

    @property
    def job_manager(self):
        if self._job_manager_instance is None:
            from shared.training_jobs import TrainingJobManager
            self._job_manager_instance = TrainingJobManager(max_concurrent=1)
        return self._job_manager_instance

    def _register_tools(self):
        self._register_system_tools()
        self._register_finetune_tools()
        self._register_test_tools()
        self._register_validate_tools()

    def _register_system_tools(self):
        @self.mcp.tool(name="system.check_resources",
                       description="Check GPU/RAM/disk status before training")
        async def check_resources() -> str:
            return json.dumps(self.finetuner.check_resources(), indent=2)

        @self.mcp.tool(name="system.preflight_check",
                       description="Estimate VRAM requirements for training config")
        async def preflight_check(
            model_name: str, quantization: str = "4bit", batch_size: int = 1,
            max_seq_length: int = 512, technique: str = "sft",
            use_lora: bool = True, lora_r: int = 8,
            gradient_checkpointing: bool = False,
        ) -> str:
            return json.dumps(self.finetuner.preflight_check(
                model_name=model_name, quantization=quantization,
                batch_size=batch_size, max_seq_length=max_seq_length,
                technique=technique, use_lora=use_lora, lora_r=lora_r,
                gradient_checkpointing=gradient_checkpointing,
            ), indent=2)

        @self.mcp.tool(name="system.setup_check",
                       description="Validate all prerequisites for AgentY")
        async def setup_check() -> str:
            checks = []
            has_key = bool(os.getenv("OPENAI_API_KEY"))
            checks.append({"name": "OPENAI_API_KEY", "status": "pass" if has_key else "warn",
                           "detail": "Set" if has_key else "Not set"})
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            checks.append({"name": "HF_TOKEN", "status": "pass" if hf_token else "warn",
                           "detail": "Set" if hf_token else "Not set"})
            try:
                import torch
                gpu_ok = torch.cuda.is_available()
                checks.append({"name": "GPU", "status": "pass" if gpu_ok else "warn",
                               "detail": torch.cuda.get_device_name(0) if gpu_ok else "No GPU"})
            except ImportError:
                checks.append({"name": "GPU", "status": "fail", "detail": "torch not installed"})
            disk = shutil.disk_usage(".")
            checks.append({"name": "Disk Space", "status": "pass" if disk.free > 10 * 1024**3 else "warn",
                           "detail": f"{round(disk.free / 1024**3, 1)} GB free"})
            for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
                try:
                    __import__(pkg)
                    checks.append({"name": f"Package: {pkg}", "status": "pass", "detail": "Installed"})
                except ImportError:
                    checks.append({"name": f"Package: {pkg}", "status": "fail", "detail": "Not installed"})
            return json.dumps({"success": True, "checks": checks, "all_passed": all(c["status"] != "fail" for c in checks)}, indent=2)

    def _register_finetune_tools(self):
        @self.mcp.tool(name="finetune.train", description="Fine-tune with SFT + QLoRA")
        async def train(
            dataset_path: str, output_dir: str, base_model: Optional[str] = None,
            num_epochs: int = 3, use_lora: bool = True, lora_r: int = 8, lora_alpha: int = 16,
            completion_only_loss: bool = True, early_stopping_patience: Optional[int] = None,
            eval_file_path: Optional[str] = None, push_to_hub: Optional[str] = None,
            max_seq_length: int = 2048, per_device_train_batch_size: int = 1,
            gradient_accumulation_steps: int = 4, learning_rate: float = 2e-4,
        ) -> str:
            load_result = await self.finetuner.load_dataset_from_file(dataset_path, "jsonl")
            if not load_result["success"]:
                return json.dumps(load_result, indent=2)
            result = await self.finetuner.train_model(
                dataset=load_result["dataset_object"], output_dir=output_dir,
                base_model=base_model, num_epochs=num_epochs, use_lora=use_lora,
                lora_r=lora_r, lora_alpha=lora_alpha,
                completion_only_loss=completion_only_loss,
                early_stopping_patience=early_stopping_patience,
                eval_file_path=eval_file_path, push_to_hub=push_to_hub,
                max_seq_length=max_seq_length,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate,
            )
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="finetune.load_dataset", description="Load a dataset from file")
        async def load_dataset(file_path: str, format: str = "jsonl") -> str:
            result = await self.finetuner.load_dataset_from_file(file_path, format)
            if "dataset_object" in result:
                del result["dataset_object"]
            return json.dumps(result, indent=2)

        @self.mcp.tool(name="finetune.merge_adapter",
                       description="Merge LoRA adapter into base model for standalone deployment")
        async def merge_adapter(
            base_model: str, adapter_path: str, output_path: str,
            push_to_hub: Optional[str] = None,
        ) -> str:
            return json.dumps(await self.finetuner.merge_adapter(
                base_model=base_model, adapter_path=adapter_path,
                output_path=output_path, push_to_hub=push_to_hub,
            ), indent=2)

        @self.mcp.tool(name="finetune.export_gguf",
                       description="Export model to GGUF format for llama.cpp/Ollama")
        async def export_gguf(
            model_path: str, output_path: str, quantization: str = "q4_k_m",
        ) -> str:
            return json.dumps(await self.finetuner.export_gguf(
                model_path=model_path, output_path=output_path, quantization=quantization,
            ), indent=2)

    def _register_test_tools(self):
        @self.mcp.tool(name="test.inference", description="Run inference on prompts using a model")
        async def run_inference(
            prompts: List[str], model_path: str,
            adapter_path: Optional[str] = None, max_new_tokens: int = 512,
        ) -> str:
            return json.dumps(await self.finetuner.run_inference(
                prompts=prompts, model_path=model_path,
                adapter_path=adapter_path, max_new_tokens=max_new_tokens,
            ), indent=2)

        @self.mcp.tool(name="test.compare_models",
                       description="Compare base vs fine-tuned model outputs side-by-side")
        async def compare_models(
            prompts: List[str], base_model_path: str,
            finetuned_adapter_path: Optional[str] = None, max_new_tokens: int = 512,
        ) -> str:
            return json.dumps(await self.finetuner.compare_models(
                prompts=prompts, base_model_path=base_model_path,
                finetuned_adapter_path=finetuned_adapter_path,
                max_new_tokens=max_new_tokens,
            ), indent=2)

    def _register_validate_tools(self):
        @self.mcp.tool(name="validate.model_info",
                       description="Get info about a local model or adapter")
        async def model_info(model_path: str) -> str:
            return json.dumps(self.finetuner.get_model_info(model_path), indent=2)

        @self.mcp.tool(name="validate.list_models",
                       description="List locally cached HuggingFace models")
        async def list_models(query: Optional[str] = None) -> str:
            return json.dumps(await self.finetuner.search_local_models(query=query), indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
