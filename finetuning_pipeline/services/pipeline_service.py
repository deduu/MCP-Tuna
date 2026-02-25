"""
Fine-tuning Pipeline Service — Thin Facade
============================================

Delegates to focused sub-services while preserving the original API
so existing callers (MCP server, scripts) keep working.
"""

from typing import Any, Dict, List, Optional

from shared.config import FinetuningConfig
from .model_discovery_service import ModelDiscoveryService


class FineTuningService:
    """Facade that composes GPU, training, inference, and discovery sub-services."""

    def __init__(self, default_base_model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        self.config = FinetuningConfig(base_model=default_base_model)
        self._gpu = None
        self._training = None
        self._inference = None
        self.discovery = ModelDiscoveryService()

    def _ensure_gpu(self):
        if self._gpu is None:
            from .gpu_service import GPUService
            self._gpu = GPUService()
        return self._gpu

    def _ensure_training(self):
        if self._training is None:
            from .training_service import TrainingService
            # TrainingService does not need GPU initialization for dataset prep/load.
            self._training = TrainingService(config=self.config, gpu=None)
        return self._training

    def _ensure_inference(self):
        if self._inference is None:
            from .inference_service import InferenceService
            self._inference = InferenceService(gpu=self._ensure_gpu())
        return self._inference

    # ---- GPU ----
    def clear_gpu_memory(self) -> Dict[str, Any]:
        try:
            return self._ensure_gpu().clear_gpu_memory()
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") == "torch":
                return {"success": False, "error": "torch is not installed. Install dependencies before using GPU tools."}
            return {"success": False, "error": str(e)}

    # ---- Training ----
    async def prepare_dataset(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_training().prepare_dataset(*a, **kw)

    async def load_dataset_from_file(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_training().load_dataset_from_file(*a, **kw)

    async def train_model(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_training().train_model(*a, **kw)

    # ---- Inference ----
    async def run_inference(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_inference().run_inference(*a, **kw)

    async def compare_models(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_inference().compare_models(*a, **kw)

    # ---- Discovery ----
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        return self.discovery.get_model_info(model_path)

    async def list_available_base_models(self, *a, **kw) -> Dict[str, Any]:
        return await self.discovery.list_available_base_models(*a, **kw)

    async def search_huggingface_models(self, *a, **kw) -> Dict[str, Any]:
        return await self.discovery.search_huggingface_models(*a, **kw)

    async def get_huggingface_model_info(self, model_id: str) -> Dict[str, Any]:
        return await self.discovery.get_huggingface_model_info(model_id)

    async def search_local_models(self, *a, **kw) -> Dict[str, Any]:
        return await self.discovery.search_local_models(*a, **kw)

    def get_recommended_models(self, use_case: str = "general") -> Dict[str, Any]:
        return self.discovery.get_recommended_models(use_case)
