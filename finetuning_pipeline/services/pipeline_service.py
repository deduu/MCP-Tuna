"""
Fine-tuning Pipeline Service — Thin Facade
============================================

Delegates to focused sub-services while preserving the original API
so existing callers (MCP server, scripts) keep working.
"""

from typing import Any, Dict, List, Optional

from AgentY.shared.config import FinetuningConfig
from .gpu_service import GPUService
from .training_service import TrainingService
from .inference_service import InferenceService
from .model_discovery_service import ModelDiscoveryService


class FineTuningService:
    """Facade that composes GPU, training, inference, and discovery sub-services."""

    def __init__(self, default_base_model: str = "meta-llama/Llama-3.2-3B-Instruct"):
        config = FinetuningConfig(base_model=default_base_model)
        self.gpu = GPUService()
        self.training = TrainingService(config=config, gpu=self.gpu)
        self.inference = InferenceService(gpu=self.gpu)
        self.discovery = ModelDiscoveryService()

    # ---- GPU ----
    def clear_gpu_memory(self) -> Dict[str, Any]:
        return self.gpu.clear_gpu_memory()

    # ---- Training ----
    async def prepare_dataset(self, *a, **kw) -> Dict[str, Any]:
        return await self.training.prepare_dataset(*a, **kw)

    async def load_dataset_from_file(self, *a, **kw) -> Dict[str, Any]:
        return await self.training.load_dataset_from_file(*a, **kw)

    async def train_model(self, *a, **kw) -> Dict[str, Any]:
        return await self.training.train_model(*a, **kw)

    # ---- Inference ----
    async def run_inference(self, *a, **kw) -> Dict[str, Any]:
        return await self.inference.run_inference(*a, **kw)

    async def compare_models(self, *a, **kw) -> Dict[str, Any]:
        return await self.inference.compare_models(*a, **kw)

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
