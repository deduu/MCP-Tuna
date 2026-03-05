"""
Fine-tuning Pipeline Service — Thin Facade
============================================

Delegates to focused sub-services while preserving the original API
so existing callers (MCP server, scripts) keep working.
"""

from typing import Any, Dict

from shared.config import FinetuningConfig
from .model_discovery_service import ModelDiscoveryService


class FineTuningService:
    """Facade that composes GPU, training, inference, and discovery sub-services."""

    def __init__(self, default_base_model: str = "Qwen/Qwen3-1.7B"):
        self.config = FinetuningConfig(base_model=default_base_model)
        self._gpu = None
        self._training = None
        self._inference = None
        self._curriculum = None
        self._sequential = None
        self._resources = None
        self._adapter = None
        self.discovery = ModelDiscoveryService()

    def _ensure_gpu(self):
        if self._gpu is None:
            from .gpu_service import GPUService
            self._gpu = GPUService()
        return self._gpu

    def _ensure_training(self):
        if self._training is None:
            from .training_service import TrainingService
            self._training = TrainingService(config=self.config, gpu=self._ensure_gpu())
        return self._training

    def _ensure_inference(self):
        if self._inference is None:
            from .inference_service import InferenceService
            self._inference = InferenceService(gpu=self._ensure_gpu())
        return self._inference

    def _ensure_curriculum(self):
        if self._curriculum is None:
            from .curriculum_service import CurriculumService
            self._curriculum = CurriculumService(config=self.config)
        return self._curriculum

    def _ensure_sequential(self):
        if self._sequential is None:
            from .sequential_service import SequentialTrainingService
            self._sequential = SequentialTrainingService(config=self.config)
        return self._sequential

    def _ensure_resources(self):
        if self._resources is None:
            from .resource_service import ResourceService
            self._resources = ResourceService()
        return self._resources

    # ---- Resources ----
    def check_resources(self) -> Dict[str, Any]:
        return self._ensure_resources().check_resources()

    def preflight_check(self, *a, **kw) -> Dict[str, Any]:
        return self._ensure_resources().preflight_check(*a, **kw)

    def prescribe(self, *a, **kw) -> Dict[str, Any]:
        return self._ensure_resources().prescribe(*a, **kw)

    def disk_preflight(self, *a, **kw) -> Dict[str, Any]:
        return self._ensure_resources().disk_preflight(*a, **kw)

    def prescribe_pipeline(self, *a, **kw) -> Dict[str, Any]:
        return self._ensure_resources().prescribe_pipeline(*a, **kw)

    def auto_prescribe(self, *a, **kw) -> Dict[str, Any]:
        return self._ensure_resources().auto_prescribe(*a, **kw)

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

    async def train_dpo_model(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_training().train_dpo_model(*a, **kw)

    async def train_grpo_model(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_training().train_grpo_model(*a, **kw)

    async def train_kto_model(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_training().train_kto_model(*a, **kw)

    async def train_curriculum_model(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_curriculum().train_curriculum_model(*a, **kw)

    async def train_sequential(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_sequential().train_sequential(*a, **kw)

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

    # ---- Adapter ----
    def _ensure_adapter(self):
        if self._adapter is None:
            from .adapter_service import AdapterService
            self._adapter = AdapterService()
        return self._adapter

    async def merge_adapter(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_adapter().merge_adapter(*a, **kw)

    async def export_gguf(self, *a, **kw) -> Dict[str, Any]:
        return await self._ensure_adapter().export_gguf(*a, **kw)
