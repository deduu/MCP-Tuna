import logging
from typing import Dict, Any

from ..utils.exception import ModelRoutingError
from app.generation.qwen3_registry import route_qwen3

logger = logging.getLogger("api.chat")


class ModelRouteDecider:
    """Handles model routing logic and decisions."""

    def __init__(self, model_router):
        self.model_router = model_router

    def decide_route_and_enable_logging(self, user_prompt: str, model_name: str) -> Dict[str, Any]:
        """Determine the appropriate model route based on prompt and model name."""
        enable_logging = True

        if not self.model_router:
            # Default: no routing, just use requested model
            return {"route": "basic"}, enable_logging

        route_result = self.model_router.route(user_prompt)
        logger.info(f"Received model route decision: {route_result['route']}.")

        if route_result["route"] == "follow_up_and_chat_history":
            enable_logging = False
            logger.info(
                "Disabling logging for follow-up and chat history route.")

        if "gpt" in model_name.lower():
            return route_result, enable_logging

        return self._route_by_model_name(model_name, route_result), enable_logging

    @staticmethod
    def _route_by_model_name(model_name: str, default_route: Dict[str, Any]) -> Dict[str, Any]:
        """Map model names to specific routes."""
        model_route_map = {
            "Instant": {"route": "basic"},
            "Thinking": {"route": "complex"},
            "Fine-Tuned": {"route": "fine-tuned"},
        }
        return model_route_map.get(model_name, default_route)

    def should_enable_thinking(self, route: str) -> bool:
        """Determine if thinking should be enabled based on route."""
        if not self.model_router:
            return False
        return route in ['complex', 'multi_qa', 'other']


class ModelSelector:
    """Handles model selection and routing."""

    @staticmethod
    async def select_model(model_name: str, model_route: Dict[str, Any]) -> str:
        """Select the appropriate model based on name and route."""
        qwen_models = ['Auto', 'Instant', 'Thinking', 'Fine-Tuned']

        if model_name not in qwen_models:
            return model_name

        try:
            selected_key = await route_qwen3(model_name, model_route)
            return selected_key if selected_key else model_name
        except Exception as e:
            logger.error(f"Error occurred while routing Qwen3 model: {e}")
            raise ModelRoutingError(f"Error during Qwen3 model routing: {e}")
