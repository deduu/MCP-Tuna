import inspect
import json
from typing import Dict, List, Tuple, Any

from .base import BaseToolService


class CompositeToolService(BaseToolService):
    """
    Composite tool service that manages multiple sub-services.
    Useful for complex applications with different tool categories.
    """

    def __init__(self):
        super().__init__()
        self.services: Dict[str, BaseToolService] = {}
        self._tool_registry: Dict[str, str] = {}  # tool_name -> service_name

    def register_service(self, name: str, service: BaseToolService):
        """Register a sub-service"""
        self.services[name] = service

        # Build registry of which service owns which tool
        for desc in service.get_tool_descriptions():
            tool_name = desc.get("name")
            if tool_name:
                self._tool_registry[tool_name] = name

    def get_tool_names(self) -> List[str]:
        """Aggregate tool names from all services"""
        tool_names = []
        for service in self.services.values():
            tool_names.extend(service.get_tool_names())
        return tool_names

    def get_tool_descriptions(self) -> List[Dict]:
        """Aggregate tool descriptions from all services"""
        descriptions = []
        for service in self.services.values():
            descriptions.extend(service.get_tool_descriptions())
        return descriptions

    async def execute_tool(self, name: str, **kwargs) -> Tuple[str, Any]:
        """Route tool execution to appropriate service"""
        service_name = self._tool_registry.get(name)
        if not service_name:
            raise ValueError(
                f"Tool '{name}' not found in any registered service")

        service = self.services[service_name]
        return await service.execute_tool(name, **kwargs)

    async def validate_tool_call(self, user_question, name, function_args, called_functions):
        service_name = self._tool_registry.get(name)
        if not service_name:
            raise ValueError(
                f"Tool '{name}' not found in any registered service")
        service = self.services[service_name]
        # If the sub-service has its own validator, use it; otherwise no-op
        if hasattr(service, "validate_tool_call"):
            return await service.validate_tool_call(user_question, name, function_args, called_functions)
        return function_args, None
