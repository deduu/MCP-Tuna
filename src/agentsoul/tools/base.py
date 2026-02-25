# agentsoul/tools/service.py
import inspect
import json
from typing import Dict, List, Tuple, Any, Callable, Optional
from abc import ABC, abstractmethod
from agentsoul.core.models import Message


class BaseToolService(ABC):
    """Abstract base class for tool services"""

    def __init__(self):
        self.tools = {}

    @abstractmethod
    def get_tool_names(self) -> List[Dict]:
        """Return list of tool names for LLM"""
        pass

    @abstractmethod
    def get_tool_descriptions(self) -> List[Dict]:
        """Return list of tool descriptions for LLM"""
        pass

    @abstractmethod
    async def execute_tool(self, name: str, **kwargs) -> Tuple[str, Any]:
        """Execute a tool and return (json_result, metadata)"""
        pass

    @abstractmethod
    async def validate_tool_call(self, user_question: str, name: str, function_args: Dict, called_functions):
        """Validate a tool call and return (json_result, metadata)"""
        pass
