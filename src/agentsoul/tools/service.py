import asyncio
import inspect
import json
import logging
from typing import Dict, List
from agentsoul.utils.schema import generate_tool_schema
from .base import BaseToolService

logger = logging.getLogger(__name__)


class ToolService(BaseToolService):
    """Generic tool service interface"""

    def __init__(self):
        self.tools = {}

    def register_tool(self, name: str, func, description: Dict, validator=None):
        self.tools[name] = {"function": func,
                            "description": description, "validator": validator}

    def tool(self, name=None, description=None, validator=None):
        """Decorator to register a function as a tool with auto-schema generation.

        Usage::

            ts = ToolService()

            @ts.tool()
            def get_weather(city: str, units: str = "celsius") -> str:
                \"\"\"Get current weather for a city.

                Args:
                    city: The city name to look up.
                    units: Temperature units (celsius or fahrenheit).
                \"\"\"
                return f"Sunny in {city}"
        """
        def decorator(func):
            tool_name = name or func.__name__
            tool_desc = description or (inspect.getdoc(func) or f"Function: {tool_name}")
            schema = generate_tool_schema(func)
            self.register_tool(
                name=tool_name,
                func=func,
                description={"name": tool_name, "description": tool_desc, "parameters": schema},
                validator=validator,
            )
            return func
        return decorator

    def get_tool_names(self) -> List[str]:
        return list(self.tools.keys())

    def get_tool_descriptions(self) -> List[Dict]:
        return [tool["description"] for tool in self.tools.values()]

    def get_available_functions(self) -> Dict:
        return {name: tool["function"] for name, tool in self.tools.items()}

    async def execute_tool(self, name: str, **kwargs):
        if name not in self.tools:
            raise ValueError(f"Tool '{name}' not found")
        func = self.tools[name]["function"]

        if inspect.iscoroutinefunction(func):
            tool_result = await func(**kwargs)
        else:
            tool_result = await asyncio.to_thread(func, **kwargs)
        if isinstance(tool_result, (dict, list)):
            json_result = json.dumps(tool_result, ensure_ascii=False)
        elif isinstance(tool_result, (int, float, bool, str)):
            json_result = str(tool_result)
        else:
            try:
                json_result = json.dumps(
                    tool_result, default=str, ensure_ascii=False)
            except Exception:
                json_result = str(tool_result)

        return json_result, tool_result

    async def validate_tool_call(self, user_question: str, name: str, function_args: Dict, called_functions):
        # if user_question not a string, convert to string
        if not isinstance(user_question, str):
            user_question = str(user_question)

        warnings = None
        logger.debug("Validating %s with user question: %s", name, user_question)

        if "validator" in self.tools[name]:
            logger.debug("Tool call argument validation for %s", name)
            validator_func = self.tools[name]["validator"]
            if validator_func:
                if inspect.iscoroutinefunction(validator_func):
                    validated_args, func_warnings = await validator_func(question=user_question, name=name, args=function_args, called_functions=called_functions)
                else:
                    validated_args, func_warnings = validator_func(
                        question=user_question, name=name, args=function_args, called_functions=called_functions)
                function_args = {**function_args, **validated_args}
                warnings = func_warnings

        return function_args, warnings
