import ast
import re
import uuid
from typing import List, Dict, Any, Optional

from agentsoul.core.tool_strategy import ToolCallingStrategy
from agentsoul.core.models import ToolCall


# Map JSON Schema types to Python type hints
_JSON_SCHEMA_TO_PYTHON = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
}

# AST node types that are never allowed
_BLOCKED_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.Lambda,
)

# Function names that are never allowed to be called
_BLOCKED_CALLS = frozenset({
    "eval", "exec", "__import__", "compile", "open",
    "getattr", "setattr", "delattr", "globals", "locals",
    "breakpoint", "exit", "quit", "input", "help",
    "vars", "dir", "type", "super", "classmethod",
    "staticmethod", "property",
})


class CodeActStrategy(ToolCallingStrategy):
    """
    Code-act strategy (AST-only): the LLM generates Python code that calls
    tool functions directly. The code is parsed with ast — never executed.
    Function calls are extracted and routed through ToolService.execute_tool().
    """

    def format_tools_for_prompt(
        self, tool_descriptions: List[Dict[str, Any]]
    ) -> str:
        """Convert JSON schema tool descriptions to Python function stubs."""
        stubs = []
        for tool in tool_descriptions:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            properties = params.get("properties", {})
            required = set(params.get("required", []))

            # Build parameter list: required params first, then optional
            required_parts = []
            optional_parts = []
            for pname, pschema in properties.items():
                ptype = _JSON_SCHEMA_TO_PYTHON.get(
                    pschema.get("type", "string"), "Any"
                )
                pdesc = pschema.get("description", "")
                if pname in required:
                    required_parts.append((pname, ptype, pdesc))
                else:
                    optional_parts.append((pname, ptype, pdesc))

            param_strs = []
            for pname, ptype, _ in required_parts:
                param_strs.append(f"{pname}: {ptype}")
            for pname, ptype, _ in optional_parts:
                param_strs.append(f"{pname}: {ptype} = None")

            sig = ", ".join(param_strs)

            # Build docstring with param descriptions
            doc_lines = [desc] if desc else []
            all_params = required_parts + optional_parts
            if all_params:
                doc_lines.append("")
                doc_lines.append("Args:")
                for pname, ptype, pdesc in all_params:
                    doc_lines.append(f"    {pname} ({ptype}): {pdesc or pname}")

            docstring = "\n    ".join(doc_lines)
            stub = f'def {name}({sig}):\n    """{docstring}"""\n    ...\n'
            stubs.append(stub)

        return "\n".join(stubs)

    def should_pass_tools_to_api(self) -> bool:
        return False

    def get_system_prompt_addition(self) -> str:
        return (
            "\n\n## Tool Calling Instructions\n"
            "You have access to the Python functions listed above. "
            "To call tools, write Python code inside a ```python code block. "
            "Call the functions directly with keyword arguments.\n\n"
            "Example:\n"
            "```python\n"
            "result = search(query=\"hello world\", limit=5)\n"
            "```\n\n"
            "Rules:\n"
            "- ONLY call the functions listed above.\n"
            "- Use keyword arguments for all parameters.\n"
            "- Do NOT use import statements.\n"
            "- Do NOT use eval(), exec(), or __import__().\n"
            "- Do NOT define new functions or classes.\n"
            "- You may call multiple functions in a single code block.\n"
            "- After receiving tool results, provide your final answer in plain text.\n"
        )

    def parse_tool_calls(
        self, content: str, registered_tool_names: set
    ) -> Optional[List[ToolCall]]:
        """
        Extract Python code blocks from LLM output, parse with ast,
        and convert function calls to ToolCall objects.
        """
        code_blocks = self._extract_code_blocks(content)
        if not code_blocks:
            return None

        all_calls: List[ToolCall] = []
        for code in code_blocks:
            calls = self._parse_code_block(code, registered_tool_names)
            all_calls.extend(calls)

        return all_calls if all_calls else None

    @staticmethod
    def _extract_code_blocks(content: str) -> List[str]:
        """Extract code from ```python ... ``` fenced blocks."""
        pattern = r"```python\s*\n(.*?)```"
        matches = re.findall(pattern, content, re.DOTALL)
        return [m.strip() for m in matches if m.strip()]

    @staticmethod
    def _parse_code_block(
        code: str, registered_tool_names: set
    ) -> List[ToolCall]:
        """
        Parse a Python code block using ast and extract function calls.
        Only allows calls to registered tool functions.
        Raises ValueError for unsafe constructs.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []

        # Reject blocked node types
        for node in ast.walk(tree):
            if isinstance(node, _BLOCKED_NODES):
                raise ValueError(
                    f"Disallowed construct: {type(node).__name__}. "
                    "Only tool function calls are allowed."
                )

        tool_calls: List[ToolCall] = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            func_name = _get_call_name(node)
            if func_name is None:
                continue

            # Block dangerous builtins
            if func_name in _BLOCKED_CALLS:
                raise ValueError(
                    f"Calling '{func_name}' is not allowed in tool code."
                )

            # Only extract calls to registered tools
            if func_name not in registered_tool_names:
                continue

            arguments = _extract_arguments(node)
            tool_calls.append(ToolCall(
                id=f"codeact_{uuid.uuid4().hex[:8]}",
                name=func_name,
                arguments=arguments,
            ))

        return tool_calls


def _get_call_name(node: ast.Call) -> Optional[str]:
    """Get the function name from an ast.Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    # obj.method() style calls are not supported for tool calls
    return None


def _extract_arguments(node: ast.Call) -> Dict[str, Any]:
    """
    Extract arguments from an ast.Call node.
    Uses ast.literal_eval for safe literal evaluation.
    """
    args: Dict[str, Any] = {}

    # Positional arguments → _pos_0, _pos_1, etc.
    for i, arg in enumerate(node.args):
        try:
            value = ast.literal_eval(arg)
        except (ValueError, TypeError):
            # Non-literal (variable reference etc.) — store as string repr
            value = ast.unparse(arg)
        args[f"_pos_{i}"] = value

    # Keyword arguments (preferred)
    for kw in node.keywords:
        if kw.arg is None:
            continue  # **kwargs expansion — skip
        try:
            value = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            value = ast.unparse(kw.value)
        args[kw.arg] = value

    return args
