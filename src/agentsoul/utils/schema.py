"""
Shared schema generation utilities.

Extracts JSON Schema from Python function signatures and type hints.
Used by both ToolService.tool() and MCPServer.tool() decorators.
"""

import inspect
import re
from typing import Any, Callable, Dict, Union, get_type_hints


def python_type_to_json_schema(py_type: type) -> dict:
    """Map a Python type annotation to a JSON Schema dict.

    Handles primitives, ``List[X]``, ``Dict[K, V]``, and ``Optional[X]``.
    """
    type_map = {
        int: "integer",
        float: "number",
        str: "string",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    origin = getattr(py_type, "__origin__", None)

    # List[X] -> {"type": "array", "items": {...}}
    if origin is list:
        args = getattr(py_type, "__args__", ())
        inner = python_type_to_json_schema(args[0]) if args else {"type": "string"}
        return {"type": "array", "items": inner}

    # Dict[K, V] -> {"type": "object"}
    if origin is dict:
        return {"type": "object"}

    # Optional[X] / Union[X, None]
    if origin is Union:
        args = getattr(py_type, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return python_type_to_json_schema(non_none[0])

    json_type = type_map.get(py_type, "string")
    return {"type": json_type}


def parse_param_descriptions(func: Callable) -> Dict[str, str]:
    """Extract per-parameter descriptions from a function's docstring.

    Supports Google-style ``Args:`` sections and reST-style ``:param name:`` lines.
    Returns a mapping of ``{param_name: description}``.
    """
    doc = inspect.getdoc(func)
    if not doc:
        return {}

    descriptions: Dict[str, str] = {}

    # --- Google-style: "Args:" section ---
    args_match = re.search(r"^Args:\s*$", doc, re.MULTILINE)
    if args_match:
        # Everything after "Args:\n"
        block = doc[args_match.end():]
        for line in block.splitlines():
            # Stop at the next top-level section (e.g. "Returns:", "Raises:")
            if re.match(r"^\S", line):
                break
            # Match "    param_name: description" or "    param_name (type): description"
            m = re.match(r"^\s+(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+)", line)
            if m:
                descriptions[m.group(1)] = m.group(2).strip()

    # --- reST-style: ":param name: description" ---
    for m in re.finditer(r":param\s+(\w+)\s*:\s*(.+)", doc):
        descriptions[m.group(1)] = m.group(2).strip()

    return descriptions


def generate_tool_schema(func: Callable) -> dict:
    """Build a JSON Schema ``{"type": "object", "properties": ..., "required": ...}``
    from a function's signature and type hints.

    Parameter descriptions are pulled from the docstring when available.
    """
    sig = inspect.signature(func)

    try:
        type_hints = get_type_hints(func)
    except Exception:
        # Gracefully handle `from __future__ import annotations` or other failures
        type_hints = {}

    param_descriptions = parse_param_descriptions(func)

    properties: Dict[str, dict] = {}
    required = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        py_type = type_hints.get(param_name, Any)
        prop = python_type_to_json_schema(py_type)
        prop["description"] = param_descriptions.get(
            param_name, f"Parameter: {param_name}"
        )
        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema
