import ast
import asyncio
import inspect
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple

from agentsoul.core.models import ToolCall
from agentsoul.core.strategies.code_act import (
    CodeActStrategy,
    _BLOCKED_NODES,
    _BLOCKED_CALLS,
)


_SENTINEL = object()


class CodeActExecStrategy(CodeActStrategy):
    """
    Code-act strategy with sandboxed execution: the LLM generates Python code
    that calls tool functions directly. The code is validated via AST, then
    executed in a restricted namespace with only registered tools available.

    Advantage over AST-only: supports variable chaining between tool calls
    within a single code block, e.g.:
        result = search(query="hello")
        detail = lookup(id=result["id"])

    When single_shot=True (default), the strategy captures a 'result' variable
    from the sandbox namespace. If the LLM stores its final answer there, the
    agent can skip the finalization LLM turn entirely (1 API call instead of 2+).
    """

    def __init__(self, single_shot: bool = True):
        self._last_exec_results: List[Dict[str, Any]] = []
        self._final_result: Optional[str] = None
        self._single_shot: bool = single_shot

    @property
    def last_exec_results(self) -> List[Dict[str, Any]]:
        """Results captured from the last code execution."""
        return self._last_exec_results

    @property
    def final_result(self) -> Optional[str]:
        """Synthesized final answer from the last execution, if available."""
        return self._final_result

    def get_system_prompt_addition(self) -> str:
        base = super().get_system_prompt_addition()
        if not self._single_shot:
            return base
        return base + (
            "\n\n## Important: Store Final Answer\n"
            "Always store your final computed answer in a variable named `result`.\n"
            "This variable should contain the complete answer to the user's question.\n\n"
            "Example:\n"
            "```python\n"
            "data = search(query=\"hello world\")\n"
            "result = data[0][\"title\"]\n"
            "```\n"
        )

    def parse_tool_calls(
        self, content: str, registered_tool_names: set,
        tool_service=None,
    ) -> Optional[List[ToolCall]]:
        """
        Extract Python code blocks, validate with AST, then execute
        in a sandboxed namespace with registered tools as callables.

        Args:
            content: LLM output text
            registered_tool_names: set of valid tool names
            tool_service: ToolService instance (needed for exec mode to get callables)
        """
        code_blocks = self._extract_code_blocks(content)
        if not code_blocks:
            return None

        all_calls: List[ToolCall] = []
        self._last_exec_results = []
        self._final_result = None
        last_captured = _SENTINEL

        for code in code_blocks:
            calls, results, captured = self._execute_code_block(
                code, registered_tool_names, tool_service
            )
            all_calls.extend(calls)
            self._last_exec_results.extend(results)
            if captured is not _SENTINEL:
                last_captured = captured

        # Single-shot: if enabled and a 'result' variable was captured, format it
        if self._single_shot and last_captured is not _SENTINEL:
            self._final_result = self._format_final_result(last_captured)

        return all_calls if all_calls else None

    def _execute_code_block(
        self,
        code: str,
        registered_tool_names: set,
        tool_service=None,
    ) -> Tuple[List[ToolCall], List[Dict[str, Any]], Any]:
        """
        Validate and execute a code block in a restricted namespace.
        Returns (tool_calls, execution_results, captured_result).
        captured_result is _SENTINEL if no 'result' variable was set.
        """
        # Phase 1: AST validation (same as CodeActStrategy)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return [], [], _SENTINEL

        for node in ast.walk(tree):
            if isinstance(node, _BLOCKED_NODES):
                raise ValueError(
                    f"Disallowed construct: {type(node).__name__}. "
                    "Only tool function calls are allowed."
                )

        # Check for blocked function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in _BLOCKED_CALLS:
                        raise ValueError(
                            f"Calling '{node.func.id}' is not allowed."
                        )

        # Phase 2: Build sandboxed namespace with tool wrappers
        if tool_service is None:
            # Fall back to AST-only parsing if no tool_service
            calls = self._parse_code_block(code, registered_tool_names)
            return calls, [], _SENTINEL

        call_log: List[Dict[str, Any]] = []
        namespace = _build_sandbox_namespace(
            tool_service, registered_tool_names, call_log
        )

        # Phase 3: Execute in sandbox
        try:
            exec(compile(tree, "<code_act>", "exec"), {"__builtins__": {}}, namespace)
        except Exception as e:
            raise ValueError(f"Code execution failed: {e}")

        # Phase 3.5: Capture 'result' variable for single-shot optimization
        captured_result = namespace.get("result", _SENTINEL)

        # Phase 4: Convert call log to ToolCall objects
        tool_calls = []
        results = []
        for entry in call_log:
            tc = ToolCall(
                id=f"codeact_exec_{uuid.uuid4().hex[:8]}",
                name=entry["name"],
                arguments=entry["args"],
            )
            tool_calls.append(tc)
            results.append({
                "tool_call": tc,
                "result": entry["result"],
            })

        return tool_calls, results, captured_result

    @staticmethod
    def _format_final_result(value: Any) -> Optional[str]:
        """Format a captured 'result' variable into a final answer string.

        Returns None if the value is not meaningful enough to use as a final answer.
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value if value.strip() else None
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(value)
        try:
            return json.dumps(value, default=str, indent=2, ensure_ascii=False)
        except Exception:
            return str(value)


def _build_sandbox_namespace(
    tool_service, registered_tool_names: set, call_log: list
) -> Dict[str, Any]:
    """
    Build a restricted execution namespace containing only registered
    tool functions as sync-callable wrappers.
    """
    namespace: Dict[str, Any] = {}
    available_funcs = tool_service.get_available_functions()

    for name in registered_tool_names:
        func = available_funcs.get(name)
        if func is None:
            continue

        def _make_wrapper(tool_name: str, tool_func):
            def wrapper(*args, **kwargs):
                # Map positional args to keyword args if possible
                # (tools use **kwargs, so positional args need handling)
                if args:
                    # Store positional args as _pos_N
                    for i, v in enumerate(args):
                        kwargs[f"_pos_{i}"] = v

                # Execute the tool function
                if inspect.iscoroutinefunction(tool_func):
                    # Run async tool in event loop
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None

                    if loop and loop.is_running():
                        # We're inside an async context — run in the existing loop
                        future = asyncio.run_coroutine_threadsafe(tool_func(**kwargs), loop)
                        result = future.result()
                    else:
                        result = asyncio.run(tool_func(**kwargs))
                else:
                    result = tool_func(**kwargs)

                # Normalize result
                if isinstance(result, (dict, list)):
                    json_result = json.dumps(result, ensure_ascii=False)
                elif isinstance(result, (int, float, bool, str)):
                    json_result = str(result)
                else:
                    try:
                        json_result = json.dumps(result, default=str, ensure_ascii=False)
                    except Exception:
                        json_result = str(result)

                call_log.append({
                    "name": tool_name,
                    "args": kwargs,
                    "result": json_result,
                })

                # Return the raw result so chaining works
                return result

            return wrapper

        namespace[name] = _make_wrapper(name, func)

    # Add safe builtins that are commonly needed
    namespace["len"] = len
    namespace["str"] = str
    namespace["int"] = int
    namespace["float"] = float
    namespace["bool"] = bool
    namespace["list"] = list
    namespace["dict"] = dict
    namespace["tuple"] = tuple
    namespace["set"] = set
    namespace["range"] = range
    namespace["enumerate"] = enumerate
    namespace["zip"] = zip
    namespace["sorted"] = sorted
    namespace["reversed"] = reversed
    namespace["min"] = min
    namespace["max"] = max
    namespace["sum"] = sum
    namespace["abs"] = abs
    namespace["round"] = round
    namespace["print"] = lambda *a, **kw: None  # no-op print
    namespace["True"] = True
    namespace["False"] = False
    namespace["None"] = None
    namespace["isinstance"] = isinstance
    namespace["json"] = type("json_module", (), {
        "loads": staticmethod(json.loads),
        "dumps": staticmethod(json.dumps),
    })()

    return namespace
