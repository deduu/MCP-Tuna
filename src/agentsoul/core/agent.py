import asyncio
import json
import time
import re
from typing import Awaitable, Callable, List, Dict, Any, Optional, AsyncGenerator, Union
from agentsoul.providers.base import BaseLLM
from agentsoul.tools.service import ToolService
from agentsoul.core.models import ToolCall, MessageRole, Message, ReflectionMode, ReflectionPolicy
from agentsoul.core.message import MessageFormatter
from agentsoul.core.tool_strategy import ToolCallingStrategy
from agentsoul.core.strategies.json_schema import JsonSchemaStrategy
from agentsoul.utils.logger import get_logger
from agentsoul.tools.orchestrator import ToolOrchestrator

# --- auditi tracing: optional ---
try:
    from auditi import trace_agent, trace_tool, trace_llm
except ImportError:
    # No-op decorator fallbacks when auditi is not installed
    def _noop_decorator(**kwargs):
        def wrapper(func):
            return func
        return wrapper

    trace_agent = _noop_decorator
    trace_tool = _noop_decorator
    trace_llm = _noop_decorator

# --- Plain string template (replaces langchain PromptTemplate) ---
_PROMPT_TEMPLATE = """# Question:
{query_text}

# Note:
If the question contains technical terms use the RAG tools to get definitions first, DO NOT interpret the meaning by yourself! Then, create a plan to answer the user's question. But, do not include your plan in the final answer."""

_REFLECTION_PROMPT = (
    "Reflect on your progress so far. Do you have enough information to produce "
    "a complete, accurate final answer?\n\n"
    "Respond with EXACTLY one of:\n"
    "- READY\n"
    "- NEED_MORE: <brief explanation of what is missing or wrong>\n\n"
    "Do not include anything else."
)

_SYNTHESIS_INSTRUCTION = (
    "Now answer the user's question using the data above.\n"
    "- Only use information that is DIRECTLY relevant to the question. "
    "Ignore sources that discuss a different topic even if they share keywords.\n"
    "- If a source discusses 'Product A' but the user asked about 'Product B', "
    "do NOT assume they are related or that Product A's details apply to Product B.\n"
    "- If the sources do not directly answer the user's question, say so honestly. "
    "Then share what you found about related subjects, clearly labeled as 'Related information'.\n"
    "- NEVER extrapolate or speculate. Do NOT write 'is expected to', 'is likely to', "
    "'is poised to', or 'is anticipated to' unless a source explicitly states that.\n"
    "- Cite each claim with a markdown link to its source: [Title](URL).\n"
    "- Be thorough and well-structured, but never fabricate connections "
    "between unrelated sources.\n"
    "- Do not mention the tools or search process used."
)

_FINALIZATION_INSTRUCTION = (
    "Now produce a comprehensive final answer for the user.\n\n"
    "- ONLY state facts that are directly supported by the gathered sources. "
    "If a source discusses topic X but the user asked about topic Y, do NOT "
    "assume they are related — even if they share keywords.\n"
    "- If the sources do not contain specific information about the user's "
    "exact question, say so clearly: 'I could not find specific information "
    "about [topic].' Then share what you DID find about closely related "
    "subjects, clearly labeled as such.\n"
    "- NEVER extrapolate, speculate, or predict based on tangentially related "
    "sources. Do NOT write 'is expected to', 'is likely to', or 'is poised to' "
    "unless a source explicitly states that.\n"
    "- Cite every factual claim with a markdown link: [Title](URL).\n"
    "- Synthesize across sources into a coherent narrative rather than listing "
    "sources one by one.\n"
    "- Use proper Markdown formatting (headings, bold, lists) for readability.\n"
    "- Do NOT mention the tools or process used to gather data.\n"
    "- Do NOT include your reasoning steps or plan.\n"
    "- Answer the user's question directly and substantively.\n"
)


def compute_metrics(start_time, first_token_time, end_time, total_tokens):
    ttft = first_token_time - start_time
    total_latency = end_time - start_time
    denominator = end_time - first_token_time
    tps = (total_tokens / denominator) if total_tokens > 1 and denominator > 0 else 0
    return {
        "ttft": round(ttft, 3),
        "tps": round(tps, 3),
        "total_latency": round(total_latency, 3)
    }


class AgentSoul:
    """Generic agent that works with any LLM provider"""

    def __init__(
        self,
        llm_provider: BaseLLM,
        tool_service: Optional[ToolService] = None,
        max_turns: int = 10,
        system_prompt: str = None,
        verbose: bool = True,
        enable_logging: bool = True,
        tool_strategy: Optional[ToolCallingStrategy] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        memory: Optional[Any] = None,
        reflection: Optional[ReflectionPolicy] = None,
        parallel_tool_calls: bool = True,
        tool_pre_execute: Optional[Callable[[str, Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]] = None,
    ):
        self.llm_provider = llm_provider
        self.tool_service = tool_service
        self.max_turns = max_turns
        self.system_prompt = system_prompt or "You are a helpful AI assistant."
        self.verbose = verbose
        self.logger = get_logger(self.__class__.__name__, verbose=verbose)
        self.message_formatter = MessageFormatter(llm_provider)
        self.model_id = llm_provider.model_id if hasattr(
            llm_provider, 'model_id') else 'unknown_model'
        self._run_id = None
        self.enable_logging = enable_logging
        self.tool_strategy = tool_strategy or JsonSchemaStrategy()
        self.name = name or f"agent_{id(self)}"
        self.description = description
        self.memory = memory
        self.reflection = reflection or ReflectionPolicy()
        self.parallel_tool_calls = parallel_tool_calls
        self.default_provider_kwargs: Dict[str, Any] = {}
        self.tool_pre_execute = tool_pre_execute

        self.logger.debug("AgentSoul initialized with model_id: %s", self.model_id)

    @classmethod
    async def create(
        cls,
        llm_provider: BaseLLM,
        tools: Optional[Union[List[Any], ToolService]] = None,
        **kwargs,
    ):
        """
        Asynchronously create a AgentSoul instance and register MCP tools.
        """
        self = cls(llm_provider, **kwargs)

        # Async normalize tools (MCP, HTTP, stdio)
        tr = ToolOrchestrator(logger=self.logger)
        if isinstance(tools, ToolService):
            self.tool_service = tools
        else:
            self.tool_service = await tr.normalize_tools_async(tools)

        return self

    async def invoke(self, user_input: str, **kwargs) -> str:
        """Run the agent and return only the final answer string.

        Convenience method for programmatic use (e.g., multi-agent delegation).
        Consumes the full run() generator and returns the complete event's content.
        """
        final = "Could not determine an answer."
        async for event in self.run(user_input, **kwargs):
            if event["type"] == "complete":
                final = event.get("content", final)
        return final

    async def _store_to_memory(self, messages: List[Message]) -> None:
        """Store messages to memory backend if configured."""
        if self.memory:
            try:
                await self.memory.add_messages(messages)
            except Exception as e:
                self.logger.warning("Failed to store messages to memory: %s", e)

    def _should_reflect(
        self,
        turn: int,
        tool_turns_since_reflection: int,
        reflections_done: int,
        confidence_level: Optional[str],
        is_pre_finalization: bool = False,
    ) -> bool:
        """Pure logic: decide whether to run a reflection pass."""
        policy = self.reflection
        if policy.mode == ReflectionMode.NEVER:
            return False
        if reflections_done >= policy.max_reflections:
            return False

        if policy.mode == ReflectionMode.BEFORE_FINAL:
            return is_pre_finalization

        if policy.mode == ReflectionMode.EVERY_N_TURNS:
            if is_pre_finalization:
                return False
            return tool_turns_since_reflection >= policy.every_n_turns

        if policy.mode == ReflectionMode.ON_LOW_CONFIDENCE:
            if confidence_level is None:
                return False
            try:
                conf_val = float(confidence_level)
            except (ValueError, TypeError):
                return False
            return conf_val < policy.confidence_threshold

        return False

    @staticmethod
    def _parse_reflection_verdict(content: str) -> tuple:
        """Parse LLM reflection response into (is_ready, explanation).

        Fails open: unparseable content is treated as READY.
        """
        if not content:
            return (True, "")
        text = content.strip()
        if text.upper().startswith("READY"):
            return (True, "")
        if text.upper().startswith("NEED_MORE"):
            explanation = text.split(":", 1)[1].strip() if ":" in text else ""
            return (False, explanation)
        # Fail open — unparseable means ready
        return (True, "")

    async def _execute_reflection(
        self, messages: List[Message], **provider_kwargs
    ) -> tuple:
        """Run one reflection pass (non-streaming).

        Returns (is_ready, explanation).
        """
        prompt = self.reflection.custom_prompt or _REFLECTION_PROMPT
        reflection_messages = list(messages) + [
            Message(MessageRole.USER, prompt)
        ]
        response = await self.llm_provider.chat(
            [m.to_dict() for m in reflection_messages],
            tools=None,
            enable_thinking=False,
            **provider_kwargs,
        )
        return self._parse_reflection_verdict(response.content or "")

    async def _execute_reflection_streaming(
        self, messages: List[Message], **provider_kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run one reflection pass (streaming variant).

        Yields phase_start, reflection_result, phase_end events.
        Reflection tokens are NOT streamed to user.
        """
        yield {"type": "phase_start", "phase": "reflection"}

        prompt = self.reflection.custom_prompt or _REFLECTION_PROMPT
        reflection_messages = list(messages) + [
            Message(MessageRole.USER, prompt)
        ]
        content_accumulator = ""
        async for chunk in self.llm_provider.stream(
            [m.to_dict() for m in reflection_messages],
            tools=None,
            enable_thinking=False,
            **provider_kwargs,
        ):
            if chunk.content:
                content_accumulator += chunk.content

        is_ready, explanation = self._parse_reflection_verdict(content_accumulator)
        yield {
            "type": "reflection_result",
            "is_ready": is_ready,
            "explanation": explanation,
        }
        yield {"type": "phase_end", "phase": "reflection"}

    @trace_agent(name="AgentSoul")
    async def run(
        self,
        user_input,
        user_id: str = None,
        session_id: str = None,
        enable_thinking: bool = False,
        enable_streaming: bool = False,
        **provider_kwargs
    ):
        """
        Main agent execution loop.
        Returns either:
            - dict: if enable_streaming=False
            - async generator of dict events: if enable_streaming=True
        """

        # Merge default provider kwargs with caller-provided ones
        effective_kwargs = {**self.default_provider_kwargs, **provider_kwargs}
        provider_kwargs = effective_kwargs

        if isinstance(user_input, str):
            user_prompt = user_input
            messages = [
                Message(MessageRole.SYSTEM, self.system_prompt),
                Message(MessageRole.USER, user_prompt)
            ]
            user_question = user_input
            self.logger.debug("user_input is string")

        elif isinstance(user_input, list):
            messages = [
                Message(MessageRole.SYSTEM, self.system_prompt)
            ] + [
                Message(
                    MessageRole(msg["role"]) if "role" in msg else MessageRole.USER,
                    msg.get("content", "")
                )
                for msg in user_input
            ]

            user_question = next(
                (m["content"] for m in reversed(user_input) if m.get("role") == "user"), ""
            )

            user_prompt = _PROMPT_TEMPLATE.format(query_text=user_question)
            messages[-1] = Message(MessageRole.USER, user_prompt)

            self.logger.debug("user_input is list with %d messages", len(user_input))

        else:
            raise ValueError(
                "user_message must be either a string or a list of {'role','content'} dicts")

        # --- Memory: inject context from previous runs ---
        if self.memory:
            try:
                context = await self.memory.get_context_messages(query=user_question)
                if context:
                    # Insert after system prompt, before user message
                    messages = [messages[0]] + context + messages[1:]
            except Exception as e:
                self.logger.warning("Failed to retrieve memory context: %s", e)

        if self.tool_service:
            tool_descriptions = self.tool_service.get_tool_descriptions()
            tool_list = self.tool_service.get_tool_names()
            self.logger.debug("Tool descriptions: %s", tool_descriptions)
            self.logger.debug("Tool list: %s", tool_list)
        else:
            tool_descriptions = []
            tool_list = []

        # --- Strategy-based tool presentation ---
        if tool_descriptions and not self.tool_strategy.should_pass_tools_to_api():
            # Code-act mode: inject tool signatures into system prompt
            tool_sigs = self.tool_strategy.format_tools_for_prompt(tool_descriptions)
            prompt_addition = self.tool_strategy.get_system_prompt_addition()
            system_content = (
                self.system_prompt
                + "\n\n## Available Tools\n"
                + tool_sigs
                + prompt_addition
            )
            messages[0] = Message(MessageRole.SYSTEM, system_content)
            tool_descriptions_for_api = []  # Don't pass to provider
        else:
            tool_descriptions_for_api = tool_descriptions

        agent_timestamp = time.strftime("%Y%m%d_%H%M%S")
        agent_history = []

        if enable_streaming and self.llm_provider.supports_streaming():
            # --- STREAMING MODE ---
            async for event in self._run_streaming(
                agent_timestamp, user_question, messages, tool_list,
                tool_descriptions_for_api, agent_history, enable_thinking, **provider_kwargs
            ):
                if event.get("type") == "complete":
                    await self._store_to_memory(messages)
                yield event
            return

        # --- NON-STREAMING MODE ---
        else:
            result = await self._run_non_streaming(
                agent_timestamp, user_question, messages, tool_list,
                tool_descriptions_for_api, agent_history, enable_thinking, **provider_kwargs
            )
            await self._store_to_memory(messages)
            yield {"type": "complete", "content": result["final_answer"],
                   "history": result.get("history", []), "usage": result.get("usage", {})}
            return

    async def _run_streaming(
        self,
        agent_timestamp: str,
        user_question: str,
        messages: List[Message],
        tool_list: List[str],
        tool_descriptions: List[Dict[str, Any]],
        agent_history: List[Dict],
        enable_thinking: bool,
        **provider_kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Streaming mode execution with per-turn tracing"""

        called_functions = set()
        consecutive_reasoning_turns = 0
        global_start_time = time.perf_counter()
        accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        reflections_done = 0
        tool_turns_since_reflection = 0

        for turn in range(self.max_turns):
            self.logger.info(f"[START] Start streaming turn {turn + 1}/{self.max_turns}")

            # Reset per-turn state
            content_accumulator = ""
            tool_calls_buffer = []
            perplexity = None
            confidence_level = None
            turn_break_sent = False

            cond_is_last_turn = (turn == self.max_turns - 1)

            async for event in self._execute_streaming_turn(
                turn=turn,
                messages=messages,
                tool_descriptions=tool_descriptions,
                enable_thinking=enable_thinking,
                model_id=self.model_id,
                **provider_kwargs,
            ):
                if event["type"] == "token":
                    content_accumulator += event["content"]
                    if turn > 0 and not turn_break_sent:
                        yield {"type": "token", "content": "\n\n"}
                        turn_break_sent = True
                    # Yield tokens as they come
                    yield event
                elif event["type"] == "turn_metadata":
                    # Capture metadata
                    tool_calls_buffer = event.get("tool_calls", [])
                    perplexity = event.get("perplexity")
                    confidence_level = event.get("confidence_level")
                    # Accumulate usage across turns
                    turn_usage = event.get("usage")
                    if turn_usage and isinstance(turn_usage, dict):
                        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            accumulated_usage[key] += turn_usage.get(key, 0)


            if enable_thinking and "<think>" in content_accumulator:
                thoughts = re.findall(
                    r"<think>(.*?)</think>", content_accumulator, re.DOTALL
                )
                response_without_thoughts = re.sub(
                    r"<think>.*?</think>", "", content_accumulator, flags=re.DOTALL
                )
            else:
                thoughts = None
                response_without_thoughts = content_accumulator

            # Create history entry
            turn_record = {
                "turn_number": turn + 1,
                "thought": thoughts,
                "tool_calls": [],
                "called_functions": [],
                "response": response_without_thoughts,
                "perplexity": perplexity,
                "confidence_level": confidence_level
            }
            agent_history.append(turn_record)

            # ─────────────────────────────
            # CODE-ACT: parse tool calls from text if strategy requires it
            # ─────────────────────────────
            if not tool_calls_buffer and not self.tool_strategy.should_pass_tools_to_api():
                try:
                    tool_names_set = set(tool_list)
                    parsed = self.tool_strategy.parse_tool_calls(
                        content_accumulator, tool_names_set,
                        **({"tool_service": self.tool_service}
                           if hasattr(self.tool_strategy, '_last_exec_results') else {})
                    )
                    if parsed:
                        tool_calls_buffer = parsed
                except ValueError as e:
                    self.logger.warning(f"Code-act parse error: {e}")
                    messages.append(Message(
                        MessageRole.USER,
                        f"Error in your code: {e}. Please only call the available tool functions."
                    ))

            # ─────────────────────────────
            # TOOL EXECUTION (if any)
            # ─────────────────────────────
            if tool_calls_buffer:
                consecutive_reasoning_turns = 0
                self.logger.info(f"[TOOLS] Executing {len(tool_calls_buffer)} tool(s)")

                # -- Pre-execution gate (generic hook) --
                if self.tool_pre_execute:
                    for _tc in tool_calls_buffer:
                        _args = _tc.arguments if isinstance(_tc.arguments, dict) else {}
                        gate_result = await self.tool_pre_execute(_tc.name, _args)
                        if gate_result is not None:
                            yield gate_result
                            yield {
                                "type": "complete",
                                "content": gate_result.get("message", "Confirmation required."),
                                "history": agent_history,
                                "messages": [m.to_dict() for m in messages],
                                "usage": accumulated_usage,
                                "interrupted": True,
                            }
                            return

