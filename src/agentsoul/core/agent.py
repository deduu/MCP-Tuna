import asyncio
import json
import time
import re
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
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

                # Check if exec strategy already executed tools
                exec_results = getattr(self.tool_strategy, '_last_exec_results', [])

                if exec_results:
                    # CodeActExecStrategy: tools already executed, just append results
                    messages.append(
                        self.message_formatter.create_assistant_message(
                            content_accumulator, tool_calls=[]
                        )
                    )
                    for entry in exec_results:
                        tc = entry["tool_call"]
                        yield {"type": "tool_exec_start", "tool": tc.name}
                        self._append_code_act_result(
                            messages, tc, entry["result"], turn_record
                        )
                        yield {"type": "tool_exec_end", "tool": tc.name}
                        called_functions.add(tc.name)
                    self.tool_strategy._last_exec_results = []

                    # ─── SINGLE-SHOT OPTIMIZATION ───
                    single_shot_result = getattr(self.tool_strategy, '_final_result', None)
                    if single_shot_result is not None:
                        self.logger.info(
                            "[SINGLE-SHOT] Captured 'result' from sandbox, "
                            "skipping finalization turn"
                        )
                        answer = self._build_single_shot_answer(
                            content_accumulator, single_shot_result
                        )
                        self.tool_strategy._final_result = None

                        self._log_messages_snapshot(turn, messages)

                        global_end_time = time.perf_counter()
                        total_duration = global_end_time - global_start_time
                        self.logger.warning(
                            f"[DONE] Agent finished: Single-shot. "
                            f"Total duration: {total_duration:.3f}s"
                        )

                        yield {
                            "type": "complete",
                            "content": answer,
                            "history": agent_history,
                            "messages": [m.to_dict() for m in messages],
                            "usage": accumulated_usage,
                        }
                        return
                    # ─── END SINGLE-SHOT ───
                else:
                    # Normal flow: execute tools via ToolService
                    messages.append(
                        self.message_formatter.create_assistant_message(
                            content_accumulator,
                            tool_calls_buffer
                        )
                    )
                    if self.parallel_tool_calls and len(tool_calls_buffer) > 1:
                        for tc in tool_calls_buffer:
                            yield {"type": "tool_exec_start", "tool": tc.name}
                        await self._execute_tools_parallel(
                            tool_calls_buffer, user_question, messages, turn_record, called_functions
                        )
                        for tc in tool_calls_buffer:
                            yield {"type": "tool_exec_end", "tool": tc.name}
                    else:
                        for tool_call in tool_calls_buffer:
                            yield {"type": "tool_exec_start", "tool": tool_call.name}
                            await self._execute_tool_call(
                                tool_call, user_question, messages, turn_record, called_functions
                            )
                            yield {"type": "tool_exec_end", "tool": tool_call.name}
                            called_functions.add(tool_call.name)

                self._log_messages_snapshot(turn, messages)

                # --- Reflection after tool execution (EVERY_N_TURNS / ON_LOW_CONFIDENCE) ---
                tool_turns_since_reflection += 1
                if self._should_reflect(
                    turn, tool_turns_since_reflection, reflections_done,
                    confidence_level, is_pre_finalization=False,
                ):
                    self.logger.info("[REFLECT] Running reflection pass (post-tool, streaming)")
                    async for refl_event in self._execute_reflection_streaming(
                        messages, **provider_kwargs
                    ):
                        if refl_event.get("type") == "reflection_result":
                            turn_record["reflection"] = {
                                "is_ready": refl_event["is_ready"],
                                "explanation": refl_event["explanation"],
                            }
                        yield refl_event
                    reflections_done += 1
                    tool_turns_since_reflection = 0

                if not cond_is_last_turn:
                    # Inject synthesis instruction so the next LLM turn
                    # produces a comprehensive answer from the tool data.
                    # Remove any previous synthesis instruction to avoid accumulation.
                    messages[:] = [m for m in messages if not (
                        m.role == MessageRole.SYSTEM and str(m.content).startswith("Now answer the user")
                    )]
                    messages.append(Message(
                        MessageRole.SYSTEM,
                        _SYNTHESIS_INSTRUCTION,
                    ))
                    continue
                else:
                    should_finalize = True
            else:
                # ─────────────────────────────
                # NO TOOL CALLS IN THIS TURN
                # ─────────────────────────────
                should_finalize = False

                if content_accumulator.strip():
                    consecutive_reasoning_turns += 1

                    MAX_CONSECUTIVE_REASONING = 3
                    if consecutive_reasoning_turns > MAX_CONSECUTIVE_REASONING:
                        self.logger.warning(
                            f"[WARN] Model produced {consecutive_reasoning_turns} consecutive "
                            f"reasoning turns without tool calls. Forcing finalization."
                        )
                        should_finalize = True
                    else:
                        messages.append(
                            self.message_formatter.create_assistant_message(
                                content_accumulator,
                                tool_calls=[]
                            )
                        )

                # ─────────────────────────────
                # CHECK IF FINALIZATION NEEDED
                # ─────────────────────────────
                if not should_finalize:
                    should_finalize = (
                        cond_is_last_turn and response_without_thoughts.strip() == ""
                    ) or (response_without_thoughts.strip() == "")

            if should_finalize:
                # --- Reflection before finalization (BEFORE_FINAL, streaming) ---
                if self._should_reflect(
                    turn, tool_turns_since_reflection, reflections_done,
                    confidence_level, is_pre_finalization=True,
                ):
                    self.logger.info("[REFLECT] Running reflection pass (pre-finalization, streaming)")
                    refl_is_ready = True
                    refl_explanation = ""
                    async for refl_event in self._execute_reflection_streaming(
                        messages, **provider_kwargs
                    ):
                        if refl_event.get("type") == "reflection_result":
                            refl_is_ready = refl_event["is_ready"]
                            refl_explanation = refl_event["explanation"]
                            turn_record["reflection"] = {
                                "is_ready": refl_is_ready,
                                "explanation": refl_explanation,
                            }
                        yield refl_event
                    reflections_done += 1
                    if not refl_is_ready:
                        self.logger.info(
                            "[REFLECT] NEED_MORE: %s -- continuing loop",
                            refl_explanation,
                        )
                        messages.append(Message(
                            MessageRole.USER,
                            f"Your self-review found issues: {refl_explanation}\n"
                            "Please address these before providing a final answer."
                        ))
                        continue

                self.logger.info("[FINALIZE] Entering answer-only finalization pass")

                messages.append(Message(
                    MessageRole.SYSTEM,
                    _FINALIZATION_INSTRUCTION,
                ))

                yield {"type": "phase_start", "phase": "final_answer"}

                # Finalization creates its own span
                final_accumulator = ""
                async for event in self._execute_finalization_turn(
                    messages=messages,
                    enable_thinking=False,
                    **provider_kwargs
                ):
                    if event.get("type") == "token" and event.get("content"):
                        final_accumulator += event["content"]
                    yield event

                yield {"type": "phase_end", "phase": "final_answer"}

                messages.append(
                    self.message_formatter.create_assistant_message(
                        final_accumulator,
                        tool_calls=[]
                    )
                )

                final_answer = self._extract_final_answer(final_accumulator)
                global_end_time = time.perf_counter()
                total_duration = global_end_time - global_start_time

                self.logger.warning(
                    f"[DONE] Agent finished: Answer Finalized. Total duration: {total_duration:.3f}s"
                )

                self._log_messages_snapshot(turn, messages)

                yield {
                    "type": "complete",
                    "content": final_answer,
                    "history": agent_history,
                    "messages": [m.to_dict() for m in messages],
                    "usage": accumulated_usage,
                }
                return
            else:
                # Direct extraction, no second call needed
                self.logger.info("[OK] No finalization needed -- extracting answer directly")

                final_answer = self._extract_final_answer(content_accumulator)
                global_end_time = time.perf_counter()
                total_duration = global_end_time - global_start_time

                self.logger.warning(
                    f"[DONE] Agent finished: No Tool Called. Total duration: {total_duration:.3f}s"
                )

                self._log_messages_snapshot(turn, messages)

                yield {
                    "type": "complete",
                    "content": final_answer,
                    "history": agent_history,
                    "messages": [m.to_dict() for m in messages],
                    "usage": accumulated_usage,
                }
                return

        # Max turns reached
        global_end_time = time.perf_counter()
        total_duration = global_end_time - global_start_time
        self.logger.warning(
            f"[DONE] Agent finished: Max turns reached. Total duration: {total_duration:.3f}s"
        )
        yield {
            "type": "complete",
            "content": "The agent could not determine a final answer within the turn limit.",
            "history": agent_history,
            "usage": accumulated_usage,
        }

    @trace_llm(name="LLMCall")
    async def _execute_streaming_turn(
        self,
        turn: int,
        messages: List[Message],
        tool_descriptions: List[Dict[str, Any]],
        enable_thinking: bool,
        model_id: str = None,
        **provider_kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute one streaming turn - creates ONE span per turn.
        """
        start_time = time.perf_counter()
        first_token_time = None
        total_tokens = 0

        content_accumulator = ""
        tool_calls_buffer: List[ToolCall] = []
        perplexity = None
        confidence_level = None
        usage = None

        async for chunk in self.llm_provider.stream(
            [m.to_dict() for m in messages],
            tools=tool_descriptions,
            enable_thinking=enable_thinking,
            **provider_kwargs
        ):
            # Estimate token count from chunk content length (~4 chars/token)
            if chunk.content or chunk.tool_calls:
                if chunk.content:
                    total_tokens += max(1, len(chunk.content) // 4)
                else:
                    total_tokens += 1
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    ttft = first_token_time - start_time
                    self.logger.info(f"[PERF] TTFT = {ttft:.3f} s")

            if chunk.content:
                # Clean and accumulate
                if "</think>" in chunk.content and "</think>" in content_accumulator:
                    content_accumulator += chunk.content
                    continue

                if "<final_answer>" in chunk.content or "</final_answer>" in chunk.content:
                    chunk.content = self._clean_final_tags(chunk.content)

                content_accumulator += chunk.content

                yield {"type": "token", "content": chunk.content}

            if chunk.tool_calls:
                tool_calls_buffer.extend(chunk.tool_calls)

            if hasattr(chunk, "perplexity") and chunk.perplexity is not None:
                perplexity = chunk.perplexity
                confidence_level = chunk.confidence_level

            if hasattr(chunk, "usage") and chunk.usage:
                usage = chunk.usage
                self.logger.debug("Captured usage: %s", usage)
        # Yield metadata at the end for the caller to process
        yield {
            "type": "turn_metadata",
            "tool_calls": tool_calls_buffer,
            "perplexity": perplexity,
            "confidence_level": confidence_level,
            "total_tokens": total_tokens,
            "usage": usage,
        }
        end_time = time.perf_counter()
        metrics = compute_metrics(
            start_time, first_token_time or end_time, end_time, total_tokens
        )
        self.logger.debug("Turn %d metrics: %s", turn + 1, json.dumps(metrics))
        yield {"type": "phase_end", "phase": "llm_turn_end"}



    @trace_llm(name="FinalizationCall")
    async def _execute_finalization_turn(
        self, messages: List[Message], enable_thinking: bool, **provider_kwargs
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute finalization turn - creates ONE span for finalization.
        UPDATED: Returns AsyncGenerator.
        """
        final_accumulator = ""

        async for chunk in self.llm_provider.stream(
            [m.to_dict() for m in messages], tools=None, enable_thinking=False, **provider_kwargs
        ):
            if chunk.content:
                clean_content = self._clean_final_tags(chunk.content)
                if not clean_content:
                    continue

                final_accumulator += clean_content
                yield {"type": "token", "content": clean_content}

    async def _run_non_streaming(
        self,
        agent_timestamp: str,
        user_question: str,
        messages: List[Message],
        tool_list: List[str],
        tool_descriptions: List[Dict[str, Any]],
        agent_history: List[Dict],
        enable_thinking: bool,
        **provider_kwargs
    ) -> Dict[str, Any]:
        """Multi-turn loop for non-streaming responses with per-turn tracing"""

        called_functions = set()
        consecutive_reasoning_turns = 0
        reflections_done = 0
        tool_turns_since_reflection = 0

        for turn in range(self.max_turns):
            self.logger.debug(f"\n--- Turn {turn + 1}/{self.max_turns} ---")

            # Execute one turn (creates one LLM span)
            response = await self._execute_non_streaming_turn(
                messages=messages,
                tool_descriptions=tool_descriptions,
                enable_thinking=enable_thinking,
                **provider_kwargs
            )

            turn_record = {
                "turn_number": turn + 1,
                "thought": response.thinking if enable_thinking else None,
                "tool_calls": [],
                "called_functions": [],
                "response": response.content,
                "perplexity": response.perplexity,
                "confidence_level": response.confidence_level
            }
            agent_history.append(turn_record)

            tool_calls = response.tool_calls or []

            # CODE-ACT: parse tool calls from text if strategy requires it
            if not tool_calls and not self.tool_strategy.should_pass_tools_to_api():
                try:
                    tool_names_set = set(tool_list)
                    parsed = self.tool_strategy.parse_tool_calls(
                        response.content or "", tool_names_set,
                        **({"tool_service": self.tool_service}
                           if hasattr(self.tool_strategy, '_last_exec_results') else {})
                    )
                    if parsed:
                        tool_calls = parsed
                except ValueError as e:
                    self.logger.warning(f"Code-act parse error: {e}")
                    messages.append(Message(
                        MessageRole.USER,
                        f"Error in your code: {e}. Please only call the available tool functions."
                    ))

            if tool_calls:
                consecutive_reasoning_turns = 0
                self.logger.info(f"[TOOLS] Executing {len(tool_calls)} tool(s)")

                # Check if exec strategy already executed tools
                exec_results = getattr(self.tool_strategy, '_last_exec_results', [])

                if exec_results:
                    # CodeActExecStrategy: tools already executed
                    messages.append(
                        self.message_formatter.create_assistant_message(
                            response.content, tool_calls=[]
                        )
                    )
                    for entry in exec_results:
                        tc = entry["tool_call"]
                        self._append_code_act_result(
                            messages, tc, entry["result"], turn_record
                        )
                        called_functions.add(tc.name)
                    self.tool_strategy._last_exec_results = []

                    # ─── SINGLE-SHOT OPTIMIZATION ───
                    single_shot_result = getattr(self.tool_strategy, '_final_result', None)
                    if single_shot_result is not None:
                        self.logger.info(
                            "[SINGLE-SHOT] Captured 'result' from sandbox, "
                            "skipping finalization turn"
                        )
                        answer = self._build_single_shot_answer(
                            response.content, single_shot_result
                        )
                        self.tool_strategy._final_result = None
                        return {
                            "final_answer": answer,
                            "history": agent_history,
                            "usage": response.usage if response.usage else {}
                        }
                    # ─── END SINGLE-SHOT ───
                else:
                    # Normal flow: execute tools via ToolService
                    messages.append(
                        self.message_formatter.create_assistant_message(
                            response.content,
                            tool_calls
                        )
                    )
                    if self.parallel_tool_calls and len(tool_calls) > 1:
                        await self._execute_tools_parallel(
                            tool_calls, user_question, messages, turn_record, called_functions
                        )
                    else:
                        for tool_call in tool_calls:
                            await self._execute_tool_call(
                                tool_call, user_question, messages, turn_record, called_functions
                            )
                            called_functions.add(tool_call.name)

                # --- Reflection after tool execution (EVERY_N_TURNS / ON_LOW_CONFIDENCE) ---
                tool_turns_since_reflection += 1
                if self._should_reflect(
                    turn, tool_turns_since_reflection, reflections_done,
                    response.confidence_level, is_pre_finalization=False,
                ):
                    self.logger.info("[REFLECT] Running reflection pass (post-tool)")
                    is_ready, explanation = await self._execute_reflection(
                        messages, **provider_kwargs
                    )
                    reflections_done += 1
                    tool_turns_since_reflection = 0
                    turn_record["reflection"] = {
                        "is_ready": is_ready, "explanation": explanation
                    }

                if turn < self.max_turns - 1:
                    # Inject synthesis instruction so the next LLM turn
                    # produces a comprehensive answer from the tool data.
                    # Remove any previous synthesis instruction to avoid accumulation.
                    messages[:] = [m for m in messages if not (
                        m.role == MessageRole.SYSTEM and str(m.content).startswith("Now answer the user")
                    )]
                    messages.append(Message(
                        MessageRole.SYSTEM,
                        _SYNTHESIS_INSTRUCTION,
                    ))
                    continue
                else:
                    should_finalize = True
            else:
                # No tool calls
                should_finalize = False

                if response.content.strip():
                    consecutive_reasoning_turns += 1

                    MAX_CONSECUTIVE_REASONING = 3
                    if consecutive_reasoning_turns > MAX_CONSECUTIVE_REASONING:
                        self.logger.warning(
                            f"[WARN] Model produced {consecutive_reasoning_turns} consecutive "
                            f"reasoning turns without tool calls. Forcing finalization."
                        )
                        should_finalize = True
                    else:
                        messages.append(
                            self.message_formatter.create_assistant_message(
                                response.content,
                                tool_calls=[]
                            )
                        )

                should_finalize = should_finalize or (
                    turn == self.max_turns - 1 and response.content.strip() == ""
                ) or (response.content.strip() == "")

            if should_finalize:
                # --- Reflection before finalization (BEFORE_FINAL) ---
                if self._should_reflect(
                    turn, tool_turns_since_reflection, reflections_done,
                    response.confidence_level, is_pre_finalization=True,
                ):
                    self.logger.info("[REFLECT] Running reflection pass (pre-finalization)")
                    is_ready, explanation = await self._execute_reflection(
                        messages, **provider_kwargs
                    )
                    reflections_done += 1
                    turn_record["reflection"] = {
                        "is_ready": is_ready, "explanation": explanation
                    }
                    if not is_ready:
                        self.logger.info(
                            "[REFLECT] NEED_MORE: %s -- continuing loop", explanation
                        )
                        should_finalize = False
                        messages.append(Message(
                            MessageRole.USER,
                            f"Your self-review found issues: {explanation}\n"
                            "Please address these before providing a final answer."
                        ))
                        continue

                self.logger.info("[FINALIZE] Generating answer-only response")
                return await self._finalize_response(
                    messages, response, agent_history, enable_thinking, **provider_kwargs
                )
            else:
                self.logger.info("[OK] No finalization needed -- extracting answer directly")
                final_answer = self._extract_final_answer(response.content)
                return {
                    "final_answer": final_answer,
                    "history": agent_history,
                    "usage": response.usage if response.usage else {}
                }

        # Max turns reached
        self.logger.debug("[DONE] Agent finished: Max turns reached.")
        return {
            "final_answer": "The agent could not determine a final answer within the turn limit.",
            "history": agent_history
        }

    @trace_llm(name="LLMCall")
    async def _execute_non_streaming_turn(
        self,
        messages: List[Message],
        tool_descriptions: List[Dict[str, Any]],
        enable_thinking: bool,
        **provider_kwargs
    ):
        """
        Execute one non-streaming turn - creates ONE span per turn.
        """
        start_time = time.perf_counter()

        response = await self.llm_provider.chat(
            [m.to_dict() for m in messages],
            tools=tool_descriptions,
            enable_thinking=enable_thinking,
            **provider_kwargs
        )

        end_time = time.perf_counter()
        self.logger.debug(
            "Model response received (%d chars) in %.3fs",
            len(response.content or ""),
            end_time - start_time
        )

        return response

    @trace_tool(name="ToolExecution")
    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
        user_question: str,
        messages: List[Message],
        turn_record: Dict = None,
        called_functions: set = None
    ) -> Dict[str, Any]:
        """Execute a tool call and append result as a tool message - creates ONE span per tool call"""
        self.logger.debug(f"[DEBUG] tool_call type: {type(tool_call)}")
        self.logger.debug(f"[DEBUG] tool_call: {tool_call}")

        # Validate tool call arguments
        try:
            self.logger.debug(
                f"[DEBUG] Validating tool call: {tool_call.name} with arguments: {tool_call.arguments}"
            )
            validated_args, warnings = await self.tool_service.validate_tool_call(
                user_question, tool_call.name, tool_call.arguments, called_functions
            )
            self.logger.debug("validated_args: %s", validated_args)
            args = validated_args if isinstance(validated_args, dict) else {}
        except Exception as e:
            args = tool_call.arguments or {}
            warnings = None
            self.logger.error(
                f"[ERROR] Tool call argument validation failed for tool '{tool_call.name}': {e}"
            )

        tool_result = None
        metadata = None
        status = "error"
        error = None

        try:
            # Execute the tool
            tool_result, metadata = await self.tool_service.execute_tool(tool_call.name, **args)

            # Truncate tool result if necessary
            tool_result = await self._truncate_tool_result(tool_result)

            # Append warnings if any
            if warnings:
                self.logger.warning(f"[WARNING] Tool '{tool_call.name}' warnings: {warnings}")
                tool_result += f"\n\n[WARNING] {warnings}"

            self.logger.debug(f"[DEBUG] Executed tool '{tool_call.name}', result length: {len(str(tool_result))} chars")
            self.logger.debug(f"[DEBUG] Tool result preview: {str(tool_result)[:500]}")

            # Use message formatter to create tool result message
            messages.append(
                self.message_formatter.create_tool_result_message(
                    tool_call,
                    tool_result,
                    metadata,
                    warnings
                )
            )

            if turn_record is not None:
                turn_record["tool_calls"].append({
                    "name": tool_call.name,
                    "arguments": args,
                    "result": metadata or tool_result
                })
                turn_record["called_functions"].append(tool_call.name)

            status = "success"
            self.logger.debug("Tool '%s' executed successfully.", tool_call.name)

        except Exception as e:
            self.logger.error("Tool '%s' execution failed: %s", tool_call.name, e)
            tool_result = f"Error: {str(e)}"
            messages.append(
                self.message_formatter.create_tool_result_message(
                    tool_call,
                    tool_result
                )
            )
            if turn_record is not None:
                turn_record["tool_calls"].append({
                    "name": tool_call.name,
                    "arguments": args,
                    "result": tool_result
                })
                turn_record["called_functions"].append(tool_call.name)
            error = str(e)

        return {
            "tool_name": tool_call.name,
            "arguments": args,
            "result": tool_result,
            "metadata": metadata,
            "status": status,
            "error": error
        }

    @trace_tool(name="ToolExecution")
    async def _execute_tool_call_isolated(
        self,
        tool_call: ToolCall,
        user_question: str,
        called_functions: set,
    ) -> Dict[str, Any]:
        """Execute a tool call without appending to messages — returns components for later assembly."""
        self.logger.debug(f"[PARALLEL] Executing tool: {tool_call.name}")

        try:
            validated_args, warnings = await self.tool_service.validate_tool_call(
                user_question, tool_call.name, tool_call.arguments, called_functions
            )
            args = validated_args if isinstance(validated_args, dict) else {}
        except Exception as e:
            args = tool_call.arguments or {}
            warnings = None
            self.logger.error(
                f"[ERROR] Tool call argument validation failed for tool '{tool_call.name}': {e}"
            )

        tool_result = None
        metadata = None

        try:
            tool_result, metadata = await self.tool_service.execute_tool(tool_call.name, **args)
            tool_result = await self._truncate_tool_result(tool_result)

            if warnings:
                self.logger.warning(f"[WARNING] Tool '{tool_call.name}' warnings: {warnings}")
                tool_result += f"\n\n[WARNING] {warnings}"

            self.logger.debug(f"[PARALLEL] Tool '{tool_call.name}' succeeded, result length: {len(str(tool_result))} chars")

            message = self.message_formatter.create_tool_result_message(
                tool_call, tool_result, metadata, warnings
            )
            record = {"name": tool_call.name, "arguments": args, "result": metadata or tool_result}

        except Exception as e:
            self.logger.error("Tool '%s' execution failed: %s", tool_call.name, e)
            tool_result = f"Error: {str(e)}"
            message = self.message_formatter.create_tool_result_message(tool_call, tool_result)
            record = {"name": tool_call.name, "arguments": args, "result": tool_result}

        return {
            "tool_call": tool_call,
            "message": message,
            "record": record,
            "called_name": tool_call.name,
        }

    async def _execute_tools_parallel(
        self,
        tool_calls: List[ToolCall],
        user_question: str,
        messages: List[Message],
        turn_record: Dict,
        called_functions: set,
    ) -> None:
        """Execute multiple tool calls concurrently and append results in original order."""
        tasks = [
            self._execute_tool_call_isolated(tc, user_question, called_functions)
            for tc in tool_calls
        ]
        results = await asyncio.gather(*tasks)
        for res in results:
            messages.append(res["message"])
            turn_record["tool_calls"].append(res["record"])
            turn_record["called_functions"].append(res["called_name"])
            called_functions.add(res["called_name"])

    def _build_single_shot_answer(
        self, llm_content: str, computed_result: str
    ) -> str:
        """Build the final answer for single-shot mode.

        Combines the LLM's natural language text (if meaningful)
        with the computed result from sandbox execution.
        """
        # Remove think tags
        text = re.sub(r"<think>.*?</think>", "", llm_content, flags=re.DOTALL)
        # Remove code blocks
        text = re.sub(r"```python\s*\n.*?```", "", text, flags=re.DOTALL)
        # Remove final_answer tags
        text = self._clean_final_tags(text)
        preamble = text.strip()

        # If the LLM wrote meaningful natural language around the code,
        # include it as context before the result
        if preamble and len(preamble) > 20:
            return f"{preamble}\n\n{computed_result}"

        return computed_result

    def _append_code_act_result(
        self,
        messages: List[Message],
        tool_call: ToolCall,
        result: str,
        turn_record: Dict = None,
    ):
        """Append a pre-executed code-act tool result as a user message."""
        args_str = ", ".join(
            f"{k}={v!r}" for k, v in tool_call.arguments.items()
        )
        formatted = f"# Result of {tool_call.name}({args_str})\n{result}"
        messages.append(Message(MessageRole.USER, formatted))

        if turn_record is not None:
            turn_record["tool_calls"].append({
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "result": result,
            })
            turn_record["called_functions"].append(tool_call.name)

    async def _truncate_tool_result(self, tool_result: Any, max_token_length: int = 15000) -> str:
        """Truncate tool result to fit within max_length characters."""
        result_str = self._handle_tool_result(tool_result)
        try:
            if hasattr(self.llm_provider, '_count_tokens'):
                num_tokens = self.llm_provider._count_tokens(result_str)
            else:
                # Character-based heuristic: ~4 chars per token
                num_tokens = len(result_str) // 4
            if num_tokens > max_token_length:
                self.logger.warning(
                    "Tool result too long (%d tokens, %d chars), truncating to %d tokens.",
                    num_tokens, len(result_str), max_token_length
                )
                avg_token_length = len(result_str) / num_tokens
                max_char_length = int(max_token_length * avg_token_length)
                result_str = result_str[:max_char_length]
                self.logger.debug("Truncated tool result length: %d characters.", len(result_str))
        except Exception:
            return result_str
        return result_str

    def _clean_final_tags(self, content: str) -> str:
        """Remove <final_answer> tags from content"""
        return content.replace("<final_answer>", "").replace("</final_answer>", "")

    def _extract_final_answer(self, content: str) -> str:
        """Extract final answer from content, removing think tags"""
        if not content:
            return "Could not determine an answer."
        # Remove think tags
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        # Remove final_answer tags
        content = self._clean_final_tags(content)
        return content.strip()

    def _log_messages_snapshot(self, turn: int, messages: List[Message]):
        """Log current message state for debugging (count + last 3 messages)"""
        if self.verbose:
            self.logger.debug("Messages after turn %d: %d total", turn + 1, len(messages))
            for msg in messages[-3:]:
                self.logger.debug("  [last] %s: %s...", msg.role, str(msg.content)[:100])

    def _handle_tool_result(self, tool_result: Any) -> str:
        """Convert tool result to string format"""
        if isinstance(tool_result, str):
            return tool_result
        elif isinstance(tool_result, (dict, list)):
            return json.dumps(tool_result, indent=2)
        else:
            return str(tool_result)

    async def _finalize_response(
        self,
        messages: List[Message],
        last_response,
        agent_history: List[Dict],
        enable_thinking: bool,
        **provider_kwargs
    ) -> Dict[str, Any]:
        """Generate final answer-only response"""
        messages.append(Message(
            MessageRole.SYSTEM,
            _FINALIZATION_INSTRUCTION,
        ))

        # This creates its own span
        final_response = await self._execute_non_streaming_turn(
            messages=messages,
            tool_descriptions=[],
            enable_thinking=False,
            **provider_kwargs
        )

        messages.append(
            self.message_formatter.create_assistant_message(
                final_response.content,
                tool_calls=[]
            )
        )

        final_answer = self._extract_final_answer(final_response.content)
        return {
            "final_answer": final_answer,
            "history": agent_history,
            "usage": final_response.usage if final_response.usage else {}
        }

    def _get_provider_specs(self) -> Dict[str, Any]:
        """Get LLM provider capabilities."""
        return {
            "supports_tools": self.llm_provider.supports_tools(),
            "supports_thinking": self.llm_provider.supports_thinking(),
            "supports_streaming": self.llm_provider.supports_streaming() if hasattr(self.llm_provider, "supports_streaming") else False,
            "specs": self.llm_provider.get_specs() if hasattr(self.llm_provider, "get_specs") else {"unknown": True}
        }

    def _log_agent_history_snapshot(self, agent_timestamp: str, turn: int, agent_history: List[Dict]):
        # Log agent history snapshot to the database
        if not self.enable_logging:
            return

        # Ensure run_id exists
        if not self._run_id:
            self._run_id = self.log_repo.ensure_run(
                agent_timestamp, self.model_name, self.model_id, "")

        turn_number = turn + 1
        turn_history = next(
            (h for h in agent_history if h["turn_number"] == turn_number), None)
        if turn_history:
            self.log_repo.upsert_turn_history(
                self._run_id, turn_number, turn_history)
