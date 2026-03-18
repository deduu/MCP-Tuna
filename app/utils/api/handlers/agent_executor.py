from __future__ import annotations

import traceback
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
from ..responses.builder import ResponseBuilder
from ..utils.exception import AgentExecutionError
from ..utils.timing import TimingContext
from shared.costing import estimate_cost_usd
from shared.diagnostics import (
    emit_agent_event,
    emit_request_end,
    emit_error,
)

from app.core.agent_factory import create_agent
import logging

logger = logging.getLogger("api.chat")


class AgentExecutor:
    """Handles agent creation and execution."""

    @staticmethod
    async def create_agent(
        model_name: str,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Create an agent with the specified configuration, optionally wired to MCP servers."""
        logger.info(
            f"Creating agent | model={model_name} | "
            f"mcp_servers={[s.get('server_label') for s in (mcp_servers or [])]}"
        )

        agent = await create_agent(
            model_name=model_name,
            mcp_servers=mcp_servers,
            api_key=api_key,
            base_url=base_url,
        )

        logger.debug(f"Agent created successfully: {agent}")
        return agent

    @staticmethod
    async def run_non_streaming(
        agent,
        messages: List[Dict[str, Any]],
        enable_thinking: bool,
        timing: TimingContext
    ) -> str:
        """Execute agent in non-streaming mode and return final content."""
        logger.info("⚙️ Running agent (non-streaming mode)")
        timing.start_inference()

        result = None
        async for event in agent.run(
            user_input=messages,
            enable_thinking=enable_thinking,
            enable_streaming=False
        ):
            if event["type"] == "complete":
                result = event
                break

        timing.end_inference()

        if not result:
            raise AgentExecutionError("Agent did not return a complete event")

        # logger.info(f"Agent result received: {result}")
        _content = result.get("content", "[no content returned]")

        logger.info(
            f"✅ Completed non-streaming chat | "
            f"Δtotal={timing.total_time:.2f}s | "
            f"Δinference={timing.inference_time:.2f}s"
        )

        return result


class StreamHandler:
    """Handles streaming response generation with granular agent events."""

    def __init__(self, response_builder: ResponseBuilder):
        self.response_builder = response_builder

    def _make_agent_event(self, event_type: str, payload: Dict[str, Any]) -> str:
        """Build an SSE line for a custom agent event (non-OpenAI)."""
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    async def stream_agent_response(
        self,
        agent,
        messages: List[Dict[str, Any]],
        model_name: str,
        enable_thinking: bool,
        timing: TimingContext
    ) -> AsyncGenerator[str, None]:
        """Stream agent responses as SSE events with granular agent activity."""
        logger.info("Running agent (streaming mode)")
        timing.start_inference()

        # Track tool call metadata between turn_metadata and exec events
        _pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        _current_turn = 0
        _last_confidence = None
        _last_perplexity = None
        _tool_call_count = 0
        _total_tool_time_ms = 0.0
        _tool_names: list[str] = []

        try:
            async for event in agent.run(
                user_input=messages,
                enable_streaming=True,
                enable_thinking=enable_thinking,
            ):
                event_type = event["type"]
                content = event.get("content", "")

                # --- Content tokens ---
                if event_type == "token":
                    chunk = self.response_builder.build_openai_chat_chunk(
                        content, model_name)
                    yield f"data: {json.dumps(chunk)}\n\n"

                elif event_type == "final_token":
                    timing.record_first_token()
                    chunk = self.response_builder.build_openai_chat_chunk(
                        content, model_name)
                    yield f"data: {json.dumps(chunk)}\n\n"

                # --- Turn metadata (tool calls, confidence, usage) ---
                elif event_type == "turn_metadata":
                    _current_turn += 1
                    tool_calls = event.get("tool_calls", [])
                    for tc in tool_calls:
                        _pending_tool_calls[tc.name] = {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    _last_confidence = event.get("confidence_level")
                    _last_perplexity = event.get("perplexity")

                    # Emit per-turn metrics
                    usage = event.get("usage")
                    yield self._make_agent_event("metrics", {
                        "turn": _current_turn,
                        "confidence": _last_confidence,
                        "perplexity": _last_perplexity,
                        "tokens": event.get("total_tokens"),
                        "usage": {
                            "prompt_tokens": usage.prompt_tokens if usage else 0,
                            "completion_tokens": usage.completion_tokens if usage else 0,
                        } if usage else None,
                    })

                # --- Tool execution start ---
                elif event_type == "tool_exec_start":
                    tool_name = event.get("tool", "")
                    tc_info = _pending_tool_calls.get(tool_name, {})
                    import time as _time
                    _pending_tool_calls.setdefault(tool_name, {})
                    _pending_tool_calls[tool_name]["_start"] = _time.perf_counter()

                    yield self._make_agent_event("tool_start", {
                        "tool": tool_name,
                        "arguments": tc_info.get("arguments", {}),
                    })
                    await emit_agent_event(
                        event_type, payload={"tool": tool_name},
                    )

                # --- Tool execution end ---
                elif event_type == "tool_exec_end":
                    tool_name = event.get("tool", "")
                    tc_info = _pending_tool_calls.pop(tool_name, {})
                    import time as _time
                    start = tc_info.get("_start", _time.perf_counter())
                    duration_ms = (_time.perf_counter() - start) * 1000
                    _tool_call_count += 1
                    _total_tool_time_ms += duration_ms
                    if tool_name and tool_name not in _tool_names:
                        _tool_names.append(tool_name)

                    yield self._make_agent_event("tool_end", {
                        "tool": tool_name,
                        "duration_ms": round(duration_ms, 1),
                    })
                    await emit_agent_event(
                        event_type, payload={"tool": tool_name},
                    )

                # --- Phase transitions ---
                elif event_type in ("phase_start", "phase_end"):
                    phase = event.get("phase", "")
                    yield self._make_agent_event("phase", {
                        "phase": phase,
                        "action": "start" if event_type == "phase_start" else "end",
                    })
                    await emit_agent_event(
                        event_type, payload={"phase": phase},
                    )

                # --- Thinking (extracted from content in thinking turns) ---
                elif event_type == "thinking":
                    yield self._make_agent_event("thinking", {
                        "content": content,
                    })

                # --- Reflection result ---
                elif event_type == "reflection_result":
                    yield self._make_agent_event("reflection", {
                        "is_ready": event.get("is_ready"),
                        "explanation": event.get("explanation"),
                    })
                    await emit_agent_event(
                        event_type,
                        payload={
                            "is_ready": event.get("is_ready"),
                            "explanation": event.get("explanation"),
                        },
                    )

                # --- Confirmation needed (pre-execution gate) ---
                elif event_type == "confirmation_needed":
                    yield self._make_agent_event("confirmation_needed", {
                        "tool": event.get("tool", ""),
                        "arguments": event.get("arguments", {}),
                        "message": event.get("message", ""),
                    })

                # --- Completion ---
                elif event_type == "complete":
                    timing.end_inference()
                    usage = event.get("usage", {}) or {}
                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or (prompt_tokens + completion_tokens))
                    decode_seconds = timing.inference_time - timing.first_token_latency
                    output_tps = (
                        round(completion_tokens / decode_seconds, 3)
                        if completion_tokens > 0 and decode_seconds > 0
                        else None
                    )
                    estimated_cost = estimate_cost_usd(getattr(agent, "model_id", model_name), usage)
                    summary_metrics = {
                        "prompt_tokens": prompt_tokens or None,
                        "completion_tokens": completion_tokens or None,
                        "total_tokens": total_tokens or None,
                        "latency_ms": round(timing.inference_time * 1000, 1) if timing.inference_time > 0 else None,
                        "ttft_ms": round(timing.first_token_latency * 1000, 1) if timing.first_token_latency > 0 else None,
                        "output_tokens_per_second": output_tps,
                        "estimated_cost_usd": round(estimated_cost, 6) if estimated_cost is not None else None,
                        "confidence": _last_confidence,
                        "perplexity": _last_perplexity,
                        "tool_call_count": _tool_call_count,
                        "tool_time_ms": round(_total_tool_time_ms, 1) if _total_tool_time_ms > 0 else 0.0,
                        "tool_names": _tool_names,
                    }

                    # Send final history with full turn details
                    history = event.get("history", [])
                    yield self._make_agent_event("complete", {
                        "turn_count": len(history),
                        "history": history,
                        "usage": usage,
                        "metrics": summary_metrics,
                        "model_id": getattr(agent, "model_id", model_name),
                        "interrupted": event.get("interrupted", False),
                    })

                    await emit_agent_event(
                        "complete",
                        payload={
                            "turn_count": len(history),
                            "total_usage": event.get("usage", {}),
                        },
                    )
                    await emit_request_end(
                        status="ok",
                        total_time_s=timing.total_time,
                        inference_time_s=timing.inference_time,
                        first_token_latency_s=timing.first_token_latency,
                        token_usage=event.get("usage", {}),
                    )
                    logger.info(
                        f"Completed streaming chat | "
                        f"total={timing.inference_time:.2f}s | "
                        f"first_token={timing.first_token_latency:.2f}s"
                    )
                    yield "data: [DONE]\n\n"
                    return

            # Safety fallback
            timing.end_inference()
            logger.info(
                f"Stream closed normally | total={timing.inference_time:.2f}s")
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception(f"Stream error: {e}")
            await emit_error(
                error_type=type(e).__name__,
                message=str(e),
                component="agent",
                traceback_snippet=traceback.format_exc().splitlines()[-3:],
            )
            error_chunk = self.response_builder.build_openai_chat_chunk(
                f"[ERROR] {str(e)}",
                model_name,
                finish_reason="stop"
            )
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"
