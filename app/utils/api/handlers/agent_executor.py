from __future__ import annotations

import traceback
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
from ..responses.builder import ResponseBuilder
from ..utils.exception import AgentExecutionError
from ..utils.timing import TimingContext
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
        content = result.get("content", "[no content returned]")

        logger.info(
            f"✅ Completed non-streaming chat | "
            f"Δtotal={timing.total_time:.2f}s | "
            f"Δinference={timing.inference_time:.2f}s"
        )

        return result


class StreamHandler:
    """Handles streaming response generation."""

    def __init__(self, response_builder: ResponseBuilder):
        self.response_builder = response_builder

    async def stream_agent_response(
        self,
        agent,
        messages: List[Dict[str, Any]],
        model_name: str,
        enable_thinking: bool,
        timing: TimingContext
    ) -> AsyncGenerator[str, None]:
        """Stream agent responses as SSE events."""
        logger.info("⚙️ Running agent (streaming mode)")
        timing.start_inference()

        try:
            async for event in agent.run(
                user_input=messages,
                enable_streaming=True,
                enable_thinking=enable_thinking,
            ):
                event_type = event["type"]
                content = event.get("content", "")

                # Handle thinking tokens
                if event_type == "token":
                    logger.debug(f"Thinking stream: {content}")
                    chunk = self.response_builder.build_openai_chat_chunk(
                        content, model_name)
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Handle final answer tokens
                elif event_type == "final_token":
                    timing.record_first_token()
                    chunk = self.response_builder.build_openai_chat_chunk(
                        content, model_name)
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Handle tool execution events
                elif event_type in ("tool_exec_start", "tool_exec_end"):
                    logger.debug(f"tool_call: {content}")
                    await emit_agent_event(
                        event_type,
                        payload={"tool": event.get("name")},
                    )

                # Handle pipeline phase events
                elif event_type in ("phase_start", "phase_end"):
                    await emit_agent_event(
                        event_type,
                        payload={"phase": event.get("phase")},
                    )

                # Handle reflection result events
                elif event_type == "reflection_result":
                    await emit_agent_event(
                        event_type,
                        payload={
                            "is_ready": event.get("is_ready"),
                            "explanation": event.get("explanation"),
                        },
                    )

                # Handle completion
                elif event_type == "complete":
                    timing.end_inference()
                    await emit_agent_event(
                        "complete",
                        payload={
                            "turn_count": len(event.get("history", [])),
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
                        f"✅ Completed streaming chat | "
                        f"Δtotal={timing.inference_time:.2f}s | "
                        f"Δfirst_token={timing.first_token_latency:.2f}s"
                    )
                    yield "data: [DONE]\n\n"
                    return

            # Safety fallback
            timing.end_inference()
            logger.info(
                f"Stream closed normally | Δtotal={timing.inference_time:.2f}s")
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
