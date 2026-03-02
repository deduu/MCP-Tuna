from __future__ import annotations

import uuid
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Any, Optional

from .handlers.request_parser import RequestParser
from .handlers.model_router import ModelRouteDecider, ModelRoutingError, ModelSelector
from .handlers.tool_selector import ToolSelector
from .handlers.agent_executor import AgentExecutor, StreamHandler
from .models.request import AgentContext
from .responses.builder import ResponseBuilder
from .utils.timing import TimingContext
from .utils.exception import AgentExecutionError
from shared.diagnostics import (
    trace_id_var,
    emit_request_start,
    emit_request_end,
    emit_error,
)
import logging

logger = logging.getLogger("chat.api")


class ChatAPIOrchestrator:
    def __init__(self, model_router: Optional[Any] = None):
        self.parser = RequestParser()
        self.route_decider = ModelRouteDecider(model_router)
        self.tool_selector = ToolSelector()
        self.model_selector = ModelSelector()
        self.agent_executor = AgentExecutor()
        self.response_builder = ResponseBuilder()
        self.stream_handler = StreamHandler(self.response_builder)

    async def prepare_context(self, request: Request) -> AgentContext:

        chat_request = await self.parser.parse_openai_chat_completion(request)

        route_result, enable_logging = self.route_decider.decide_route_and_enable_logging(
            chat_request.user_prompt,
            chat_request.model_name
        )

        enable_thinking = self.route_decider.should_enable_thinking(
            route_result['route'])
        selected_tools = self.tool_selector.select_tools(
            route_result['route'],
            chat_request.selected_tools
        )
        mcp_servers = self.tool_selector.select_mcp_servers(
            route_result['route']
        )

        logger.info(
            f"Model route decision: {route_result['route']}. "
            f"Enable thinking: {enable_thinking}. "
            f"MCP servers: {[s['server_label'] for s in (mcp_servers or [])]}"
        )

        model_id = await self.model_selector.select_model(
            chat_request.model_name,
            route_result
        )
        return AgentContext(
            chat_request=chat_request,
            model_name=chat_request.model_name,
            model_id=model_id,
            enable_thinking=enable_thinking,
            selected_tools=selected_tools,
            mcp_servers=mcp_servers,
            route=route_result['route'],
            enable_logging=enable_logging
        )

    async def handle_request(self, request: Request):
        """Handle the complete chat completion request."""
        timing = TimingContext()
        trace_id_var.set(str(uuid.uuid4()))

        try:
            # Prepare context
            context = await self.prepare_context(request)
            chat_req = context.chat_request

            await emit_request_start(
                user_prompt_preview=chat_req.user_prompt[:200],
                model_name=chat_req.model_name,
                route=context.route,
                stream=chat_req.stream,
                selected_tools=context.selected_tools or [],
            )

            # Create agent (wired to MCP servers when available)
            agent = await self.agent_executor.create_agent(
                model_name=context.model_name,
                mcp_servers=context.mcp_servers,
                api_key=chat_req.api_key,
                base_url=chat_req.base_url,
            )

            # Handle non-streaming
            if not chat_req.stream:
                content = await self.agent_executor.run_non_streaming(
                    agent,
                    chat_req.truncated_messages,
                    context.enable_thinking,
                    timing
                )

                response = self.response_builder.build_openai_chat_completion(
                    content["content"],
                    context.model_name,
                    content.get("usage", {})
                )

                await emit_request_end(
                    status="ok",
                    total_time_s=timing.total_time,
                    inference_time_s=timing.inference_time,
                    first_token_latency_s=timing.first_token_latency,
                    token_usage=content.get("usage", {}),
                )

                return JSONResponse(content=response)

            # Handle streaming
            return StreamingResponse(
                self.stream_handler.stream_agent_response(
                    agent,
                    chat_req.truncated_messages,
                    context.model_name,
                    context.enable_thinking,
                    timing
                ),
                media_type="text/event-stream"
            )

        except ModelRoutingError as e:
            await emit_error(
                error_type=type(e).__name__,
                message=str(e),
                component="chat_api",
                traceback_snippet=traceback.format_exc().splitlines()[-3:],
            )
            await emit_request_end(
                status="error",
                total_time_s=timing.total_time,
                inference_time_s=timing.inference_time,
                first_token_latency_s=timing.first_token_latency,
                token_usage={},
                error=str(e),
            )
            return self.response_builder.build_openai_error_response(e, 500)
        except AgentExecutionError as e:
            await emit_error(
                error_type=type(e).__name__,
                message=str(e),
                component="chat_api",
                traceback_snippet=traceback.format_exc().splitlines()[-3:],
            )
            await emit_request_end(
                status="error",
                total_time_s=timing.total_time,
                inference_time_s=timing.inference_time,
                first_token_latency_s=timing.first_token_latency,
                token_usage={},
                error=str(e),
            )
            return self.response_builder.build_openai_error_response(e, 500)
        except Exception as e:
            logger.error(f"🚨 Fatal error in /v1/chat/completions: {e}")
            traceback.print_exc()
            await emit_error(
                error_type=type(e).__name__,
                message=str(e),
                component="chat_api",
                traceback_snippet=traceback.format_exc().splitlines()[-3:],
            )
            await emit_request_end(
                status="error",
                total_time_s=timing.total_time,
                inference_time_s=timing.inference_time,
                first_token_latency_s=timing.first_token_latency,
                token_usage={},
                error=str(e),
            )
            return self.response_builder.build_openai_error_response(e, 500)
