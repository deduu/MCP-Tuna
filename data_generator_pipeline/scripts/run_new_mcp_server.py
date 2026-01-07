"""
MCP Server Builder - Fixed StdioTransport
==========================================

Uses the MCP SDK's built-in stdio handling correctly.
"""

from ..services.pipeline_service import PipelineService
import inspect
import json
import asyncio
import uuid
import logging
import warnings
from typing import Any, Callable, Optional, get_type_hints, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import datetime
import sys
import os
import argparse
import yaml
from dotenv import load_dotenv
from src.agent_framework.providers.openai import OpenAIProvider

load_dotenv(override=True)  # override system env if needed
model = os.getenv("OPENAI_MODEL", "gpt-4o")
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_API_BASE")


def log(tag, *msg):
    """Log to stderr to avoid polluting stdout in stdio mode"""
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] [{tag}] ",
          *msg, file=sys.stderr)
    sys.stderr.flush()  # Ensure logs are written immediately


class Transport(ABC):
    """Abstract base class for MCP transports."""

    @abstractmethod
    async def start(self, mcp_server_wrapper):
        """Start the transport layer with the given MCP server wrapper."""
        pass


class StdioTransport(Transport):
    async def start(self, mcp_server_wrapper):
        from mcp.server.stdio import stdio_server

        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.CRITICAL)

        log("StdioTransport", "Starting...")

        async with stdio_server() as (read_stream, write_stream):
            await mcp_server_wrapper.server.run(
                read_stream,
                write_stream,
                mcp_server_wrapper.server.create_initialization_options()
            )


@dataclass
class SSEConnection:
    """SSE connection tracking."""
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class Session:
    """MCP session data."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    initialized: bool = False
    client_info: Optional[Dict[str, Any]] = None
    sse_queue: Optional[asyncio.Queue] = None


class HTTPTransport(Transport):
    """
    HTTP transport with support for both:
    - New Streamable HTTP (single /mcp endpoint)
    - Legacy HTTP+SSE (separate /sse and /message endpoints)
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        mcp_endpoint: str = "/mcp",
        legacy_sse_endpoint: str = "/sse",
        legacy_message_endpoint: str = "/message",
        enable_cors: bool = True
    ):
        """Initialize HTTP transport with dual support."""
        self.host = host
        self.port = port
        self.mcp_endpoint = mcp_endpoint
        self.legacy_sse_endpoint = legacy_sse_endpoint
        self.legacy_message_endpoint = legacy_message_endpoint
        self.enable_cors = enable_cors
        self.sessions: Dict[str, Session] = {}
        self.sse_connections: Dict[str, SSEConnection] = {}

    async def start(self, mcp_server_wrapper):
        """Start the HTTP transport with dual protocol support."""
        try:
            from fastapi import FastAPI, Request, Response, Header
            from fastapi.responses import StreamingResponse
            from sse_starlette.sse import EventSourceResponse
            import uvicorn
        except ImportError:
            raise ImportError(
                "HTTP transport requires: pip install fastapi uvicorn sse-starlette"
            )

        app = FastAPI(title="MCP Server", version="1.0.0")

        def add_cors_headers(response: Response):
            """Add CORS headers if enabled."""
            if self.enable_cors:
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = (
                    "Content-Type, Authorization, Mcp-Session-Id"
                )
                response.headers["Access-Control-Expose-Headers"] = "Mcp-Session-Id"
            return response

        def get_or_create_session(session_id: Optional[str] = None) -> Session:
            """Get existing session or create new one."""
            if session_id and session_id in self.sessions:
                return self.sessions[session_id]

            session = Session()
            self.sessions[session.id] = session
            return session

        # ==================== LEGACY HTTP+SSE TRANSPORT ====================

        @app.get(self.legacy_sse_endpoint)
        async def legacy_sse_endpoint_handler(request: Request):
            """Legacy SSE endpoint - establishes persistent connection."""
            log("LEGACY-SSE", "Legacy SSE CONNECTED")
            connection = SSEConnection()
            self.sse_connections[connection.session_id] = connection

            async def event_generator():
                try:
                    # Send endpoint event (required by old protocol)
                    yield {
                        "event": "endpoint",
                        "data": json.dumps({
                            "endpoint": self.legacy_message_endpoint
                        })
                    }

                    # Stream messages from queue
                    while True:
                        if await request.is_disconnected():
                            break

                        try:
                            message = await asyncio.wait_for(
                                connection.queue.get(),
                                timeout=30.0
                            )
                            yield {
                                "event": "message",
                                "data": json.dumps(message)
                            }
                        except asyncio.TimeoutError:
                            yield {
                                "event": "heartbeat",
                                "data": json.dumps({"timestamp": asyncio.get_event_loop().time()})
                            }

                except asyncio.CancelledError:
                    pass
                finally:
                    if connection.session_id in self.sse_connections:
                        del self.sse_connections[connection.session_id]

            response = EventSourceResponse(event_generator())
            response.headers["X-Session-Id"] = connection.session_id
            return add_cors_headers(response)

        @app.post(self.legacy_message_endpoint)
        async def legacy_message_endpoint_handler(
            request: Request,
            x_session_id: Optional[str] = Header(None)
        ):
            """Legacy message endpoint - receives client messages."""
            try:
                message = await request.json()
                connection = self.sse_connections.get(
                    x_session_id) if x_session_id else None
                result = await self._handle_message(mcp_server_wrapper, message, connection)

                if connection:
                    await connection.queue.put(result)
                    response = Response(status_code=202)
                else:
                    response = Response(
                        content=json.dumps(result),
                        media_type="application/json"
                    )

                return add_cors_headers(response)

            except Exception as e:
                response = Response(
                    content=json.dumps({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        },
                        "id": None
                    }),
                    status_code=500,
                    media_type="application/json"
                )
                return add_cors_headers(response)

        # ==================== NEW STREAMABLE HTTP TRANSPORT ====================

        @app.options(self.mcp_endpoint)
        async def mcp_options_handler():
            """Handle CORS preflight for new endpoint."""
            response = Response()
            return add_cors_headers(response)

        @app.post(self.mcp_endpoint)
        async def mcp_post_handler(
            request: Request,
            mcp_session_id: Optional[str] = Header(None)
        ):
            """New Streamable HTTP POST endpoint."""
            try:
                log("HTTP-POST", f"POST /mcp session={mcp_session_id}")
                body = await request.json()
                log("HTTP-POST", "Body received:", body)
                session = get_or_create_session(mcp_session_id)

                # Handle single message or batch
                is_batch = isinstance(body, list)
                messages = body if is_batch else [body]

                # Process messages
                results = []
                for message in messages:
                    result = await self._handle_message(mcp_server_wrapper, message, None)
                    results.append(result)

                # Create response
                response_data = results if is_batch else results[0]
                response = Response(
                    content=json.dumps(response_data),
                    media_type="application/json"
                )

                response.headers["Mcp-Session-Id"] = session.id
                return add_cors_headers(response)

            except json.JSONDecodeError:
                response = Response(
                    content=json.dumps({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        },
                        "id": None
                    }),
                    status_code=400,
                    media_type="application/json"
                )
                return add_cors_headers(response)
            except Exception as e:
                log("HTTP-POST", f"Error: {e}")
                response = Response(
                    content=json.dumps({
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": f"Internal error: {str(e)}"
                        },
                        "id": None
                    }),
                    status_code=500,
                    media_type="application/json"
                )
                return add_cors_headers(response)

        @app.get(self.mcp_endpoint)
        async def mcp_get_handler(
            request: Request,
            mcp_session_id: Optional[str] = Header(None)
        ):
            """Streamable HTTP GET endpoint - keeps connection alive."""
            log("HTTP-GET",
                f"GET /mcp (Streamable HTTP) session={mcp_session_id}")
            session = get_or_create_session(mcp_session_id)

            async def event_generator():
                try:
                    while True:
                        if await request.is_disconnected():
                            break

                        await asyncio.sleep(30)
                        yield {"comment": "keepalive"}

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log("HTTP-GET", f"Error in SSE: {e}")

            response = EventSourceResponse(event_generator())
            response.headers["Mcp-Session-Id"] = session.id
            return add_cors_headers(response)

        @app.delete(self.mcp_endpoint)
        async def mcp_delete_handler(
            request: Request,
            mcp_session_id: Optional[str] = Header(None)
        ):
            """DELETE endpoint - allows client to explicitly close SSE connection."""
            log("HTTP-DELETE", f"DELETE /mcp session={mcp_session_id}")

            # Clean up session if it exists
            if mcp_session_id and mcp_session_id in self.sessions:
                del self.sessions[mcp_session_id]
                log("HTTP-DELETE", f"Cleaned up session {mcp_session_id}")

            response = Response(status_code=200)
            return add_cors_headers(response)

        # ==================== SHARED HANDLERS ====================
        async def _handle_message(
            mcp_server_wrapper,
            message: dict,
            sse_connection: Optional[SSEConnection]
        ) -> dict:
            """Handle a single JSON-RPC message."""
            method = message.get("method")
            params = message.get("params", {})
            msg_id = message.get("id")
            log("MCP-HANDLE",
                f"📨 Full message: {json.dumps(message, indent=2)}")
            log("MCP-HANDLE", f"Received method={method} id={msg_id}")

            try:
                if method == "initialize":
                    client_caps = params.get("capabilities", {})
                    client_protocol = params.get(
                        "protocolVersion", "2025-06-18")

                    # Build server capabilities - EXACTLY per spec format
                    server_caps = {}

                    # Only advertise capabilities we actually support
                    if len(mcp_server_wrapper._tools) > 0:
                        server_caps["tools"] = {"listChanged": False}

                    # Always include empty objects for other capabilities
                    server_caps["resources"] = {"listChanged": False}
                    server_caps["prompts"] = {"listChanged": False}

                    result = {
                        "protocolVersion": client_protocol,
                        "capabilities": server_caps,
                        "serverInfo": {
                            "name": mcp_server_wrapper.name,
                            "version": mcp_server_wrapper.version
                        }
                    }

                    log("MCP-HANDLE", f"✅ Initialize successful!")
                    log("MCP-HANDLE",
                        f"   Protocol: {result['protocolVersion']}")
                    log("MCP-HANDLE",
                        f"   Server: {result['serverInfo']['name']} v{result['serverInfo']['version']}")
                    log("MCP-HANDLE",
                        f"   Tools available: {len(mcp_server_wrapper._tools)}")
                    log("MCP-HANDLE",
                        f"   Capabilities sent: {json.dumps(result['capabilities'], indent=2)}")

                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": result
                    }

                elif method == "tools/list":
                    # Access tools directly from our wrapper
                    tools_list = [
                        {
                            "name": name,
                            "description": tool_info['description'],
                            "inputSchema": tool_info['schema']
                        }
                        for name, tool_info in mcp_server_wrapper._tools.items()
                    ]

                    result = {
                        "tools": tools_list
                    }

                    log("MCP-HANDLE",
                        f"✅ tools/list: Returning {len(tools_list)} tools")
                    for tool in tools_list:
                        log("MCP-HANDLE",
                            f"   - {tool['name']}: {tool['description']}")

                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": result
                    }

                elif method == "tools/call":
                    tool_name = params.get("name")
                    arguments = params.get("arguments", {})

                    log("MCP-HANDLE",
                        f"Calling tool: {tool_name} with args: {arguments}")

                    # Execute the tool directly from our wrapper
                    if tool_name not in mcp_server_wrapper._tools:
                        raise ValueError(f"Unknown tool: {tool_name}")

                    tool_info = mcp_server_wrapper._tools[tool_name]
                    func = tool_info['func']

                    try:
                        # Call the function (handle both sync and async)
                        if inspect.iscoroutinefunction(func):
                            result = await func(**arguments)
                        else:
                            result = func(**arguments)

                        # Format the result
                        if isinstance(result, str):
                            output = result
                        else:
                            output = json.dumps(result, indent=2)

                        log("MCP-HANDLE",
                            f"Tool {tool_name} returned: {output[:100]}...")

                        return {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": {
                                "content": [
                                    {"type": "text", "text": output}
                                ]
                            }
                        }

                    except Exception as e:
                        log("MCP-HANDLE", f"Tool {tool_name} error: {e}")
                        return {
                            "jsonrpc": "2.0",
                            "id": msg_id,
                            "result": {
                                "content": [
                                    {"type": "text",
                                        "text": f"Error executing {tool_name}: {str(e)}"}
                                ]
                            }
                        }
                elif method == "resources/list":
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "resources": []
                        }
                    }

                elif method == "prompts/list":
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "result": {
                            "prompts": []
                        }
                    }

                elif method.startswith("notifications/"):
                    # Notifications don't expect a response per JSON-RPC spec
                    # But we must acknowledge them properly
                    log("MCP-HANDLE", f"✅ Received notification: {method}")
                    # Return None to signal "no response needed"
                    return None

                else:
                    return {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }

            except Exception as e:
                log("MCP-HANDLE", f"Error: {e}")
                import traceback
                traceback.print_exc()
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    }
                }

        self._handle_message = _handle_message

        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "sessions": len(self.sessions),
                "sse_connections": len(self.sse_connections),
                "tools": len(mcp_server_wrapper._tools)
            }

        # Run the server - print startup info to stderr for consistency
        log("HTTP", "=" * 70)
        log("HTTP", f"🚀 MCP Server running at: http://{self.host}:{self.port}")
        log("HTTP", "=" * 70)
        log("HTTP", "")
        log("HTTP", "📡 Endpoints:")
        log("HTTP",
            f"   • New Streamable HTTP: http://localhost:{self.port}{self.mcp_endpoint}")
        log("HTTP",
            f"   • Legacy SSE:          http://localhost:{self.port}{self.legacy_sse_endpoint}")
        log("HTTP",
            f"   • Legacy Messages:     http://localhost:{self.port}{self.legacy_message_endpoint}")
        log("HTTP",
            f"   • Health Check:        http://localhost:{self.port}/health")
        log("HTTP", "")
        log("HTTP", "✨ Features:")
        log("HTTP",
            f"   • CORS: {'enabled' if self.enable_cors else 'disabled'}")
        log("HTTP", f"   • Backward compatible with old MCP clients")
        log("HTTP", f"   • Forward compatible with new Streamable HTTP clients")
        log("HTTP", "")
        log("HTTP",
            f"💡 For new clients, use: http://localhost:{self.port}{self.mcp_endpoint}")
        log("HTTP",
            f"   For old clients, use: http://localhost:{self.port}{self.legacy_sse_endpoint}")
        log("HTTP", "")
        log("HTTP", "=" * 70)

        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server_instance = uvicorn.Server(config)
        await server_instance.serve()


class MCPServer:
    """Modular MCP server builder."""

    def __init__(self, llm_provider, config: Dict[str, Any], name: str, version: str = "1.0.0", ):
        """Initialize an MCP server."""
        self.name = name
        self.version = version
        self.server = Server(name)
        self._tools = {}
        self.service = PipelineService(llm_provider, config)

        self._register_tools()
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up the core MCP handlers."""

        # Keep a reference to self for closures
        server_instance = self

        @self.server.list_resources()
        async def list_resources():
            """List available resources (empty for now)."""
            log("HANDLER", "list_resources called")
            return []

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all registered tools."""
            log("HANDLER",
                f"list_tools called, returning {len(server_instance._tools)} tools")
            tools = [
                Tool(
                    name=name,
                    description=tool_info['description'],
                    inputSchema=tool_info['schema']
                )
                for name, tool_info in server_instance._tools.items()
            ]
            log("HANDLER", f"Tools: {[t.name for t in tools]}")
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Execute a tool with given arguments."""
            log("HANDLER", f"call_tool called: {name} with {arguments}")
            if name not in server_instance._tools:
                raise ValueError(f"Unknown tool: {name}")

            tool_info = server_instance._tools[name]
            func = tool_info['func']

            try:
                if inspect.iscoroutinefunction(func):
                    result = await func(**arguments)
                else:
                    result = func(**arguments)

                if isinstance(result, str):
                    output = result
                else:
                    output = json.dumps(result, indent=2)

                log("HANDLER", f"Tool {name} returned: {output[:100]}")
                return [TextContent(type="text", text=output)]

            except Exception as e:
                log("HANDLER", f"Tool {name} error: {e}")
                return [TextContent(
                    type="text",
                    text=f"Error executing {name}: {str(e)}"
                )]

    def tool(self, name: Optional[str] = None, description: Optional[str] = None):
        """Decorator to register a function as an MCP tool."""
        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool_desc = description or (
                inspect.getdoc(func) or "No description")
            schema = self._generate_schema(func)

            self._tools[tool_name] = {
                'func': func,
                'description': tool_desc,
                'schema': schema
            }

            return func

        return decorator

    def _generate_schema(self, func: Callable) -> dict[str, Any]:
        """Generate JSON schema from function signature."""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue

            param_type = type_hints.get(param_name, Any)
            json_type = self._python_type_to_json_type(param_type)

            properties[param_name] = {
                "type": json_type,
                "description": f"Parameter: {param_name}"
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "object",
            "properties": properties
        }

        if required:
            schema["required"] = required

        return schema

    def _python_type_to_json_type(self, py_type: type) -> str:
        """Map Python types to JSON schema types."""
        type_map = {
            int: "integer",
            float: "number",
            str: "string",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        origin = getattr(py_type, '__origin__', None)
        if origin is not None:
            return type_map.get(origin, "string")

        return type_map.get(py_type, "string")

    def add_tool(self, func: Callable, name: Optional[str] = None,
                 description: Optional[str] = None):
        """Programmatically add a tool."""
        tool_name = name or func.__name__
        tool_desc = description or (inspect.getdoc(func) or "No description")
        schema = self._generate_schema(func)

        self._tools[tool_name] = {
            'func': func,
            'description': tool_desc,
            'schema': schema
        }

    def _register_tools(self):
        """Register all tools with the MCP server."""
        # First, populate the _tools dictionary
        self._tools = {
            "load_document": {
                "func": self.service.load_document,
                "description": "Load and parse a document file (PDF, Markdown, etc.)",
                "schema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document file"
                        }
                    },
                    "required": ["file_path"]
                }
            },
            "generate_from_page": {
                "func": self.service.generate_from_page,
                "description": "Generate fine-tuning data from a single page",
                "schema": {
                    "type": "object",
                    "properties": {
                        "technique": {
                            "type": "string",
                            "enum": ["sft", "dpo", "grpo"],
                            "description": "Fine-tuning technique"
                        },
                        "page_text": {
                            "type": "string",
                            "description": "Text content of the page"
                        },
                        "page_index": {
                            "type": "integer",
                            "description": "Index of the page"
                        },
                        "file_name": {
                            "type": "string",
                            "description": "Name of the source file"
                        },
                        "custom_template": {
                            "type": "string",
                            "description": "Optional custom prompt template"
                        }
                    },
                    "required": ["technique", "page_text", "page_index", "file_name"]
                }
            },
            "generate_from_document": {
                "func": self.service.generate_from_document,
                "description": "Generate fine-tuning data from an entire document",
                "schema": {
                    "type": "object",
                    "properties": {
                        "technique": {
                            "type": "string",
                            "enum": ["sft", "dpo", "grpo"],
                            "description": "Fine-tuning technique"
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Path to the document file"
                        },
                        "custom_template": {
                            "type": "string",
                            "description": "Optional custom prompt template"
                        },
                        "start_page": {
                            "type": "integer",
                            "description": "Start page (optional)"
                        },
                        "end_page": {
                            "type": "integer",
                            "description": "End page (optional)"
                        }
                    },
                    "required": ["technique", "file_path"]
                }
            },
            "generate_batch": {
                "func": self.service.generate_batch,
                "description": "Generate fine-tuning data from multiple documents",
                "schema": {
                    "type": "object",
                    "properties": {
                        "technique": {
                            "type": "string",
                            "enum": ["sft", "dpo", "grpo"],
                            "description": "Fine-tuning technique"
                        },
                        "file_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of document file paths"
                        },
                        "custom_template": {
                            "type": "string",
                            "description": "Optional custom prompt template"
                        }
                    },
                    "required": ["technique", "file_paths"]
                }
            },
            "export_dataset": {
                "func": self.service.export_dataset,
                "description": "Export generated dataset to file",
                "schema": {
                    "type": "object",
                    "properties": {
                        "data_points": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Array of data points to export"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Path for output file"
                        },
                        "format": {
                            "type": "string",
                            "enum": ["json", "jsonl", "excel", "csv", "huggingface"],
                            "description": "Output format"
                        }
                    },
                    "required": ["data_points", "output_path", "format"]
                }
            },
            "list_techniques": {
                "func": self.service.list_techniques,
                "description": "List all available fine-tuning techniques",
                "schema": {
                    "type": "object",
                    "properties": {}
                }
            },
            "get_technique_schema": {
                "func": self.service.get_technique_schema,
                "description": "Get the data schema for a specific technique",
                "schema": {
                    "type": "object",
                    "properties": {
                        "technique": {
                            "type": "string",
                            "description": "Technique name"
                        }
                    },
                    "required": ["technique"]
                }
            }
        }

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name=name,
                    description=tool_info['description'],
                    inputSchema=tool_info['schema']
                )
                for name, tool_info in self._tools.items()
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            """Handle tool calls."""
            if name not in self._tools:
                raise ValueError(f"Unknown tool: {name}")

            tool_info = self._tools[name]
            func = tool_info['func']

            if inspect.iscoroutinefunction(func):
                result = await func(**arguments)
            else:
                result = func(**arguments)

            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2) if not isinstance(
                    result, str) else result
            )]

    def run(self, transport: Transport = None):
        """Start the MCP server with specified transport."""
        if transport is None:
            transport = StdioTransport()

        # All logging goes to stderr for both stdio and HTTP
        log("SERVER", "=" * 60)
        log("SERVER", f"Starting {self.name} v{self.version}")
        log("SERVER",
            f"Registered {len(self._tools)} tools: {', '.join(self._tools.keys())}")
        log("SERVER", f"Transport: {transport.__class__.__name__}")
        log("SERVER", "=" * 60)

        asyncio.run(transport.start(self))


# Example Usage
if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Run Fine-tuning MCP Server")
    parser.add_argument(
        "--config",
        type=str,
        default="./AgentY/data_generator_pipeline/scripts/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "transport",
        nargs="?",
        default="stdio",
        choices=["stdio", "http"],
        help="Transport type: stdio or http"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transport"
    )

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup LLM provider
    # llm = OpenAIProvider(
    #     model=config["model"],
    #     api_key=os.getenv("OPENAI_API_KEY"),
    # )
    llm = OpenAIProvider(model=model, api_key=api_key, base_url=base_url)

    server = MCPServer(llm_provider=llm, name="my_server", config=config)

    print("🚀 Fine-tuning MCP Server starting...", file=sys.stderr)
    print("📝 Available tools:", file=sys.stderr)
    print("   - load_document", file=sys.stderr)
    print("   - generate_from_page", file=sys.stderr)
    print("   - generate_from_document", file=sys.stderr)
    print("   - generate_batch", file=sys.stderr)
    print("   - export_dataset", file=sys.stderr)
    print("   - list_techniques", file=sys.stderr)
    print("   - get_technique_schema", file=sys.stderr)
    print("\n✨ Server ready for connections!", file=sys.stderr)

    if len(sys.argv) > 1 and sys.argv[1] == "http":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        server.run(HTTPTransport(host="0.0.0.0", port=port))
    else:
        server.run(StdioTransport())
