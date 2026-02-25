"""
MCP Server Builder - Fixed for Codex Compatibility
===================================================

Fixed initialization response to properly advertise capabilities.
"""

import inspect
import json
import asyncio
import logging
import uuid
from typing import Any, Callable, Optional, Union, get_type_hints, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio
import sys
from agentsoul.utils.schema import generate_tool_schema, python_type_to_json_schema

logger = logging.getLogger(__name__)


def log(tag, *msg):
    """Log to stderr to avoid polluting stdout in stdio mode"""
    logger.debug("[%s] %s", tag, " ".join(str(m) for m in msg))


class Transport(ABC):
    """Abstract base class for MCP transports."""

    @abstractmethod
    async def start(self, mcp_server_wrapper):
        """Start the transport layer with the given MCP server wrapper."""
        pass


class StdioTransport(Transport):
    """Standard input/output transport for MCP."""

    def __init__(self):
        """Initialize stdio transport."""
        pass

    async def start(self, mcp_server_wrapper):
        """Start the stdio transport."""
        log("StdioTransport", "Starting stdio server...")

        # Use the MCP SDK's run method which handles stdio properly
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            log("StdioTransport", "Connected to stdin/stdout")
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
                    # Only include results that aren't None (notifications return None)
                    if result is not None:
                        results.append(result)

                # Create response - don't send empty array for notifications
                if not results:
                    # This was just notifications, return 202 Accepted
                    response = Response(status_code=202)
                else:
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

                    log("MCP-HANDLE", f"[OK] Initialize successful!")
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
                        f"[OK] tools/list: Returning {len(tools_list)} tools")
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
                            f"Tool {tool_name} returned: {output}")
                        # log("MCP-HANDLE",
                        #     f"Tool {tool_name} returned: {output[:100]}...")

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
                    log("MCP-HANDLE", f"[OK] Received notification: {method}")
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

        @app.get("/.well-known/oauth-authorization-server")
        async def oauth_discovery():
            """OAuth discovery endpoint - signals no auth required."""
            return Response(status_code=404)

        @app.get("/.well-known/oauth-authorization-server/mcp")
        async def oauth_discovery_mcp():
            """OAuth discovery for MCP endpoint - signals no auth required."""
            return Response(status_code=404)

        @app.get("/mcp/.well-known/oauth-authorization-server")
        async def oauth_discovery_mcp_prefix():
            """OAuth discovery with MCP prefix - signals no auth required."""
            return Response(status_code=404)

        @app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "sessions": len(self.sessions),
                "sse_connections": len(self.sse_connections),
                "tools": len(mcp_server_wrapper._tools)
            }

        # Run the server
        logger.info(
            "MCP Server running at http://%s:%s | "
            "Streamable HTTP: %s | Legacy SSE: %s | CORS: %s",
            self.host, self.port, self.mcp_endpoint,
            self.legacy_sse_endpoint,
            "enabled" if self.enable_cors else "disabled",
        )

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

    def __init__(self, name: str, version: str = "1.0.0"):
        """Initialize an MCP server."""
        self.name = name
        self.version = version
        self.server = Server(name)
        self._tools = {}
        self._setup_handlers()

    def _setup_handlers(self):
        """Set up the core MCP handlers."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List all registered tools."""
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
            """Execute a tool with given arguments."""
            if name not in self._tools:
                raise ValueError(f"Unknown tool: {name}")

            tool_info = self._tools[name]
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

                return [TextContent(type="text", text=output)]

            except Exception as e:
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
        return generate_tool_schema(func)

    def _python_type_to_json_schema(self, py_type: type) -> dict:
        """Map Python types to JSON schema dicts, including items for arrays."""
        return python_type_to_json_schema(py_type)

    # Keep backward compat for any external callers
    def _python_type_to_json_type(self, py_type: type) -> str:
        """Map Python types to JSON schema type strings (legacy)."""
        return self._python_type_to_json_schema(py_type).get("type", "string")

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

    def run(self, transport: Transport = None):
        """Start the MCP server with specified transport."""
        import sys

        if transport is None:
            transport = StdioTransport()

        logger.info(
            "Starting %s v%s | %d tools: %s | Transport: %s",
            self.name, self.version, len(self._tools),
            ", ".join(self._tools.keys()),
            transport.__class__.__name__,
        )

        asyncio.run(transport.start(self))


# Example Usage
if __name__ == "__main__":
    import sys

    server = MCPServer("my_server")

    @server.tool()
    def calculate_sum(a: float, b: float) -> str:
        """Calculate sum of two numbers"""
        result = a + b
        return f"The sum of {a} and {b} is {result}"

    @server.tool()
    def get_user_info(user_id: str) -> str:
        """Get user information"""
        return f"User {user_id}: John Doe (john@example.com)"

    @server.tool()
    def search_database(query: str, limit: int = 10) -> str:
        """Search database"""
        results = [f"Result {i} for '{query}'" for i in range(
            1, min(limit + 1, 4))]
        return f"Found {len(results)} results: {results}"

    if len(sys.argv) > 1 and sys.argv[1] == "http":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        server.run(HTTPTransport(host="0.0.0.0", port=port))
    else:
        server.run(StdioTransport())
