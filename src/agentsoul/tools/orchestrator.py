""" Modular Agent Framework with separated concerns: - ToolRegistry: Tool normalization and registration - A2ACommunicator: Agent-to-agent communication - AgentSoul: Core orchestration logic """
from typing import Any, Dict, List, Optional, Union, Callable, TYPE_CHECKING
import asyncio
import httpx
import inspect
import itertools
import json
import shlex
from dataclasses import dataclass, field
from enum import Enum

from agentsoul.tools.service import ToolService
from agentsoul.tools.base import BaseToolService
from agentsoul.core.models import MessageRole, Message
from agentsoul.utils.logger import get_logger

# Only import for type checking, not at runtime
if TYPE_CHECKING:
    from agentsoul.core.agent import AgentSoul

# ═══════════════════════════════════════════════════════════
# A2A Communication Module
# ═══════════════════════════════════════════════════════════


class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"
    RESPONSE = "response"
    DELEGATION = "delegation"
    NOTIFICATION = "notification"


class TypeMapper:
    """Maps Python types to JSON Schema types"""

    @staticmethod
    def map_type(annotation) -> str:
        """Enhanced type mapping with support for complex types"""
        if annotation == inspect.Parameter.empty:
            return "string"

        # Handle Optional[T] and Union types
        origin = getattr(annotation, "__origin__", None)
        if origin is Union:
            args = getattr(annotation, "__args__", ())
            # Filter out NoneType for Optional
            non_none = [arg for arg in args if arg is not type(None)]
            if non_none:
                return TypeMapper.map_type(non_none[0])

        # Handle List[T]
        if origin is list:
            return "array"

        # Handle Dict[K,V]
        if origin is dict:
            return "object"

        # Basic types
        mapping = {
            int: "integer",
            float: "number",
            bool: "boolean",
            str: "string",
            dict: "object",
            list: "array",
        }
        return mapping.get(annotation, "string")


class MCPServerConnection:
    """Manages connection to an MCP server and tool execution"""

    def __init__(self, server_config: Dict[str, Any], logger):
        self.server_label = server_config.get("server_label", "unknown")
        self.server_url = server_config.get("server_url")
        self.server_description = server_config.get("server_description", "")
        self.require_approval = server_config.get("require_approval", "never")
        self.logger = logger
        self._tools: Dict[str, Dict] = {}
        self._connected = False
        self._http_client = None
        self._transport_type = None  # 'stdio', 'http', 'sse', or 'streamable_http'
        self._session = None  # Shared session for stdio, sse, and streamable_http
        self._stdio_client = None  # Keep reference to stdio client
        self._lock = asyncio.Lock()  # guard shared session/connection
        self._msg_id_counter = itertools.count(1)  # unique JSON-RPC message IDs

    async def __aenter__(self):
        """Enter the async context manager, establishing the connection."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the async context manager, ensuring cleanup."""
        await self.close()

    async def connect(self):
        """Connect to the MCP server and fetch available tools"""
        if self._connected:
            return

        try:
            self.logger.info(
                f"Connecting to MCP server: {self.server_label} at {self.server_url}")

            # Determine transport type based on URL

            if self.server_url.startswith("stdio://"):
                await self._connect_stdio()
            elif self.server_url.startswith(("http://", "https://")):
                await self._connect_http()
            else:
                raise ValueError(
                    f"Unsupported MCP server URL format: {self.server_url}")

            self._connected = True
            self.logger.info(
                f"Connected to {self.server_label}: found {len(self._tools)} tools")
        except ConnectionError:
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to connect to MCP server {self.server_label}: {e}")
            raise

    async def _connect_stdio(self):
        """Connect via stdio transport"""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            raise ImportError(
                "MCP package required. Install with: pip install mcp")

        self._transport_type = "stdio"

        # Extract and parse the command from stdio:// URL
        command_string = self.server_url.replace("stdio://", "")

        # Use shlex to properly split the command (handles spaces in paths, quotes, etc.)
        command_parts = shlex.split(command_string)

        if not command_parts:
            raise ValueError(f"Invalid stdio URL: {self.server_url}")

        # First part is the command, rest are arguments
        command = command_parts[0]
        args = command_parts[1:] if len(command_parts) > 1 else []

        self.logger.debug(f"stdio command: {command}, args: {args}")

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=None
        )

        # Create client and session without context managers to keep them alive
        self._stdio_client = stdio_client(server_params)
        read, write = await self._stdio_client.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()

        tools_result = await self._session.list_tools()
        for tool in tools_result.tools:
            self._tools[tool.name] = {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema,
            }

    async def _check_reachable(self):
        """Fast TCP-level check to see if the server is listening."""
        from urllib.parse import urlparse
        parsed = urlparse(self.server_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        # On Windows, 'localhost' resolves to IPv6 (::1) first which times
        # out when servers bind to 0.0.0.0 (IPv4 only).  Force IPv4.
        if host in ("localhost", "::1"):
            host = "127.0.0.1"
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=2.0
            )
            writer.close()
            await writer.wait_closed()
        except (OSError, asyncio.TimeoutError) as e:
            raise ConnectionError(
                f"Server {self.server_label} not reachable at {host}:{port}"
            ) from e

    async def _connect_http(self):
        """Connect via HTTP/HTTPS transport with Streamable HTTP support"""
        self._transport_type = "http"
        self._http_client = httpx.AsyncClient(timeout=30.0)

        # Quick reachability check before trying protocols
        await self._check_reachable()

        _CONNECT_TIMEOUT = 10  # seconds – fail fast so we can try the next transport

        try:
            # Try MCP Streamable HTTP (official Microsoft Learn format)
            try:
                await asyncio.wait_for(
                    self._connect_streamable_http(), timeout=_CONNECT_TIMEOUT
                )
                return
            except (Exception, asyncio.TimeoutError) as streamable_error:
                self.logger.warning(
                    f"Streamable HTTP connection failed, trying SSE: {streamable_error}")

            # Try MCP SSE transport
            try:
                await asyncio.wait_for(
                    self._connect_sse(), timeout=_CONNECT_TIMEOUT
                )
                return
            except (Exception, asyncio.TimeoutError) as sse_error:
                self.logger.warning(
                    f"SSE connection failed, trying JSON-RPC MCP: {sse_error}")

            # Fallback to unified JSON-RPC MCP endpoint (/mcp)
            await self._connect_jsonrpc_mcp()
        except httpx.HTTPError as e:
            self.logger.error(f"HTTP request failed: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to parse MCP HTTP response: {e}")
            raise

    async def _connect_streamable_http(self):
        """Connect via Streamable HTTP transport (used by Microsoft Learn MCP)."""
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client
        except ImportError as e:
            # Let caller decide to try SSE / JSON-RPC instead
            raise RuntimeError(
                "MCP streamable-http client not available (install 'mcp[cli]')"
            ) from e

        self._transport_type = "streamable_http"
        self.logger.info(
            f"Attempting Streamable HTTP connection to {self.server_url}")

        async with streamablehttp_client(self.server_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_result = await session.list_tools()

                if not getattr(tools_result, "tools", None):
                    self.logger.warning(
                        f"Streamable HTTP connected but no tools returned from {self.server_url}"
                    )

                for tool in tools_result.tools:
                    self._tools[tool.name] = {
                        "name": tool.name,
                        "description": getattr(tool, "description", "") or "",
                        "inputSchema": getattr(tool, "inputSchema", {}) or {},
                    }

        self.logger.info(
            f"Streamable HTTP connection successful, found {len(self._tools)} tools"
        )
        self._connected = True

    async def _connect_sse(self):
        from mcp import ClientSession
        from mcp.client.sse import sse_client

        self._transport_type = "sse"
        self.logger.info(f"Attempting SSE connection to {self.server_url}")

        try:
            # sse_client must be used with "async with", not "await"
            async with sse_client(self.server_url) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    tools = getattr(tools_result, "tools", None) or []

                    if not tools:
                        self.logger.warning(
                            f"SSE connected to {self.server_url} but returned no tools"
                        )

                    for tool in tools:
                        self._tools[tool.name] = {
                            "name": tool.name,
                            "description": getattr(tool, "description", "") or "",
                            "inputSchema": getattr(tool, "inputSchema", {}) or {},
                        }

            self.logger.info(
                f"SSE connection successful, found {len(self._tools)} tools"
            )

        except Exception as e:
            self.logger.error(f"SSE connection failed: {e}", exc_info=True)
            raise

    async def _connect_jsonrpc_mcp(self):
        """Fallback for MCP servers that expose a single POST /mcp endpoint (e.g. echo.mcp.inevitable.fyi)"""
        self.logger.info(
            f"Attempting JSON-RPC MCP connection to {self.server_url}")

        payload_init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "AgentSoul", "version": "1.0.0"}
            }
        }

        # initialize session
        resp = await self._http_client.post(self.server_url, json=payload_init, timeout=10)
        resp.raise_for_status()
        init_data = resp.json()
        self.logger.debug(f"Initialization OK: {init_data}")

        # list tools
        payload_list = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list"
        }
        resp_tools = await self._http_client.post(self.server_url, json=payload_list, timeout=10)
        resp_tools.raise_for_status()
        data = resp_tools.json()
        tools_data = data.get("result", {}).get("tools", [])

        for tool in tools_data:
            self._tools[tool["name"]] = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "inputSchema": tool.get("inputSchema", {}),
            }

        self.logger.info(
            f"JSON-RPC MCP connection successful, found {len(self._tools)} tools")

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool on the MCP server"""
        # Only hold the lock for the connection check, not during execution
        async with self._lock:
            if not self._connected:
                await self.connect()

        if tool_name not in self._tools:
            msg = f"Tool '{tool_name}' not found on server {self.server_label}"
            self.logger.warning(msg)
            return {"error": msg, "tool": tool_name}

        try:
            if self._session is not None:
                # Session-based transports (stdio/SSE) need the lock
                # because they share a single session
                async with self._lock:
                    return await self._execute_tool_with_session(
                        tool_name, kwargs, self._session
                    )
            elif self._transport_type == "streamable_http":
                # Streamable HTTP creates a new connection per call — safe to run concurrently
                return await self._execute_tool_streamable_http(tool_name, kwargs)
            elif self._transport_type == "http":
                return await self._execute_tool_http(tool_name, kwargs)
            else:
                raise ValueError(
                    f"Unknown transport type: {self._transport_type}"
                )
        except Exception as e:
            self.logger.error(
                f"Failed to execute MCP tool {tool_name}: {e}", exc_info=True
            )
            return {"error": str(e), "tool": tool_name}

    async def _execute_tool_with_session(self, tool_name: str, arguments: Dict, session) -> Any:
        """Execute tool via MCP session (stdio or SSE)"""
        result = await session.call_tool(tool_name, arguments=arguments)

        if not hasattr(result, "content"):
            return result

        parts = []
        content = getattr(result, "content", None) or []

        for item in content:
            # Most MCP clients use TextContent / JsonContent etc.
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            else:
                # Fallback for JSON / other payloads
                try:
                    parts.append(json.dumps(item.__dict__, default=str))
                except Exception:
                    parts.append(str(item))

        return "\n".join(parts) if parts else str(result)

    async def _execute_tool_streamable_http(self, tool_name: str, arguments: Dict) -> Any:
        """Execute tool via direct JSON-RPC POST.

        Avoids creating a full MCP session (with SSE keepalive) per call,
        which causes httpx.ReadTimeout for long-running tools.
        """
        msg_id = next(self._msg_id_counter)
        payload = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        response = await self._http_client.post(
            self.server_url, json=payload, timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise Exception(f"MCP error: {data['error']}")

        result = data.get("result", {})
        # Extract text from MCP content array
        content_items = result.get("content", [])
        parts = [
            item["text"] for item in content_items
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(parts) if parts else json.dumps(result)

    async def _execute_tool_http(self, tool_name: str, arguments: Dict) -> Any:
        """Execute tool via HTTP transport"""
        msg_id = next(self._msg_id_counter)
        payload = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        response = await self._http_client.post(
            self.server_url, json=payload, timeout=300,
        )
        response.raise_for_status()
        data = response.json()

        if "error" in data:
            raise Exception(f"MCP error: {data['error']}")

        result = data.get("result", {})
        # Extract text from MCP content array
        content_items = result.get("content", [])
        parts = [
            item["text"] for item in content_items
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(parts) if parts else json.dumps(result)

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from this server"""
        return list(self._tools.values())

    async def close(self):
        """Close all connections"""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._session:
            await self._session.__aexit__(None, None, None)
            self._session = None

        if self._stdio_client:
            await self._stdio_client.__aexit__(None, None, None)
            self._stdio_client = None

        self._connected = False


class ToolOrchestrator:
    """Handles tool registration and normalization with MCP support"""

    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger or get_logger(self.__class__.__name__)
        self.type_mapper = TypeMapper()
        self.mcp_connections: Dict[str, MCPServerConnection] = {}

    async def normalize_tools_async(
        self,
        tools: Optional[Union[List[Any], BaseToolService]]
    ) -> Optional[BaseToolService]:
        self.logger.debug(
            "\n[DEBUG] normalize_tools_async called =======================")
        self.logger.debug(f"[DEBUG] tools type: {type(tools)}")

        if tools is None:
            self.logger.debug("[DEBUG] tools is None -> return None")
            return None

        if isinstance(tools, BaseToolService):
            self.logger.debug(
                f"[DEBUG] Detected BaseToolService subclass: {tools.__class__.__name__}"
            )
            return tools

        tool_service = ToolService()

        # If it's already a list, keep as is
        if isinstance(tools, list):
            raw_tools = tools
        else:
            # Accept tuples, sets, generators, etc.
            try:
                raw_tools = list(tools) if not isinstance(
                    tools, (str, bytes)) else [tools]
            except TypeError:
                self.logger.warning(
                    f"[DEBUG] tools is not iterable (type={type(tools)}), wrapping into list"
                )
                raw_tools = [tools]

        # Drop Nones defensively
        tools_list = [t for t in raw_tools if t is not None]

        if not tools_list:
            self.logger.warning(
                "[DEBUG] tools list is empty after normalization")
            return None

        mcp_server_configs = []
        other_tools = []

        for i, tool in enumerate(tools_list):
            self.logger.debug(f"[DEBUG] Analyzing tool[{i}]: {tool}")
            if isinstance(tool, dict) and tool.get("type") == "mcp" and "server_label" in tool:
                self.logger.debug(" [OK] Identified as MCP server config")
                mcp_server_configs.append(tool)
            else:
                other_tools.append(tool)

        # Connect to all MCP servers in parallel
        if mcp_server_configs:
            await asyncio.gather(
                *(self._register_mcp_server(tool_service, config)
                  for config in mcp_server_configs)
            )

        # Register plain callables / schemas
        for i, tool in enumerate(other_tools):
            try:
                if callable(tool) and not isinstance(tool, type):
                    self._register_callable(tool_service, tool)
                elif isinstance(tool, dict) and "name" in tool:
                    self._register_mcp_schema(tool_service, tool)
            except Exception as e:
                self.logger.error(f"[DEBUG] Failed to register tool[{i}]: {e}")

        descs = tool_service.get_tool_descriptions()
        self.logger.info(
            f"[DEBUG] Final ToolService has {len(descs)} tools: {[d.get('name') for d in descs]}"
        )
        return tool_service

    async def _register_mcp_server(self, tool_service: ToolService, server_config: Dict[str, Any]):
        """Connect to MCP server and register all its tools"""
        server_label = server_config.get("server_label", "unknown")

        try:
            connection = MCPServerConnection(server_config, self.logger)
            await connection.connect()

            if not connection.get_tools():
                self.logger.warning(
                    f"MCP server {server_label} connected but returned no tools"
                )

            self.mcp_connections[server_label] = connection

            for tool_info in connection.get_tools():
                await self._register_mcp_tool(tool_service, tool_info, connection)

        except ConnectionError as e:
            self.logger.warning(
                f"[MCP] Server '{server_label}' unavailable: {e}"
            )
        except Exception as e:
            self.logger.error(
                f"[MCP] Failed to register server '{server_label}': {e}", exc_info=True
            )

    async def _register_mcp_tool(
        self,
        tool_service: ToolService,
        tool_info: Dict[str, Any],
        connection: MCPServerConnection
    ):
        """Register a single tool from an MCP server"""
        tool_name = tool_info["name"]

        # Create wrapper function that calls the MCP server
        async def mcp_tool_wrapper(**kwargs):
            return await connection.execute_tool(tool_name, **kwargs)

        # Set proper name and docstring
        mcp_tool_wrapper.__name__ = tool_name
        mcp_tool_wrapper.__doc__ = tool_info.get(
            "description", f"MCP tool: {tool_name}")

        # Convert MCP inputSchema to our tool description format
        input_schema = tool_info.get("inputSchema", {})
        description = {
            "name": tool_name,
            "description": tool_info.get("description", ""),
            "parameters": input_schema
        }

        tool_service.register_tool(
            name=tool_name,
            func=mcp_tool_wrapper,
            description=description)

        self.logger.info(
            f"Registered MCP tool: {tool_name} from {connection.server_label}")

    def _register_callable(self, tool_service: ToolService, func: Callable):
        """Register a plain Python function as a tool"""
        name = func.__name__
        sig = inspect.signature(func)
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue

            param_type = self.type_mapper.map_type(param.annotation)
            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter: {param_name}"
            }

            # Mark as required if no default value
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        description = {
            "name": name,
            "description": func.__doc__ or f"Function: {name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }

        tool_service.register_tool(
            name=name,
            func=func,
            description=description)

        self.logger.debug(f"Registered callable tool: {name}")

    def _register_mcp_schema(self, tool_service: ToolService, schema: Dict[str, Any]):
        """Register an MCP-declared tool schema (without server connection)"""
        name = schema["name"]

        # Create a stub that returns an error
        async def mcp_stub(**kwargs):
            return {
                "error": f"Tool '{name}' is declared but not connected to an MCP server. "
                "Use 'type: mcp' with server_label to connect to a server.",
                "schema": schema
            }

        tool_service.register_tool(name, mcp_stub, schema)
        self.logger.debug(f"Registered MCP schema tool (stub): {name}")

    async def cleanup(self):
        """Close all MCP connections"""
        for connection in self.mcp_connections.values():
            await connection.close()
