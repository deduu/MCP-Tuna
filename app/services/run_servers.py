"""
Launch all utility MCP servers.

Usage:
    # Start all utility servers
    python -m app.services.run_servers

    # Start a specific server
    python -m app.services.run_servers database
    python -m app.services.run_servers email
    python -m app.services.run_servers files
    python -m app.services.run_servers web
"""

import sys

from agentsoul.server import HTTPTransport

# Server registry: name → (module, class, default_port)
SERVERS = {
    "database": ("app.services.database.mcp_server", "DatabaseMCPServer", 8002),
    "email": ("app.services.email.mcp_server", "EmailMCPServer", 8003),
    "files": ("app.services.files.mcp_server", "FileMCPServer", 8004),
    "web": ("app.services.web.mcp_server", "WebMCPServer", 8005),
}


def start_server(name: str, port: int = None):
    """Start a single MCP server by name."""
    if name not in SERVERS:
        print(f"Unknown server: {name}. Available: {list(SERVERS.keys())}")
        sys.exit(1)

    module_path, class_name, default_port = SERVERS[name]
    port = port or default_port

    import importlib
    module = importlib.import_module(module_path)
    server_class = getattr(module, class_name)
    server = server_class()

    print(f"Starting {name} MCP server on http://0.0.0.0:{port}", file=sys.stderr)
    print(f"Health: http://localhost:{port}/health", file=sys.stderr)

    transport = HTTPTransport(host="0.0.0.0", port=port)
    server.run(transport)


def main():
    if len(sys.argv) > 1:
        name = sys.argv[1]
        port = int(sys.argv[2]) if len(sys.argv) > 2 else None
        start_server(name, port)
    else:
        print("Usage: python -m app.services.run_servers <server_name> [port]")
        print("\nAvailable servers:")
        for name, (_, _, port) in SERVERS.items():
            print(f"  {name:12s}  (default port: {port})")


if __name__ == "__main__":
    main()
