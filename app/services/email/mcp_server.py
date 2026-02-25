"""MCP server for email tools."""

import json
from typing import Optional

from agentsoul.server import MCPServer
from .service import EmailService
from app.core.config import settings


class EmailMCPServer:
    """Exposes email operations as MCP tools."""

    def __init__(self):
        email = settings.email
        self.service = EmailService(
            smtp_host=email.smtp_host,
            smtp_port=email.smtp_port,
            smtp_user=email.smtp_user,
            smtp_pass=email.smtp_pass,
            sender=email.sender,
        )
        self.mcp = MCPServer("email-tools", "1.0.0")
        self._register_tools()

    def _register_tools(self):
        svc = self.service

        @self.mcp.tool(
            name="email.send",
            description="Send an email. Supports plain text or HTML body.",
        )
        async def email_send(
            to: str, subject: str, body: str,
            html: bool = False, cc: Optional[str] = None,
        ) -> str:
            result = await svc.send(to=to, subject=subject, body=body, html=html, cc=cc)
            return json.dumps(result, indent=2)

        @self.mcp.tool(
            name="email.send_template",
            description="Send an email using a named template with variables. "
                        "Templates: welcome (name), password_reset (name, link), notification (subject, message).",
        )
        async def email_send_template(
            to: str, template_name: str, variables: str,
        ) -> str:
            parsed_vars = json.loads(variables)
            result = await svc.send_template(to=to, template_name=template_name, variables=parsed_vars)
            return json.dumps(result, indent=2)

    def run(self, transport=None):
        self.mcp.run(transport)
