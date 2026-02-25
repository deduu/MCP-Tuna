"""Email service — send emails via SMTP using configured credentials."""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Dict, List, Optional


class EmailService:
    """Provides email sending operations for MCP tool consumption."""

    def __init__(self, smtp_host: str, smtp_port: int, smtp_user: str, smtp_pass: str, sender: str):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_pass = smtp_pass
        self.sender = sender

    async def send(
        self,
        to: str,
        subject: str,
        body: str,
        html: bool = False,
        cc: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send an email."""
        if not self.smtp_host:
            return {"success": False, "error": "SMTP not configured. Set SMTP_HOST in environment."}

        msg = MIMEMultipart("alternative")
        msg["From"] = self.sender
        msg["To"] = to
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = cc

        content_type = "html" if html else "plain"
        msg.attach(MIMEText(body, content_type))

        recipients = [to]
        if cc:
            recipients.extend(addr.strip() for addr in cc.split(","))

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                if self.smtp_user and self.smtp_pass:
                    server.login(self.smtp_user, self.smtp_pass)
                server.sendmail(self.sender, recipients, msg.as_string())
            return {"success": True, "to": to, "subject": subject}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def send_template(
        self,
        to: str,
        template_name: str,
        variables: Dict[str, str],
    ) -> Dict[str, Any]:
        """Send an email from a named template with variable substitution."""
        templates = {
            "welcome": {
                "subject": "Welcome, {name}!",
                "body": "Hello {name},\n\nWelcome to our platform!\n\nBest regards,\nThe Team",
            },
            "password_reset": {
                "subject": "Password Reset Request",
                "body": "Hello {name},\n\nClick here to reset your password: {link}\n\nThis link expires in 24 hours.",
            },
            "notification": {
                "subject": "{subject}",
                "body": "{message}",
            },
        }

        if template_name not in templates:
            return {
                "success": False,
                "error": f"Unknown template '{template_name}'. Available: {list(templates.keys())}",
            }

        tmpl = templates[template_name]
        subject = tmpl["subject"].format(**variables)
        body = tmpl["body"].format(**variables)

        return await self.send(to=to, subject=subject, body=body)
