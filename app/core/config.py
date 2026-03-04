import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from urllib.parse import quote_plus
from dotenv import load_dotenv

# --------------------------------------------------
# Load environment variables ONCE
# --------------------------------------------------
load_dotenv()


# --------------------------------------------------
# Database
# --------------------------------------------------
@dataclass
class DatabaseSettings:
    driver: str = os.getenv("DB_DRIVER", "postgresql+asyncpg")
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    name: str = os.getenv("DB_NAME", "dbname")
    user: str = os.getenv("DB_USER", "user")
    password: str = os.getenv("DB_PASSWORD", "password")

    pool_size: int = int(os.getenv("DB_POOL_SIZE", "5"))
    max_overflow: int = int(os.getenv("DB_MAX_OVERFLOW", "10"))
    pool_timeout: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))

    # @property
    # def url(self) -> str:
    #     return str(URL.create(
    #         drivername=self.driver,
    #         username=self.user,
    #         password=self.password,
    #         host=self.host,
    #         port=self.port,
    #         database=self.name,
    #     ))
    @property
    def url(self) -> str:
        safe_password = quote_plus(self.password)
        return (
            f"{self.driver}://{self.user}:{safe_password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


# --------------------------------------------------
# Files / Uploads
# --------------------------------------------------
@dataclass
class FileSettings:
    upload_root: Path = Path(os.getenv("UPLOAD_ROOT", "./uploads"))
    images_dir: Path = field(init=False)

    def __post_init__(self):
        self.images_dir = self.upload_root / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Email
# --------------------------------------------------
@dataclass
class EmailSettings:
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_pass: str = os.getenv("SMTP_PASS", "")
    sender: str = os.getenv("SMTP_SENDER", "")


# --------------------------------------------------
# MCP Server Connections
# --------------------------------------------------
@dataclass
class MCPServerConfig:
    """Config for a single MCP server connection."""
    server_label: str
    server_url: str
    server_description: str = ""
    require_approval: str = "never"


@dataclass
class MCPSettings:
    """MCP server connections available to agents."""
    servers: List[MCPServerConfig] = field(default_factory=lambda: [
        MCPServerConfig(
            server_label="mcp-tuna-gateway",
            server_url=os.getenv("MCP_TUNA_GATEWAY_URL", "http://localhost:8002/mcp"),
            server_description="MCP Tuna pipeline tools (generate, clean, evaluate, finetune, host)",
        ),
        MCPServerConfig(
            server_label="database-tools",
            server_url=os.getenv("DB_MCP_URL", "http://localhost:8002/mcp"),
            server_description="Database tools (query, list tables, describe, insert)",
        ),
        MCPServerConfig(
            server_label="email-tools",
            server_url=os.getenv("EMAIL_MCP_URL", "http://localhost:8003/mcp"),
            server_description="Email tools (send, send from template)",
        ),
        MCPServerConfig(
            server_label="file-tools",
            server_url=os.getenv("FILE_MCP_URL", "http://localhost:8004/mcp"),
            server_description="File management tools (read, write, list, upload)",
        ),
        MCPServerConfig(
            server_label="web-tools",
            server_url=os.getenv("WEB_MCP_URL", "http://localhost:8005/mcp"),
            server_description="Web tools (fetch URLs, search the web)",
        ),
    ])
    auto_connect: bool = True


# --------------------------------------------------
# App
# --------------------------------------------------
@dataclass
class AppSettings:
    app_name: str = os.getenv("APP_NAME", "Prompt test challenge App")
    env: str = os.getenv("ENV", "development")

    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    files: FileSettings = field(default_factory=FileSettings)
    email: EmailSettings = field(default_factory=EmailSettings)
    mcp: MCPSettings = field(default_factory=MCPSettings)

    backend_url: str = os.getenv("BACKEND_URL", "")
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:8008/")


# --------------------------------------------------
# Singleton
# --------------------------------------------------
settings = AppSettings()
