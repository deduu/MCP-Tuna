import logging
import re
from logging.handlers import RotatingFileHandler
import os
import sys

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "agentic.log")


class ColorFormatter(logging.Formatter):
    """Color-coded console formatter for MCP server logs."""

    RESET = "\033[0m"
    GREY = "\033[38;5;245m"

    LEVEL_COLORS = {
        logging.DEBUG:    "\033[36m",     # Cyan
        logging.INFO:     "\033[32m",     # Green
        logging.WARNING:  "\033[33m",     # Yellow
        logging.ERROR:    "\033[31m",     # Red
        logging.CRITICAL: "\033[1;31m",   # Bold Red
    }

    TAG_COLORS = {
        # Transport — bright cyan
        "HTTP-POST":      "\033[96m",
        "HTTP-GET":       "\033[96m",
        "HTTP-DELETE":    "\033[91m",
        # Legacy / stdio — bright blue
        "LEGACY-SSE":     "\033[94m",
        "StdioTransport": "\033[94m",
        # MCP protocol — bright green
        "MCP-HANDLE":     "\033[92m",
        # Success marker — bold bright green
        "OK":             "\033[1;92m",
    }

    _TAG_RE = re.compile(r"\[([A-Za-z_-]+)\]")

    def format(self, record):
        ts = self.formatTime(record, self.datefmt)
        level_color = self.LEVEL_COLORS.get(record.levelno, "")
        msg = record.getMessage()

        # Color [TAG] tokens
        def _color_tag(m):
            tag = m.group(1)
            c = self.TAG_COLORS.get(tag, "\033[33m")  # default yellow
            return f"{c}[{tag}]{self.RESET}"
        msg = self._TAG_RE.sub(_color_tag, msg)

        # Highlight tool names in "Calling tool: <name>"
        msg = re.sub(
            r"(Calling tool: )(\S+)",
            lambda m: f"{m.group(1)}\033[1;95m{m.group(2)}{self.RESET}",
            msg,
        )
        # Highlight tool names in "Tool <name> returned:" / "Tool <name> error:"
        msg = re.sub(
            r"(Tool )(\S+)( (?:returned|error):)",
            lambda m: f"{m.group(1)}\033[1;95m{m.group(2)}{self.RESET}{m.group(3)}",
            msg,
        )

        # Tint the whole message red on ERROR+
        if record.levelno >= logging.ERROR:
            msg = f"\033[31m{msg}{self.RESET}"

        return (
            f"{self.GREY}{ts}{self.RESET} "
            f"{level_color}{record.levelname:<8}{self.RESET} "
            f"{msg}"
        )


def configure_logging(level=logging.INFO, console=True):
    """
    Configure global logging for the entire project.
    Call this once at application startup.
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    root_logger = logging.getLogger()
    # Prevent duplicate handlers on reload
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    # File handler (rotating) — plain text, no colors
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
    plain_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(plain_formatter)
    root_logger.addHandler(file_handler)

    if console:
        stream = open(sys.stderr.fileno(), mode='w', encoding='utf-8',
                      closefd=False, errors='replace')
        console_handler = logging.StreamHandler(stream)
        # Use colors when connected to a real terminal, plain text otherwise
        if sys.stderr.isatty():
            console_handler.setFormatter(ColorFormatter())
        else:
            console_handler.setFormatter(plain_formatter)
        root_logger.addHandler(console_handler)

    root_logger.setLevel(level)


def get_logger(name: str, verbose: bool = False):
    """
    Return a logger that respects the verbose flag.
    Example:
        logger = get_logger(__name__, verbose=True)
        logger.info("Running agent...")
    """
    logger = logging.getLogger(name)
    # Set dynamic level
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger
