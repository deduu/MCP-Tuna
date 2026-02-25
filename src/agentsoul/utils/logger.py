import logging
from logging.handlers import RotatingFileHandler
import os
import sys

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "agentic.log")


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

    # File handler (rotating)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    if console:
        stream = open(sys.stderr.fileno(), mode='w', encoding='utf-8',
                      closefd=False, errors='replace')
        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
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
