import logging
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware
from app.api.api_chat import router as chat_router
from app.db.session import shutdown_db, startup_db

from agentsoul.utils.logger import configure_logging
from shared.diagnostics import init_diagnostics, session_id_var

import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

configure_logging()
logger = logging.getLogger(__name__)

# Allow Swagger locally and from Docker network
INTERNAL_IPS = {"127.0.0.1", "localhost", "::1"}
DOCKER_NETWORK = "172.17.0.0/16"  # Docker default bridge network


class RestrictRootAccessMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        client_ip = request.client.host

        # Check if request is for root/index
        if path in ("/", "/index.html"):
            # Allow internal IPs
            if client_ip in INTERNAL_IPS:
                return await call_next(request)

            # Allow Docker network IPs (172.17.x.x)
            if client_ip and client_ip.startswith("172.17."):
                return await call_next(request)

            # Block everyone else
            return HTMLResponse("<h1>Forbidden</h1>", status_code=403)

        # For all other paths, allow through
        return await call_next(request)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown lifecycle handler for preloading LLM models.
    """
    logger.info("Starting backend... initializing registry")

    # registry = get_registry()

    # preload_items = [
    #     # ("Auto", None),
    #     # ("Qwen/Qwen2.5-VL-7B-Instruct", None),
    # ]

    try:
        logger.info("Setting up initial database connections...")
        await startup_db()
    except Exception as e:
        logger.exception(f"Failed to set up database connections: {e}")

    try:
        logger.info("Preloading models... this may take a minute")
        # await registry.preload(preload_items)
        logger.info("Preload complete. Models ready for inference.")
    except Exception as e:
        logger.exception(f"Failed to preload models: {e}")

    _session_id = str(uuid.uuid4())
    session_id_var.set(_session_id)
    _diag_writer = init_diagnostics(log_root="logs")
    logger.info(f"Diagnostic session: {_session_id}")

    yield

    await _diag_writer.close()
    await shutdown_db()
    logger.info("Shutting down backend... releasing resources.")


app = FastAPI(title="Asistent Agent Server", lifespan=lifespan)

# Add middleware in correct order (last added = first executed)
app.add_middleware(RestrictRootAccessMiddleware)  # Add this first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
