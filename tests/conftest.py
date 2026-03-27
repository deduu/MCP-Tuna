from __future__ import annotations

import asyncio
import os
from contextlib import suppress

import pytest

os.environ.setdefault("DB_USE_NULL_POOL", "true")

from app.db.session import shutdown_db


@pytest.fixture(scope="session")
def event_loop():
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.get_event_loop_policy().new_event_loop()
    try:
        yield loop
    finally:
        with suppress(Exception):
            loop.run_until_complete(shutdown_db())
            loop.run_until_complete(asyncio.sleep(0))
        loop.close()
