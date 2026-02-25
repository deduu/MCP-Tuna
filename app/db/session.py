import logging
from typing import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
    AsyncEngine
)

from ..core.config import settings

logger = logging.getLogger(__name__)


class DatabaseSessionManager:
    def __init__(self, url: str, **engine_kwargs):
        self.engine: AsyncEngine = create_async_engine(url, **engine_kwargs)
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

    async def close(self):
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        session = self.session_factory()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def create_tables(self):
        from .base import Base
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)


# Global instance
session_manager = DatabaseSessionManager(
    str(settings.database.url),
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_pre_ping=True,
    pool_recycle=3600,
)


async def get_async_db() -> AsyncIterator[AsyncSession]:
    async with session_manager.session() as session:
        yield session


async def startup_db():
    await session_manager.create_tables()
    logger.info("Database initialized")


async def shutdown_db():
    await session_manager.close()
