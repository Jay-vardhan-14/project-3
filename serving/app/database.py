"""Async SQLAlchemy engine, session factory, and table initialization."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import SETTINGS
from app.models import Base

LOGGER = logging.getLogger(__name__)

engine = create_async_engine(SETTINGS.database_url, pool_pre_ping=True, future=True)
SessionFactory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_models() -> None:
    """Create tables if they do not already exist (coexists with the drift DAG DDL)."""

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    LOGGER.info("Database tables ensured.")


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding an async database session."""

    async with SessionFactory() as session:
        yield session
