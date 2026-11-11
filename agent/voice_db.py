"""The voice agent's own Postgres (VOICE_DATABASE_URL), separate from the platform database.

It holds an outbox: call reports the platform API couldn't take (down, restarting, network). They
are retried later, so a call's transcript and metrics are never lost because of a platform outage.
"""

import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PendingReport(Base):
    """A finished call's report waiting to be delivered to the platform API."""

    __tablename__ = "pending_reports"

    call_id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


_engine: AsyncEngine | None = None
_sessions: async_sessionmaker[AsyncSession] | None = None


def voice_db_url() -> str | None:
    url = os.getenv("VOICE_DATABASE_URL")
    return url.replace("postgresql://", "postgresql+asyncpg://", 1) if url else None


@asynccontextmanager
async def voice_session() -> AsyncIterator[AsyncSession]:
    global _engine, _sessions
    if _engine is None:
        url = voice_db_url()
        if url is None:
            raise RuntimeError("VOICE_DATABASE_URL is not set")
        # Created lazily so each agent job process gets its own engine on its own event loop.
        _engine = create_async_engine(url, pool_pre_ping=True, pool_size=2)
        _sessions = async_sessionmaker(_engine, expire_on_commit=False)
    assert _sessions is not None
    async with _sessions() as session, session.begin():
        yield session
