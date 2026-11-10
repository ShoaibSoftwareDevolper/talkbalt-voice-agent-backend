import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

_engine: AsyncEngine | None = None
_sessions: async_sessionmaker[AsyncSession] | None = None


def database_configured() -> bool:
    return bool(os.getenv("DATABASE_URL"))


def async_database_url() -> str:
    """DATABASE_URL in the form SQLAlchemy's asyncpg driver expects.

    Accepts the plain `postgresql://` / `postgres://` URLs that local Postgres and Railway
    hand out, and maps libpq's `sslmode` query parameter to asyncpg's `ssl`.
    """
    parts = urlsplit(os.environ["DATABASE_URL"])
    scheme = "postgresql+asyncpg" if parts.scheme in ("postgres", "postgresql") else parts.scheme
    query = [("ssl" if k == "sslmode" else k, v) for k, v in parse_qsl(parts.query)]
    return urlunsplit((scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def get_engine() -> AsyncEngine:
    # Created lazily so each agent job process gets its own engine on its own event loop.
    global _engine, _sessions
    if _engine is None:
        _engine = create_async_engine(async_database_url(), pool_pre_ping=True, pool_size=5)
        _sessions = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """A session that commits on success and rolls back on error."""
    get_engine()
    assert _sessions is not None
    async with _sessions() as session, session.begin():
        yield session


async def dispose_engine() -> None:
    global _engine, _sessions
    if _engine is not None:
        await _engine.dispose()
        _engine = _sessions = None
