"""Alembic migrations for the voice agent's own database (VOICE_DATABASE_URL)."""

import asyncio
import sys
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env.local")
sys.path.insert(0, str(ROOT / "agent"))
from voice_db import Base, voice_db_url  # noqa: E402


def migrate(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()


async def main() -> None:
    url = voice_db_url()
    if url is None:
        raise RuntimeError("VOICE_DATABASE_URL is not set")
    engine = create_async_engine(url)
    async with engine.connect() as connection:
        await connection.run_sync(migrate)
    await engine.dispose()


asyncio.run(main())
