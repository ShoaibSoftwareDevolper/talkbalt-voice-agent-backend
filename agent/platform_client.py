"""Everything the voice agent needs from the platform API (platform-api), over its internal API.

The agent never connects to the platform database. Calls go to PLATFORM_API_URL with the shared
INTERNAL_API_TOKEN. The voice Redis keeps the last good copy of each business's setup, so a call
still gets its business if the platform API is briefly unreachable; finished call reports that
can't be delivered wait in the voice database's outbox and are retried.
"""

import contextlib
import json
import logging
import os
import uuid
from typing import Any

import httpx2 as httpx
from redis.asyncio import Redis
from redis.exceptions import RedisError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from voice_db import PendingReport, voice_db_url, voice_session

logger = logging.getLogger("talkbalt-agent")

LAST_GOOD_TTL_SECONDS = 7 * 24 * 3600
OUTBOX_BATCH = 20
_client: httpx.AsyncClient | None = None
_redis: Redis | None = None


def configured() -> bool:
    return bool(os.getenv("PLATFORM_API_URL") and os.getenv("INTERNAL_API_TOKEN"))


def _http() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=os.environ["PLATFORM_API_URL"].rstrip("/") + "/internal",
            headers={"X-Internal-Token": os.environ["INTERNAL_API_TOKEN"]},
            timeout=httpx.Timeout(5.0, connect=2.0),
        )
    return _client


def _cache() -> Redis | None:
    global _redis
    url = os.getenv("REDIS_URL")
    if url and _redis is None:
        _redis = Redis.from_url(url, decode_responses=True, socket_timeout=0.3, socket_connect_timeout=0.3)
    return _redis


async def business_runtime(org_id: uuid.UUID) -> dict[str, Any] | None:
    """The business's current setup from the platform; its last good copy if the platform is down."""
    key = f"biz:{org_id}:last-good"
    try:
        res = await _http().get(f"/businesses/{org_id}/runtime")
        if res.status_code == 404:
            return None
        res.raise_for_status()
        runtime = res.json()
    except httpx.HTTPError as e:
        logger.warning("platform API unreachable for business %s (%s); trying the last good copy", org_id, type(e).__name__)
        cache = _cache()
        try:
            cached = await cache.get(key) if cache else None
        except RedisError:
            cached = None
        return json.loads(cached) if cached else None
    if cache := _cache():
        with contextlib.suppress(RedisError):
            await cache.set(key, json.dumps(runtime), ex=LAST_GOOD_TTL_SECONDS)
    return runtime


async def start_call(payload: dict[str, Any]) -> None:
    try:
        res = await _http().post("/calls", json=payload)
        res.raise_for_status()
    except httpx.HTTPError as e:
        # The final report creates nothing on its own, so a missing row is retried with it.
        logger.warning("could not register call %s with the platform (%s)", payload.get("id"), type(e).__name__)
        raise


async def book(payload: dict[str, Any]) -> dict[str, Any]:
    res = await _http().post("/bookings", json=payload)
    res.raise_for_status()
    return res.json()


async def _deliver(call_id: str, payload: dict[str, Any]) -> None:
    if start := payload.get("start"):
        # Re-register first: harmless if the call row exists, needed if registration had failed.
        (await _http().post("/calls", json=start)).raise_for_status()
    (await _http().post(f"/calls/{call_id}/report", json=payload["report"])).raise_for_status()


async def report_call(call_id: uuid.UUID, payload: dict[str, Any]) -> bool:
    """Delivers a finished call's report, or keeps it in the outbox. True if delivered now."""
    try:
        await _deliver(str(call_id), payload)
        return True
    except httpx.HTTPError as e:
        logger.warning("call report %s not delivered (%s); keeping it in the outbox", call_id, type(e).__name__)
        if voice_db_url() is None:
            return False
        async with voice_session() as db:
            await db.execute(
                insert(PendingReport)
                .values(call_id=call_id, payload=payload)
                .on_conflict_do_update(index_elements=["call_id"], set_={"payload": payload})
            )
        return False


async def flush_outbox() -> None:
    """Retries reports that couldn't be delivered earlier. Runs after each call."""
    if voice_db_url() is None:
        return
    async with voice_session() as db:
        pending = (await db.scalars(select(PendingReport).order_by(PendingReport.created_at).limit(OUTBOX_BATCH))).all()
        for item in pending:
            try:
                await _deliver(str(item.call_id), item.payload)
                await db.delete(item)
                logger.info("delivered queued call report %s", item.call_id)
            except httpx.HTTPError:
                break  # the platform is still unreachable; try again after the next call
