"""Call history endpoints, protected by the X-Admin-Token header (ADMIN_TOKEN in .env.local)."""

import asyncio
import os
import secrets
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Annotated, Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from db import Call, session_scope
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from storage import R2Config

RECORDING_URL_TTL_SECONDS = 300


def require_admin(x_admin_token: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Call history is not configured")
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(status_code=401, detail="Invalid admin token")


async def get_db() -> AsyncIterator[AsyncSession]:
    if not os.getenv("DATABASE_URL"):
        raise HTTPException(status_code=503, detail="Database is not configured")
    async with session_scope() as db:
        yield db


router = APIRouter(prefix="/api/calls", dependencies=[Depends(require_admin)])
Db = Annotated[AsyncSession, Depends(get_db)]


class CallSummary(BaseModel):
    id: uuid.UUID
    room_name: str
    call_type: str
    started_at: datetime
    ended_at: datetime | None
    duration_seconds: float | None
    end_reason: str | None
    llm_model: str | None
    recording_status: str
    has_recording: bool
    turn_count: int
    avg_e2e_ms: int | None
    caller_number_masked: str | None

    @classmethod
    def of(cls, call: Call) -> "CallSummary":
        return cls(
            **{f: getattr(call, f) for f in cls.model_fields if f != "has_recording"},
            has_recording=call.recording_key is not None and call.recording_status in ("processing", "completed"),
        )


class CallDetail(CallSummary):
    recording_key: str | None
    avg_stt_ms: int | None
    avg_llm_ttft_ms: int | None
    avg_tts_ttfb_ms: int | None
    transcript: list[dict[str, Any]]

    @classmethod
    def of(cls, call: Call) -> "CallDetail":
        summary = CallSummary.of(call)
        return cls(
            **summary.model_dump(),
            recording_key=call.recording_key,
            avg_stt_ms=call.avg_stt_ms,
            avg_llm_ttft_ms=call.avg_llm_ttft_ms,
            avg_tts_ttfb_ms=call.avg_tts_ttfb_ms,
            transcript=call.transcript or [],
        )


class CallPage(BaseModel):
    items: list[CallSummary]
    total: int
    limit: int
    offset: int


class RecordingUrl(BaseModel):
    url: str
    expires_in: int


@router.get("")
async def list_calls(
    db: Db,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> CallPage:
    total = await db.scalar(select(func.count()).select_from(Call))
    rows = await db.scalars(select(Call).order_by(Call.started_at.desc()).limit(limit).offset(offset))
    return CallPage(items=[CallSummary.of(c) for c in rows], total=total or 0, limit=limit, offset=offset)


async def _get_call(db: AsyncSession, call_id: uuid.UUID) -> Call:
    call = await db.get(Call, call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    return call


@router.get("/{call_id}")
async def get_call(call_id: uuid.UUID, db: Db) -> CallDetail:
    return CallDetail.of(await _get_call(db, call_id))


def _r2_client(r2: R2Config) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=r2.endpoint,
        aws_access_key_id=r2.access_key_id,
        aws_secret_access_key=r2.secret_access_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def _object_exists(client: Any, bucket: str, key: str) -> bool:
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


@router.get("/{call_id}/recording")
async def get_recording_url(call_id: uuid.UUID, db: Db) -> RecordingUrl:
    """A short-lived presigned GET URL for the call's recording. R2 keys never leave the server."""
    call = await _get_call(db, call_id)
    if not call.recording_key or call.recording_status in ("disabled", "failed"):
        raise HTTPException(status_code=404, detail="This call has no recording")
    r2 = R2Config.from_env()
    if r2 is None:
        raise HTTPException(status_code=503, detail="Recording storage is not configured")

    client = _r2_client(r2)
    try:
        exists = await asyncio.to_thread(_object_exists, client, r2.bucket, call.recording_key)
    except ClientError:
        raise HTTPException(status_code=502, detail="Could not reach recording storage") from None
    if not exists:
        # The upload finishes after the call ends; give it a moment.
        raise HTTPException(status_code=404, detail="Recording is not available yet")

    if call.recording_status != "completed":
        await db.execute(update(Call).where(Call.id == call.id).values(recording_status="completed"))

    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": r2.bucket, "Key": call.recording_key, "ResponseContentType": "audio/ogg"},
        ExpiresIn=RECORDING_URL_TTL_SECONDS,
    )
    return RecordingUrl(url=url, expires_in=RECORDING_URL_TTL_SECONDS)
