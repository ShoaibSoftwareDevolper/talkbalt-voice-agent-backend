"""Metrics, analytics and usage endpoints (X-Admin-Token protected, like call history).

Usage numbers are counted by Talkbalt from its own call records; nothing is fetched from the
providers. Costs are estimates from config/pricing.toml.
"""

import os
import uuid
from datetime import date, datetime, time, timedelta
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from db import Call, CallEvent, CallTurnMetric
from fastapi import APIRouter, Depends, HTTPException, Query
from pricing import free_tier, price
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.history import get_db, require_admin

router = APIRouter(prefix="/api", dependencies=[Depends(require_admin)])
Db = Annotated[AsyncSession, Depends(get_db)]

# Latency stages, as (key, call_turn_metrics column).
STAGES = [
    ("stt", "transcription_delay_ms"),
    ("end_of_turn", "eou_delay_ms"),
    ("llm_ttft", "llm_ttft_ms"),
    ("tts_ttfb", "tts_ttfb_ms"),
    ("e2e", "e2e_ms"),
]
RECORDED = ("recording", "processing", "completed")


def _tz() -> ZoneInfo:
    return ZoneInfo(os.getenv("BUSINESS_TIMEZONE", "Asia/Karachi").strip() or "Asia/Karachi")


def _columns(model: Any, row: Any, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    return {c.name: getattr(row, c.name) for c in model.__table__.columns if c.name not in skip}


@router.get("/calls/{call_id}/metrics")
async def call_metrics(call_id: uuid.UUID, db: Db) -> dict[str, Any]:
    call = await db.get(Call, call_id)
    if call is None:
        raise HTTPException(status_code=404, detail="Call not found")
    turns = (await db.scalars(select(CallTurnMetric).where(CallTurnMetric.call_id == call_id).order_by(CallTurnMetric.turn_index))).all()
    events = (await db.scalars(select(CallEvent).where(CallEvent.call_id == call_id).order_by(CallEvent.timestamp))).all()
    return {
        # False for calls recorded before metrics capture existed; the UI shows an empty state.
        "has_metrics": bool(turns) or call.llm_requests is not None,
        "summary": {
            "total_prompt_tokens": call.total_prompt_tokens,
            "total_cached_tokens": call.total_cached_tokens,
            "total_completion_tokens": call.total_completion_tokens,
            "total_reasoning_tokens": call.total_reasoning_tokens,
            "llm_requests": call.llm_requests,
            "total_stt_audio_seconds": call.total_stt_audio_seconds,
            "total_tts_characters": call.total_tts_characters,
            "total_tts_audio_seconds": call.total_tts_audio_seconds,
            "cache_hit_ratio": call.cache_hit_ratio,
            "estimated_cost_usd": call.estimated_cost_usd,
            "cost_breakdown": call.cost_breakdown,
        },
        "turns": [_columns(CallTurnMetric, t, skip=("id", "call_id")) for t in turns],
        "events": [_columns(CallEvent, e, skip=("id", "call_id")) for e in events],
    }


def _range(from_: date | None, to: date | None) -> tuple[date, date, datetime, datetime]:
    tz = _tz()
    end_day = to or datetime.now(tz).date()
    start_day = from_ or end_day - timedelta(days=6)
    if start_day > end_day:
        raise HTTPException(status_code=400, detail="'from' must not be after 'to'")
    if (end_day - start_day).days > 366:
        raise HTTPException(status_code=400, detail="Range is limited to one year")
    start = datetime.combine(start_day, time.min, tz)
    end = datetime.combine(end_day + timedelta(days=1), time.min, tz)
    return start_day, end_day, start, end


@router.get("/analytics")
async def analytics(
    db: Db,
    from_: Annotated[date | None, Query(alias="from")] = None,
    to: Annotated[date | None, Query()] = None,
) -> dict[str, Any]:
    start_day, end_day, start, end = _range(from_, to)
    params = {"tz": str(_tz()), "start": start, "end": end}
    day = "(c.started_at AT TIME ZONE :tz)::date"

    daily_rows = (
        await db.execute(
            text(f"""
                SELECT {day} AS day,
                       count(*) AS calls,
                       count(*) FILTER (WHERE c.call_type = 'phone') AS phone_calls,
                       coalesce(sum(c.total_prompt_tokens), 0) AS prompt_tokens,
                       coalesce(sum(c.total_cached_tokens), 0) AS cached_tokens,
                       coalesce(sum(c.total_completion_tokens), 0) AS completion_tokens,
                       coalesce(sum(c.total_reasoning_tokens), 0) AS reasoning_tokens,
                       sum(c.estimated_cost_usd) AS cost_usd,
                       count(c.estimated_cost_usd) AS calls_with_cost
                FROM calls c
                WHERE c.started_at >= :start AND c.started_at < :end
                GROUP BY 1
            """),
            params,
        )
    ).mappings().all()

    stage_sql = ", ".join(
        f"avg(t.{col}) AS {key}_avg, percentile_cont(0.95) WITHIN GROUP (ORDER BY t.{col}) AS {key}_p95"
        for key, col in STAGES
    )
    latency_rows = (
        await db.execute(
            text(f"""
                SELECT {day} AS day, {stage_sql}
                FROM call_turn_metrics t JOIN calls c ON c.id = t.call_id
                WHERE c.started_at >= :start AND c.started_at < :end
                GROUP BY 1
            """),
            params,
        )
    ).mappings().all()

    async def breakdown(column: str) -> list[dict[str, Any]]:
        rows = await db.execute(
            text(f"""
                SELECT coalesce(c.{column}, 'unknown') AS key, count(*) AS calls
                FROM calls c WHERE c.started_at >= :start AND c.started_at < :end
                GROUP BY 1 ORDER BY 2 DESC
            """),
            params,
        )
        return [dict(r) for r in rows.mappings().all()]

    by_day = {r["day"]: r for r in daily_rows}
    lat_by_day = {r["day"]: r for r in latency_rows}
    days = []
    d = start_day
    while d <= end_day:
        r = by_day.get(d)
        lat = lat_by_day.get(d)
        prompt = int(r["prompt_tokens"]) if r else 0
        cached = int(r["cached_tokens"]) if r else 0
        days.append(
            {
                "date": d.isoformat(),
                "calls": int(r["calls"]) if r else 0,
                "phone_calls": int(r["phone_calls"]) if r else 0,
                "tokens": {
                    "uncached_input": prompt - cached,
                    "cached_input": cached,
                    "output": int(r["completion_tokens"]) - int(r["reasoning_tokens"]) if r else 0,
                    "reasoning": int(r["reasoning_tokens"]) if r else 0,
                },
                "cache_hit_ratio": round(cached / prompt, 4) if prompt else None,
                "estimated_cost_usd": float(r["cost_usd"]) if r and r["cost_usd"] is not None else None,
                "latency_ms": {
                    key: {
                        "avg": round(lat[f"{key}_avg"]) if lat and lat[f"{key}_avg"] is not None else None,
                        "p95": round(lat[f"{key}_p95"]) if lat and lat[f"{key}_p95"] is not None else None,
                    }
                    for key, _ in STAGES
                },
            }
        )
        d += timedelta(days=1)

    return {
        "from": start_day.isoformat(),
        "to": end_day.isoformat(),
        "timezone": str(_tz()),
        "days": days,
        "outcomes": await breakdown("end_reason"),
        "call_types": await breakdown("call_type"),
        "models": await breakdown("llm_model"),
        "costs_are_estimates": True,
    }


@router.get("/usage")
async def usage(db: Db) -> dict[str, Any]:
    """This month's usage against free tiers, counted by Talkbalt (not fetched from providers)."""
    tz = _tz()
    now = datetime.now(tz)
    month_start = datetime.combine(now.date().replace(day=1), time.min, tz)
    day_start = datetime.combine(now.date(), time.min, tz)
    row = (
        await db.execute(
            text("""
                SELECT coalesce(sum(total_stt_audio_seconds), 0) AS stt_seconds,
                       coalesce(sum(total_tts_characters), 0) AS tts_characters,
                       coalesce(sum(ceil(duration_seconds / 60.0)) FILTER (WHERE call_type = 'phone'), 0) AS phone_minutes,
                       coalesce(sum(ceil(duration_seconds / 60.0)) FILTER (WHERE recording_status = ANY(:recorded)), 0) AS egress_minutes,
                       coalesce(sum(llm_requests) FILTER (WHERE started_at >= :day_start), 0) AS llm_requests_today,
                       coalesce(sum(total_prompt_tokens + total_completion_tokens) FILTER (WHERE started_at >= :day_start), 0) AS llm_tokens_today
                FROM calls WHERE started_at >= :month_start
            """),
            {"month_start": month_start, "day_start": day_start, "recorded": list(RECORDED)},
        )
    ).mappings().one()

    stt_minutes = float(row["stt_seconds"]) / 60
    tts_chars = int(row["tts_characters"])
    stt_price, tts_price = price("stt", "nova-3", "per_minute"), price("tts", "aura-2", "per_1k_characters")
    deepgram_used = None
    if stt_price is not None and tts_price is not None:
        deepgram_used = round(stt_minutes * stt_price + tts_chars / 1000 * tts_price, 4)

    def meter(key: str, label: str, used: float | None, limit: float | None, unit: str, note: str, period: str = "month") -> dict[str, Any]:
        return {"key": key, "label": label, "used": used, "limit": limit, "unit": unit, "period": period, "note": note}

    return {
        "source": "Counted by Talkbalt from its own call records, not fetched from the providers.",
        "month": now.strftime("%Y-%m"),
        "meters": [
            meter(
                "deepgram_credit", "Deepgram credit used", deepgram_used, free_tier("deepgram_credit_usd"), "USD",
                f"Estimated from {stt_minutes:.1f} STT minutes and {tts_chars} TTS characters this month"
                + ("" if deepgram_used is not None else "; set Deepgram prices in config/pricing.toml to estimate USD"),
            ),
            meter("livekit_inbound", "LiveKit inbound phone minutes", float(row["phone_minutes"]), free_tier("livekit_inbound_minutes"),
                  "min", "Phone call durations, rounded up per call"),
            meter("livekit_egress", "LiveKit egress minutes (recordings)", float(row["egress_minutes"]), free_tier("livekit_egress_minutes"),
                  "min", "Durations of recorded calls, rounded up per call"),
            meter("groq_requests", "Groq requests today", float(row["llm_requests_today"]), free_tier("groq_requests_per_day"),
                  "requests", "LLM requests made by the agent today", period="day"),
            meter("groq_tokens", "Groq tokens today", float(row["llm_tokens_today"]), free_tier("groq_tokens_per_day"),
                  "tokens", "Prompt plus completion tokens today", period="day"),
        ],
    }
