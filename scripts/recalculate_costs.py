"""Recompute estimated_cost_usd for calls that have usage totals, after editing config/pricing.toml.

Usage (from backend-web):  uv run scripts/recalculate_costs.py
Calls recorded before metrics capture (no token totals) are skipped.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

from db import Call, session_scope  # noqa: E402
from pricing import estimate_call_cost  # noqa: E402
from sqlalchemy import select  # noqa: E402

RECORDED = ("recording", "processing", "completed")


async def main() -> None:
    async with session_scope() as db:
        calls = (await db.scalars(select(Call).where(Call.llm_requests.is_not(None)))).all()
        for call in calls:
            cost, breakdown = estimate_call_cost(
                llm_model=call.llm_model,
                prompt_tokens=call.total_prompt_tokens or 0,
                cached_tokens=call.total_cached_tokens or 0,
                completion_tokens=call.total_completion_tokens or 0,
                stt_audio_seconds=call.total_stt_audio_seconds or 0,
                tts_characters=call.total_tts_characters or 0,
                duration_seconds=call.duration_seconds or 0,
                call_type=call.call_type,
                recorded=call.recording_status in RECORDED,
            )
            call.estimated_cost_usd, call.cost_breakdown = cost, breakdown
            print(f"{call.id} {call.started_at:%Y-%m-%d %H:%M} -> ${cost} {breakdown['components']}"
                  + (f" missing={breakdown['missing_prices']}" if breakdown["missing_prices"] else ""))
    print(f"recalculated {len(calls)} call(s)")


asyncio.run(main())
