"""Call history: one `calls` row per call, created at start and finalized at session end.

Database problems are logged and swallowed; they must never affect the call itself.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from db import Call, CallEvent, CallTurnMetric, database_configured, session_scope
from livekit.agents import CloseEvent, ConversationItemAddedEvent
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics
from pricing import estimate_call_cost
from sqlalchemy import update

logger = logging.getLogger("talkbalt-agent")

# AgentSession close reasons that don't already have a more specific end_reason.
_CLOSE_REASONS = {
    "participant_disconnected": "user_hangup",
    "error": "error",
    "job_shutdown": "shutdown",
    "user_initiated": "shutdown",
    "task_completed": "completed",
}

_USER_METRICS = ("stt_ms", "eou_ms")
_AGENT_METRICS = ("llm_ttft_ms", "tts_ttfb_ms", "e2e_ms")


def _ms(value: float | None) -> int | None:
    return round(value * 1000) if value is not None else None


def turn_metrics(item: ChatMessage) -> dict[str, int | None]:
    """Per-turn latency in ms from ChatMessage.metrics, keyed for the browser and the DB."""
    m = item.metrics
    if item.role == "user":
        return {"stt_ms": _ms(m.get("transcription_delay")), "eou_ms": _ms(m.get("end_of_turn_delay"))}
    return {
        "llm_ttft_ms": _ms(m.get("llm_node_ttft")),
        "tts_ttfb_ms": _ms(m.get("tts_node_ttfb")),
        "e2e_ms": _ms(m.get("e2e_latency")),
    }


def _ts(epoch: float | None) -> datetime | None:
    return datetime.fromtimestamp(epoch, timezone.utc) if epoch else None


def mask_phone_number(number: str | None) -> str | None:
    digits = "".join(c for c in number or "" if c.isdigit())
    return f"***{digits[-4:]}" if len(digits) >= 4 else None


class CallTracker:
    def __init__(
        self,
        *,
        room_name: str,
        call_type: str,
        llm_model: str,
        caller_number: str | None,
        recording_status: str,
    ) -> None:
        self.room_name = room_name
        self.call_type = call_type
        self.llm_model = llm_model
        self.caller_number_masked = mask_phone_number(caller_number) if call_type == "phone" else None
        self.started_at = datetime.now(timezone.utc)

        self.recording_status = recording_status
        self.recording_key: str | None = None
        self.egress_id: str | None = None

        # Set by code that knows why the call ended (end_call, crash); wins over close_reason.
        self.end_reason: str | None = None
        self.close_reason: str | None = None
        self.transcript: list[dict[str, Any]] = []

        self.call_id: uuid.UUID | None = None
        self._insert_task: asyncio.Task[None] | None = None

        # Metrics buffered in memory and written once at session end (never on the audio path).
        # LLM and TTS metrics queue up until the agent message they produced is added, so a tool
        # call round trip or a discarded preemptive generation counts toward that turn.
        self._pending_llm: list[LLMMetrics] = []
        self._pending_tts: list[TTSMetrics] = []
        self._stt_seconds_since_user_turn = 0.0
        self.turns: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.totals = {"prompt": 0, "cached": 0, "completion": 0, "reasoning": 0, "llm_requests": 0,
                       "stt_seconds": 0.0, "tts_characters": 0, "tts_seconds": 0.0}

    def add_event(self, type_: str, detail: str | None = None) -> None:
        """Records a notable event. `detail` must never contain caller content."""
        self.events.append({"timestamp": datetime.now(timezone.utc), "type": type_, "detail": (detail or "")[:255] or None})

    def on_llm_metrics(self, m: LLMMetrics) -> None:
        self._pending_llm.append(m)
        self.totals["llm_requests"] += 1
        self.totals["prompt"] += m.prompt_tokens
        self.totals["cached"] += m.prompt_cached_tokens
        self.totals["completion"] += m.completion_tokens  # already includes reasoning tokens
        self.totals["reasoning"] += m.reasoning_tokens

    def on_tts_metrics(self, m: TTSMetrics) -> None:
        self._pending_tts.append(m)
        self.totals["tts_characters"] += m.characters_count
        self.totals["tts_seconds"] += m.audio_duration

    def on_stt_metrics(self, m: STTMetrics) -> None:
        # Streaming STT reports audio in periodic chunks, not per utterance.
        self._stt_seconds_since_user_turn += m.audio_duration
        self.totals["stt_seconds"] += m.audio_duration

    def _turn_row(self, item: ChatMessage) -> dict[str, Any]:
        m = item.metrics
        row: dict[str, Any] = {
            "turn_index": len(self.turns),
            "role": "user" if item.role == "user" else "agent",
            "started_at": _ts(m.get("started_speaking_at")) or _ts(item.created_at),
            "ended_at": _ts(m.get("stopped_speaking_at")),
            "interrupted": bool(item.interrupted),
            "cancelled": False,
        }
        if item.role == "user":
            row["stt_audio_seconds"] = round(self._stt_seconds_since_user_turn, 2)
            row["eou_delay_ms"] = _ms(m.get("end_of_turn_delay"))
            row["transcription_delay_ms"] = _ms(m.get("transcription_delay"))
            self._stt_seconds_since_user_turn = 0.0
            return row

        llm, self._pending_llm = self._pending_llm, []
        tts, self._pending_tts = self._pending_tts, []
        # A request closed early (truncation net, interruption) may end before Groq sends usage.
        with_usage = [r for r in llm if r.prompt_tokens or r.completion_tokens]
        row.update(
            llm_requests=len(llm) or None,
            llm_ttft_ms=_ms(m.get("llm_node_ttft")),
            llm_duration_ms=_ms(sum(r.duration for r in llm)) if llm else None,
            prompt_tokens=sum(r.prompt_tokens for r in with_usage) if with_usage else None,
            prompt_cached_tokens=sum(r.prompt_cached_tokens for r in with_usage) if with_usage else None,
            completion_tokens=sum(r.completion_tokens for r in with_usage) if with_usage else None,
            reasoning_tokens=sum(r.reasoning_tokens for r in with_usage) if with_usage else None,
            tokens_per_second=round(mean(r.tokens_per_second for r in with_usage), 1) if with_usage else None,
            tts_ttfb_ms=_ms(m.get("tts_node_ttfb")),
            tts_characters=sum(t.characters_count for t in tts) if tts else None,
            tts_audio_seconds=round(sum(t.audio_duration for t in tts), 2) if tts else None,
            e2e_ms=_ms(m.get("e2e_latency")),
            cancelled=any(r.cancelled for r in llm),
        )
        return row

    def start(self) -> None:
        """Inserts the row in the background so the greeting isn't delayed."""
        if not database_configured():
            logger.info("call history disabled: DATABASE_URL not set")
            return
        self._insert_task = asyncio.create_task(self._insert())

    async def _insert(self) -> None:
        call = Call(
            room_name=self.room_name,
            call_type=self.call_type,
            started_at=self.started_at,
            llm_model=self.llm_model,
            recording_status=self.recording_status,
            caller_number_masked=self.caller_number_masked,
        )
        try:
            async with session_scope() as db:
                db.add(call)
            self.call_id = call.id
            logger.info("call history: row created id=%s", call.id)
        except Exception as e:
            logger.warning("call history: could not create row (%s)", e)

    def on_item_added(self, ev: ConversationItemAddedEvent) -> None:
        if not isinstance(ev.item, ChatMessage) or ev.item.role not in ("user", "assistant"):
            return
        text = ev.item.text_content or ""
        if not text:
            return
        self.turns.append(self._turn_row(ev.item))
        if ev.item.role == "assistant" and ev.item.interrupted:
            self.add_event("interruption", f"agent turn {len(self.turns) - 1} cut short by the caller")
        self.transcript.append(
            {
                "role": "user" if ev.item.role == "user" else "agent",
                "text": text,
                "timestamp": datetime.fromtimestamp(ev.item.created_at, timezone.utc).isoformat(),
                "metrics": turn_metrics(ev.item),
                # True when the caller talked over the agent; `text` is then only what was spoken.
                "interrupted": bool(ev.item.interrupted),
            }
        )

    def on_close(self, ev: CloseEvent) -> None:
        self.close_reason = ev.reason.value

    def _averages(self) -> dict[str, int | None]:
        def avg(role: str, key: str) -> int | None:
            values = [t["metrics"][key] for t in self.transcript if t["role"] == role and t["metrics"].get(key) is not None]
            return round(mean(values)) if values else None

        return {
            "avg_stt_ms": avg("user", "stt_ms"),
            "avg_llm_ttft_ms": avg("agent", "llm_ttft_ms"),
            "avg_tts_ttfb_ms": avg("agent", "tts_ttfb_ms"),
            "avg_e2e_ms": avg("agent", "e2e_ms"),
        }

    async def finalize(self) -> None:
        """Writes the end of the call. Called from on_session_end, which runs for every ending."""
        if self._insert_task is not None:
            await self._insert_task
        if self.call_id is None:
            return

        ended_at = datetime.now(timezone.utc)
        end_reason = self.end_reason or _CLOSE_REASONS.get(self.close_reason or "", "unknown")
        duration = round((ended_at - self.started_at).total_seconds(), 1)
        tot = self.totals
        cost, breakdown = estimate_call_cost(
            llm_model=self.llm_model,
            prompt_tokens=tot["prompt"],
            cached_tokens=tot["cached"],
            completion_tokens=tot["completion"],
            stt_audio_seconds=tot["stt_seconds"],
            tts_characters=tot["tts_characters"],
            duration_seconds=duration,
            call_type=self.call_type,
            recorded=self.recording_status in ("recording", "processing", "completed"),
        )
        values = {
            "ended_at": ended_at,
            "duration_seconds": duration,
            "end_reason": end_reason,
            "turn_count": sum(1 for entry in self.transcript if entry["role"] == "user"),
            "transcript": self.transcript,
            "recording_status": self.recording_status,
            "recording_key": self.recording_key,
            "egress_id": self.egress_id,
            **self._averages(),
            "total_prompt_tokens": tot["prompt"],
            "total_cached_tokens": tot["cached"],
            "total_completion_tokens": tot["completion"],
            "total_reasoning_tokens": tot["reasoning"],
            "llm_requests": tot["llm_requests"],
            "total_stt_audio_seconds": round(tot["stt_seconds"], 2),
            "total_tts_characters": tot["tts_characters"],
            "total_tts_audio_seconds": round(tot["tts_seconds"], 2),
            "cache_hit_ratio": round(tot["cached"] / tot["prompt"], 4) if tot["prompt"] else None,
            "estimated_cost_usd": cost,
            "cost_breakdown": breakdown,
        }
        try:
            async with session_scope() as db:
                await db.execute(update(Call).where(Call.id == self.call_id).values(**values))
                db.add_all(CallTurnMetric(call_id=self.call_id, **row) for row in self.turns)
                db.add_all(CallEvent(call_id=self.call_id, **ev) for ev in self.events)
            logger.info(
                "call history: row finalized id=%s end_reason=%s turns=%s metrics_rows=%s events=%s recording=%s",
                self.call_id,
                end_reason,
                values["turn_count"],
                len(self.turns),
                len(self.events),
                self.recording_status,
            )
        except Exception as e:
            logger.warning("call history: could not finalize row %s (%s)", self.call_id, e)
