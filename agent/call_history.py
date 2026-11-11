"""Call history: the call is registered with the platform API when it starts, and its full report
(transcript, per-turn metrics, events, usage totals) is sent once at session end.

Platform problems are logged and swallowed; they must never affect the call itself. A report the
platform can't take waits in the voice database's outbox (see platform_client).
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from statistics import mean
from typing import Any

from livekit.agents import CloseEvent, ConversationItemAddedEvent
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics

import platform_client

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
    return datetime.fromtimestamp(epoch, UTC) if epoch else None


class CallTracker:
    call_type = "web"  # every call comes from a browser: dashboard, public link or widget

    def __init__(self, *, room_name: str, org_id: uuid.UUID, llm_model: str, recording_status: str) -> None:
        self.room_name = room_name
        self.org_id = org_id
        self.llm_model = llm_model
        self.started_at = datetime.now(UTC)

        self.recording_status = recording_status
        self.recording_key: str | None = None
        self.egress_id: str | None = None

        # Set by code that knows why the call ended (end_call, crash); wins over close_reason.
        self.end_reason: str | None = None
        self.close_reason: str | None = None
        self.transcript: list[dict[str, Any]] = []

        # Chosen here, so bookings made during the call can reference it right away.
        self.call_id = uuid.uuid4()
        self._start_task: asyncio.Task[None] | None = None

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
        self.events.append({"timestamp": datetime.now(UTC), "type": type_, "detail": (detail or "")[:255] or None})

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

    def _start_payload(self) -> dict[str, Any]:
        return {
            "id": str(self.call_id),
            "room_name": self.room_name,
            "call_type": self.call_type,
            "llm_model": self.llm_model,
            "org_id": str(self.org_id),
            "recording_status": self.recording_status,
            "started_at": self.started_at.isoformat(),
        }

    def start(self) -> None:
        """Registers the call in the background so the greeting isn't delayed."""
        if not platform_client.configured():
            logger.info("call history disabled: PLATFORM_API_URL or INTERNAL_API_TOKEN not set")
            return
        self._start_task = asyncio.create_task(self._register())

    async def _register(self) -> None:
        try:
            await platform_client.start_call(self._start_payload())
            logger.info("call history: registered id=%s", self.call_id)
        except Exception:
            pass  # the final report re-registers it

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
                "timestamp": datetime.fromtimestamp(ev.item.created_at, UTC).isoformat(),
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
        """Sends the call's report. Called from on_session_end, which runs for every ending."""
        if self._start_task is None:
            return
        await self._start_task

        ended_at = datetime.now(UTC)
        end_reason = self.end_reason or _CLOSE_REASONS.get(self.close_reason or "", "unknown")

        def jsonable(row: dict[str, Any]) -> dict[str, Any]:
            return {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in row.items()}

        report = {
            "ended_at": ended_at.isoformat(),
            "duration_seconds": round((ended_at - self.started_at).total_seconds(), 1),
            "end_reason": end_reason,
            "transcript": self.transcript,
            "recording_status": self.recording_status,
            "recording_key": self.recording_key,
            "egress_id": self.egress_id,
            "averages": self._averages(),
            "totals": self.totals,
            "turns": [jsonable(t) for t in self.turns],
            "events": [jsonable(e) for e in self.events],
        }
        try:
            delivered = await platform_client.report_call(self.call_id, {"start": self._start_payload(), "report": report})
            logger.info(
                "call history: report %s id=%s end_reason=%s turns=%s events=%s recording=%s",
                "delivered" if delivered else "queued in the outbox",
                self.call_id, end_reason, len(self.turns), len(self.events), self.recording_status,
            )
            await platform_client.flush_outbox()
        except Exception as e:
            logger.warning("call history: could not report call %s (%s)", self.call_id, type(e).__name__)
