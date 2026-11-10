import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Call(Base):
    """One web or phone call handled by the agent.

    `transcript` is a list of {role, text, timestamp, metrics} entries, where `metrics` holds
    that turn's latency in ms (user: stt_ms, eou_ms; agent: llm_ttft_ms, tts_ttfb_ms, e2e_ms).
    `caller_number_masked` keeps only the last 4 digits of a phone caller's number.
    """

    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    room_name: Mapped[str] = mapped_column(String(255), index=True)
    call_type: Mapped[str] = mapped_column(String(16))  # web | phone
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[float | None] = mapped_column(Float)
    # caller_goodbye | user_hangup | error | shutdown | unknown
    end_reason: Mapped[str | None] = mapped_column(String(32))
    llm_model: Mapped[str | None] = mapped_column(String(128))

    recording_key: Mapped[str | None] = mapped_column(String(512))
    # disabled | starting | recording | processing | completed | failed
    recording_status: Mapped[str] = mapped_column(String(16), default="disabled")
    egress_id: Mapped[str | None] = mapped_column(String(64))

    avg_stt_ms: Mapped[int | None] = mapped_column(Integer)
    avg_llm_ttft_ms: Mapped[int | None] = mapped_column(Integer)
    avg_tts_ttfb_ms: Mapped[int | None] = mapped_column(Integer)
    avg_e2e_ms: Mapped[int | None] = mapped_column(Integer)
    turn_count: Mapped[int] = mapped_column(Integer, default=0)

    transcript: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    caller_number_masked: Mapped[str | None] = mapped_column(String(16))

    # Usage totals (null for calls recorded before metrics capture existed).
    total_prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    total_cached_tokens: Mapped[int | None] = mapped_column(Integer)
    total_completion_tokens: Mapped[int | None] = mapped_column(Integer)  # includes reasoning
    total_reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    llm_requests: Mapped[int | None] = mapped_column(Integer)
    total_stt_audio_seconds: Mapped[float | None] = mapped_column(Float)
    total_tts_characters: Mapped[int | None] = mapped_column(Integer)
    total_tts_audio_seconds: Mapped[float | None] = mapped_column(Float)
    cache_hit_ratio: Mapped[float | None] = mapped_column(Float)  # cached / prompt tokens
    # Estimated from config/pricing.toml; null when no price is configured for any component.
    estimated_cost_usd: Mapped[float | None] = mapped_column(Float)
    cost_breakdown: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class CallTurnMetric(Base):
    """Metrics for one conversation turn (a user or agent message) of a call.

    Agent turns carry the LLM and TTS usage of every request made while producing them (a tool
    call and its follow-up, a discarded preemptive generation), summed; `llm_requests` counts them.
    """

    __tablename__ = "call_turn_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    turn_index: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(8))  # user | agent
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    stt_audio_seconds: Mapped[float | None] = mapped_column(Float)
    eou_delay_ms: Mapped[int | None] = mapped_column(Integer)
    transcription_delay_ms: Mapped[int | None] = mapped_column(Integer)

    llm_requests: Mapped[int | None] = mapped_column(Integer)
    llm_ttft_ms: Mapped[int | None] = mapped_column(Integer)
    llm_duration_ms: Mapped[int | None] = mapped_column(Integer)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    prompt_cached_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)  # includes reasoning
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer)
    tokens_per_second: Mapped[float | None] = mapped_column(Float)

    tts_ttfb_ms: Mapped[int | None] = mapped_column(Integer)
    tts_characters: Mapped[int | None] = mapped_column(Integer)
    tts_audio_seconds: Mapped[float | None] = mapped_column(Float)
    e2e_ms: Mapped[int | None] = mapped_column(Integer)

    interrupted: Mapped[bool] = mapped_column(Boolean, default=False)
    cancelled: Mapped[bool] = mapped_column(Boolean, default=False)  # an LLM request was cut short


class CallEvent(Base):
    """Something notable during a call. `detail` is short and never contains caller content."""

    __tablename__ = "call_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    # truncation_net | end_call_blocked | end_call_allowed | llm_error | llm_fallback |
    # interruption | rate_limited | recording_status_change
    type: Mapped[str] = mapped_column(String(32), index=True)
    detail: Mapped[str | None] = mapped_column(String(255))
