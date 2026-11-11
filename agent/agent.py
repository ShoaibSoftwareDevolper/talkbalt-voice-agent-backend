"""Talkbalt voice agent worker: each LiveKit job is one browser call for one business.

Flow: read the business id from the dispatch metadata -> load its setup from the platform API
(before connecting, so the agent only appears once it can listen) -> wait for the caller -> run the
receptionist -> at session end, send the call report to the platform.
"""

import asyncio
import contextlib
import logging
import os
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    inference,
    llm,
    room_io,
)
from livekit.agents import tts as agents_tts
from livekit.agents.llm import LLMError
from livekit.agents.metrics import LLMMetrics
from livekit.plugins import ai_coustics, deepgram, groq, silero

from call_history import CallTracker
from groq_client import groq_client
from pipeline_events import PipelinePublisher
from receptionist import Receptionist
from recording import final_recording_status, start_recording
from tenant import Business, current_business, load_business, org_id_from_metadata
from voices import voice_for

load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

logger = logging.getLogger("talkbalt-agent")
logger.setLevel(logging.INFO)

AGENT_NAME = "talkbalt-agent"
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or "openai/gpt-oss-20b"
# Speech in English, Urdu and Hindi. STT: Deepgram Nova-3 in multilingual mode (English and Hindi with
# code-switching; spoken Urdu comes out as Hindi, which the agent reads fine), so callers are understood
# in all three. TTS: Cartesia Sonic 3.6 through LiveKit Inference (no extra API key), in the language
# and voice the business chose in its dashboard (voices.py).
TTS_MODEL = os.getenv("TTS_MODEL", "").strip() or "cartesia/sonic-3.6"
# Bounds runaway replies. gpt-oss reasoning tokens count toward it (~160 at reasoning_effort=low).
MAX_COMPLETION_TOKENS = 256
LOG_TRANSCRIPTS = os.getenv("LOG_TRANSCRIPTS", "true").strip().lower() in ("1", "true", "yes", "on")
RECORDING_ENABLED = os.getenv("RECORDING_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


class _DropTranscripts(logging.Filter):
    """Removes what callers said from SDK logs: LiveKit puts it only in `lk.pii.*` extra keys."""

    def filter(self, record: logging.LogRecord) -> bool:
        for key in [k for k in record.__dict__ if k.startswith("lk.pii.")]:
            delattr(record, key)
        return True


if not LOG_TRANSCRIPTS:
    # A logger's filters only see that exact logger's records, so attach to each.
    for name in ("livekit.agents", "livekit.plugins.deepgram", "livekit.plugins.groq", "livekit.plugins.openai",
                 "livekit.plugins.silero", "livekit.plugins.ai_coustics"):
        logging.getLogger(name).addFilter(_DropTranscripts())


def build_llm(tracker: CallTracker, on_metrics: Callable[[LLMMetrics], None]) -> llm.LLM:
    """Two instances of the same Groq model: gpt-oss sometimes emits a tool call Groq rejects with
    a 400 the SDK won't retry, and the second instance gives that turn one more attempt."""
    instances = [groq.LLM(model=LLM_MODEL, max_completion_tokens=MAX_COMPLETION_TOKENS, client=groq_client()) for _ in range(2)]

    def on_error(ev: LLMError) -> None:
        # Type and status only: Groq error bodies can echo the model's output.
        status = getattr(ev.error, "status_code", None)
        logger.warning("groq error (recoverable=%s): %s status=%s", ev.recoverable, type(ev.error).__name__, status)
        tracker.add_event("llm_error", f"{type(ev.error).__name__} status={status} recoverable={ev.recoverable}")

    for instance in instances:
        instance.on("error", on_error)
        instance.on("metrics_collected", on_metrics)
    adapter = llm.FallbackAdapter(instances)
    adapter.on(
        "llm_availability_changed",
        lambda ev: tracker.add_event("llm_fallback", f"llm instance {'recovered' if ev.available else 'marked unavailable'}"),
    )
    return adapter


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer()
server.setup_fnc = prewarm

# One call per job, keyed by job id so on_session_end finds its call.
_trackers: dict[str, CallTracker] = {}
_recording_tasks: dict[str, asyncio.Task[None]] = {}


async def finalize_call(ctx: JobContext) -> None:
    """on_session_end: runs for every ending (goodbye, hang-up, crash)."""
    tracker = _trackers.pop(ctx.job.id, None)
    recording_task = _recording_tasks.pop(ctx.job.id, None)
    if tracker is None:
        return
    if recording_task is not None:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(recording_task, timeout=10)
    if tracker.egress_id:
        tracker.recording_status = await final_recording_status(ctx.api, tracker.egress_id)
        tracker.add_event("recording_status_change", f"at call end: {tracker.recording_status}")
    await tracker.finalize()


async def record_call(ctx: JobContext, tracker: CallTracker) -> None:
    result = await start_recording(ctx.api, ctx.room.name)
    tracker.recording_status, tracker.recording_key, tracker.egress_id = result.status, result.key, result.egress_id
    tracker.add_event("recording_status_change", f"starting -> {result.status}")


@server.rtc_session(agent_name=AGENT_NAME, on_session_end=finalize_call)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    org_id = org_id_from_metadata(ctx.job.metadata)
    business = await load_business(org_id) if org_id else None
    if business is None:
        logger.warning("no business for this call (org_id=%s); ending it", org_id)
        await ctx.delete_room()
        return
    current_business.set(business)
    try:
        await run_session(ctx, business)
    except Exception:
        # A failed setup would otherwise leave the caller in a silent room.
        logger.exception("call setup failed, ending the call")
        if tracker := _trackers.get(ctx.job.id):
            tracker.end_reason = "error"
        await ctx.delete_room()
        ctx.shutdown(reason="entrypoint error")


async def run_session(ctx: JobContext, business: Business) -> None:
    await ctx.wait_for_participant()
    logger.info("call started: business=%s model=%s", business.org_id, LLM_MODEL)

    tracker = CallTracker(
        room_name=ctx.room.name,
        org_id=business.org_id,
        llm_model=LLM_MODEL,
        recording_status="starting" if RECORDING_ENABLED else "disabled",
    )
    _trackers[ctx.job.id] = tracker
    tracker.start()
    if RECORDING_ENABLED:
        # In the background, so a slow or failing egress never delays the call.
        _recording_tasks[ctx.job.id] = asyncio.create_task(record_call(ctx, tracker))

    pipeline = PipelinePublisher(ctx.room)

    def on_llm_metrics(m: LLMMetrics) -> None:
        tracker.on_llm_metrics(m)
        pipeline.on_llm_metrics(m)

    # Keyterms bias recognition toward the words this caller is most likely to need heard right.
    keyterms = [business.name, *(s.name for s in business.services), *(p.name for p in business.products)]
    # Urdu businesses get Deepgram's Urdu model, which writes Urdu script (the multilingual model writes
    # Urdu speech as Hindi, and sometimes guesses other scripts). Hindi and English use the multilingual
    # model, which follows callers switching between Hindi and English. Both take keyterms.
    stt = deepgram.STT(model="nova-3", language="ur" if business.language == "ur" else "multi", keyterm=keyterms[:50])
    speech = inference.TTS(model=TTS_MODEL, voice=voice_for(business.voice).cartesia_id, language=business.language)
    # If LiveKit Inference can't speak (outage, or the plan's inference credit is used up), Deepgram
    # Aura takes over: English only, but never a silent call.
    tts = agents_tts.FallbackAdapter([speech, deepgram.TTS(model="aura-2-asteria-en")])
    stt.on("metrics_collected", tracker.on_stt_metrics)
    tts.on("metrics_collected", tracker.on_tts_metrics)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt,
        llm=build_llm(tracker, on_llm_metrics),
        tts=tts,
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            # Short live replies ("Hello.", "No.") are often "unsure" turns; waiting at most 2s keeps
            # the agent responsive while leaving room for mid-sentence pauses.
            endpointing={"mode": "dynamic", "max_delay": 2.0},
            # Every discarded draft is a full LLM request against Groq's tokens-per-minute limit.
            preemptive_generation={"max_retries": 1},
        ),
    )
    session.on("user_state_changed", pipeline.on_user_state)
    session.on("conversation_item_added", pipeline.on_item_added)
    session.on("conversation_item_added", tracker.on_item_added)
    session.on("close", tracker.on_close)

    await session.start(
        room=ctx.room,
        agent=Receptionist(business, tracker, speech),
        room_options=room_io.RoomOptions(
            # A closing session hangs up instead of leaving the caller in silence.
            delete_room_on_close=True,
            audio_input=room_io.AudioInputOptions(noise_cancellation=ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_L)),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
