"""Talkbalt voice agent: Silero VAD + Deepgram STT/TTS + Groq LLM on LiveKit Agents."""

import asyncio
import contextlib
import json
import logging
import os
import re
import sys
import time
from collections.abc import AsyncIterable, Callable
from pathlib import Path
from typing import Any

# Run as `uv run agent/agent.py`, so make the shared backend-web packages (db, storage) importable.
sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

from call_history import CallTracker, turn_metrics
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    JobProcess,
    ModelSettings,
    RunContext,
    TurnHandlingOptions,
    UserStateChangedEvent,
    inference,
    llm,
    room_io,
    utils,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.llm import ChatItem, ChatMessage, LLMError, Toolset
from livekit.agents.metrics import LLMMetrics
from livekit.agents.voice.room_io.types import NoiseCancellationParams
from booking import BookingNotes
from business_hours import business_context
from livekit.plugins import ai_coustics, deepgram, groq, krisp, silero
from groq_client import groq_client, rate_limit
from recording import final_recording_status, start_recording

# backend-web/.env.local, resolved from this file so the cwd doesn't matter.
load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

logger = logging.getLogger("talkbalt-agent")
logger.setLevel(logging.INFO)

AGENT_NAME = "talkbalt-agent"
LLM_MODEL = os.getenv("LLM_MODEL", "").strip() or "openai/gpt-oss-20b"
MAX_COMPLETION_TOKENS = 256

LOG_TRANSCRIPTS = os.getenv("LOG_TRANSCRIPTS", "true").strip().lower() in ("1", "true", "yes", "on")
# Egress uses the Build plan's small monthly transcode allowance, so recording can be switched off.
RECORDING_ENABLED = os.getenv("RECORDING_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


class _DropTranscripts(logging.Filter):
    """Strip user content from SDK log records.

    LiveKit passes transcripts, TTS text and tool arguments only through `extra` keys
    prefixed `lk.pii.`, so removing those keeps the log line but drops the content.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        for key in [k for k in record.__dict__ if k.startswith("lk.pii.")]:
            delattr(record, key)
        return True


if not LOG_TRANSCRIPTS:
    # Filters on a logger only see records created by that exact logger, so attach to each.
    for name in (
        "livekit.agents",
        "livekit.plugins.deepgram",
        "livekit.plugins.groq",
        "livekit.plugins.openai",
        "livekit.plugins.silero",
        "livekit.plugins.ai_coustics",
    ):
        logging.getLogger(name).addFilter(_DropTranscripts())

INSTRUCTIONS = """You are Talkbalt, a friendly, professional receptionist on a live phone call. Your replies are spoken aloud.
Keep every reply to one or two short sentences of plain conversational language: no markdown, lists, emojis, asterisks or other symbols.
Be warm and helpful. If you don't know something, say so briefly and offer to take a message.

Every reply you write is one turn, and then the caller speaks. Write only your own words: never write what the caller says,
never guess or invent their answers, and never continue the conversation on their behalf.
Each reply asks at most one question, as the last thing in the reply; then stop and wait for the caller to finish and answer.

Your only tool is end_call; use it only as its description says. Never call any other tool or function; otherwise always answer in plain spoken text.
You cannot book, cancel or change appointments, look anything up, or send texts or emails, and must never say or imply you did.
Never say an appointment is booked or confirmed, and never promise a confirmation text or email.

For an appointment request you need the caller's name, the service, a preferred date, a preferred time and a phone number.
Ask only for the next missing detail, using what the caller already told you. When you have all of them, repeat them back
in one or two sentences and say a team member will call to confirm.

The system may add call notes before your reply: what is collected, what is missing, and checks on the phone number.
Follow them, and never ask again for a detail they list as collected. Read phone numbers digit by digit in small groups,
exactly as the notes give them, never formatted with dashes or as hundreds and thousands."""

GREETING_INSTRUCTIONS = (
    "Greet the caller warmly, introduce yourself as Talkbalt, and ask how you can help."
)

# Appended to EndCallTool's built-in description, which already says to end the call only when
# the user is clearly done and to generate no other text alongside the tool call.
END_CALL_WHEN = """Call it only when the caller clearly wants to finish: they say goodbye, bye, that's all, that's everything
or I'm all set, or they say no after you ask if there is anything else. Not when they just say thanks mid-conversation,
are still giving details, or you are unsure; then ask if there is anything else you can help with.
Don't say goodbye yourself first: a short goodbye is spoken right after it runs, then the call ends."""

END_CALL_GOODBYE = "Say one short, warm goodbye sentence to the caller and nothing else."

END_CALL_BLOCKED = (
    "The call was NOT ended and is still active: the caller has not clearly said they are finished. "
    "Do not say goodbye. Continue the conversation. If their last message sounded cut off, "
    "briefly invite them to go on."
)

# Spoken while a reply waits out a Groq rate limit (retry-after), so the caller doesn't hear silence.
RATE_LIMIT_NOTICE = "One moment please. "

# Phrases in the caller's latest message that clearly end the call.
_FINISHED = re.compile(
    r"\b(bye|goodbye|good bye|that's all|that is all|that's everything|that's it|that is it|i'm done|i am done"
    r"|all set|nothing else|nothing[.,!]? thanks?( you)?|no[.,!]? thanks?( you)?|have a (good|nice|great) (day|night|evening|one)"
    # Urdu / Hinglish closings, in the spellings English STT tends to produce
    r"|(allah|khuda) ha(f|ph)(i|ee?)z|bas,? shukri?y?a|th(ee|i)k hai,? bas|bas itna hi)\b"
)
# Phrases that mean the caller wants to keep going; these win over any closing phrase.
_CONTINUING = re.compile(r"\b(not (done|finished|all set)|wait|hold on|one more|one (sec|second|minute)|ek minute|ruko)\b")
# Short answers that only end the call right after the agent asked whether there is anything else.
_NO_ANSWER = re.compile(r"^(no|nope|nah|not really|nothing|i'm good|i am good)\b")
_ASKED_ANYTHING_ELSE = re.compile(r"\b(anything|something) (else|more)\b")


def _normalize(text: str) -> str:
    return text.lower().replace("’", "'").strip(" \t\n.,!?")


def caller_finished(items: list[ChatItem]) -> tuple[bool, str]:
    """Decide from the chat history whether the caller's latest message ends the call."""
    messages = [i for i in items if isinstance(i, ChatMessage) and i.role in ("user", "assistant")]
    last_user = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"), None)
    if last_user is None:
        return False, "caller has not said anything yet"

    said = _normalize(messages[last_user].text_content or "")
    asked = next(
        (_normalize(m.text_content or "") for m in reversed(messages[:last_user]) if m.role == "assistant"),
        "",
    )
    quoted = f": {said!r}" if LOG_TRANSCRIPTS else ""

    if _CONTINUING.search(said):
        return False, f"caller wants to continue{quoted}"
    if _FINISHED.search(said):
        return True, "caller said a closing phrase"
    if _NO_ANSWER.match(said) and _ASKED_ANYTHING_ELSE.search(asked):
        return True, "caller said no to 'anything else'"
    return False, f"caller's last message is not a clear goodbye{quoted}"


class GuardedEndCallTool(EndCallTool):
    """EndCallTool that refuses to hang up unless the caller's latest message clearly ends the call.

    Hanging up is irreversible, so the LLM's decision alone isn't enough. This overrides the
    private EndCallTool._end_call (bound by name in its constructor); recheck it on SDK upgrades.
    """

    tracker: CallTracker | None = None  # set by Talkbalt; only used to record events

    async def _end_call(self, ctx: RunContext) -> Any | None:
        finished, reason = caller_finished(ctx.session.history.items)
        # The part before ':' never includes the caller's quoted words.
        detail = reason.split(":", 1)[0]
        if not finished:
            logger.info("end_call blocked: %s", reason)
            if self.tracker:
                self.tracker.add_event("end_call_blocked", detail)
            return END_CALL_BLOCKED
        logger.info("end_call allowed: %s", reason)
        if self.tracker:
            self.tracker.add_event("end_call_allowed", detail)
        return await super()._end_call(ctx)


class OneQuestionFilter:
    """Keeps a reply to what one spoken turn should be, while it streams.

    gpt-oss sometimes keeps writing after its question: more questions, or the caller's answers
    ("May I have your name?My name is Alex ..."). Everything after the first question mark, or
    after the first line break once some text was written, is dropped. Spoken replies never need
    either, so a normal reply passes through unchanged.
    """

    def __init__(self) -> None:
        self.kept = ""
        self.dropped = ""
        self.cut = False

    def feed(self, delta: str) -> str:
        """Returns the part of `delta` that may be spoken; empty once the reply was cut."""
        if self.cut:
            self.dropped += delta
            return ""
        for i, char in enumerate(delta):
            if char == "\n" and self.kept.strip():
                self.cut = True
            elif char == "?":
                self.kept += char
                self.cut = True
                i += 1
            if self.cut:
                self.dropped += delta[i:]
                return delta[:i]
            self.kept += char
        return delta


class Talkbalt(Agent):
    def __init__(self, call_started_at: float, tracker: CallTracker) -> None:
        end_call = GuardedEndCallTool(
            extra_description=END_CALL_WHEN,
            delete_room=True,  # also disconnects SIP callers
            end_instructions=END_CALL_GOODBYE,
            ignore_on_enter=True,  # can't hang up during the greeting
            on_tool_called=self._on_end_call,
        )
        end_call.tracker = tracker
        # Per-call context (date, time, opening hours) goes after the static instructions, so the
        # static part stays an identical prefix across calls for Groq's prompt cache.
        super().__init__(instructions=f"{INSTRUCTIONS}\n\n{business_context()}", tools=end_call.tools)
        self._call_started_at = call_started_at
        self._tracker = tracker

    async def _on_end_call(self, ev: Toolset.ToolCalledEvent) -> None:
        self._tracker.end_reason = "caller_goodbye"
        logger.info(
            "end_call fired: ending call after goodbye, duration=%.1fs",
            time.monotonic() - self._call_started_at,
        )

    async def on_enter(self) -> None:
        # The agent speaks first.
        self.session.generate_reply(instructions=GREETING_INSTRUCTIONS)

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[llm.ChatChunk]:
        # Booking notes are rebuilt from the history and added to a copy of the context, so they
        # reach the model but never the stored history. Doing it here rather than in
        # on_user_turn_completed keeps preemptive generation valid (a changed context there
        # cancels it).
        if note := BookingNotes.from_history(chat_ctx.items).render():
            chat_ctx = chat_ctx.copy()
            chat_ctx.add_message(role="system", content=note)

        # The SDK sends tool_choice="auto" on tool follow-ups but omits it on normal turns, and
        # Groq drops the prompt cache when tool_choice changes. Sending it always keeps the
        # requests identical apart from the messages.
        if tools and not utils.is_given(model_settings.tool_choice):
            model_settings = ModelSettings(tool_choice="auto")

        # Filtering here (not in tts_node) keeps the dropped text out of the speech, the
        # transcript and the chat history alike.
        guard = OneQuestionFilter()
        reply = self._announce_rate_limit_wait(Agent.default.llm_node(self, chat_ctx, tools, model_settings))
        async with contextlib.aclosing(reply) as stream:
            async for chunk in stream:
                if isinstance(chunk, str):
                    chunk = llm.ChatChunk(id="", delta=llm.ChoiceDelta(role="assistant", content=chunk))
                if self._tracker.end_reason == "caller_goodbye" and isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.tool_calls:
                    # The call is already ending. gpt-oss tends to answer the goodbye instruction by
                    # calling end_call again, which loops up to the tool-step limit and then fails
                    # with a 400, costing 4-7 requests at the end of every call.
                    chunk = chunk.model_copy(update={"delta": chunk.delta.model_copy(update={"tool_calls": []})})
                if not isinstance(chunk, llm.ChatChunk) or chunk.delta is None or not chunk.delta.content:
                    if not guard.cut:
                        yield chunk
                    continue
                kept = guard.feed(chunk.delta.content)
                if kept:
                    yield chunk.model_copy(update={"delta": chunk.delta.model_copy(update={"content": kept})})
                if guard.cut:
                    break

            if guard.cut and not guard.dropped.strip():
                # Cut exactly at a chunk boundary: read ahead until the model either stops or
                # writes more, so a real truncation is always logged. Closing the stream then
                # stops the rest of the generation.
                async for chunk in stream:
                    delta = chunk.delta if isinstance(chunk, llm.ChatChunk) else None
                    if delta and (delta.content or delta.tool_calls):
                        guard.dropped += delta.content or f"[tool call: {', '.join(t.name for t in delta.tool_calls)}]"
                        if guard.dropped.strip():
                            break

        if guard.dropped.strip():
            dropped = f": {guard.dropped.strip()[:200]!r}" if LOG_TRANSCRIPTS else ""
            logger.info("reply truncated after first question (%d chars dropped)%s", len(guard.dropped), dropped)
            self._tracker.add_event("truncation_net", f"{len(guard.dropped)} chars dropped after the first question")

    async def _announce_rate_limit_wait(self, stream: AsyncIterable[Any]) -> AsyncIterable[Any]:
        """Passes the reply through; if its first chunk is held up by a Groq rate-limit wait
        (groq_client waits out retry-after and retries once), speaks RATE_LIMIT_NOTICE first."""
        hits = rate_limit.hits
        async with contextlib.aclosing(stream) as it:  # type: ignore[type-var]
            first = asyncio.ensure_future(anext(aiter(it), None))
            try:
                announced = False
                while not first.done():
                    await asyncio.wait({first}, timeout=0.25)
                    if not first.done() and not announced and (wait := rate_limit.seconds_to_wait()):
                        announced = True
                        logger.warning("groq rate limit: waiting %.1fs (retry-after) before replying", wait)
                        self._tracker.add_event("rate_limited", f"groq 429, waiting {wait:.0f}s (retry-after)")
                        yield RATE_LIMIT_NOTICE
            finally:
                if not first.done():
                    # Let the cancelled read unwind before aclosing() closes the stream under it.
                    first.cancel()
                    with contextlib.suppress(BaseException):
                        await first
            if not announced and rate_limit.hits > hits:
                self._tracker.add_event("rate_limited", "groq 429, retried after retry-after")
            if (chunk := first.result()) is None:
                return
            yield chunk
            async for chunk in it:
                yield chunk


def log_turn_latency(ev: ConversationItemAddedEvent) -> None:
    """Logs each turn's per-stage latency (the same numbers the browser and the DB get)."""
    if isinstance(ev.item, ChatMessage) and ev.item.role in ("user", "assistant"):
        stages = " ".join(f"{k}={v if v is not None else 'n/a'}" for k, v in turn_metrics(ev.item).items())
        logger.info("latency %s turn: %s", "user" if ev.item.role == "user" else "agent", stages)


PIPELINE_TOPIC = "talkbalt.pipeline"


class PipelinePublisher:
    """Sends Silero VAD state, per-turn latency and LLM token usage to the browser's pipeline visualizer.

    Uses a text stream on PIPELINE_TOPIC (one small JSON message per event). Only states and
    numbers are sent, never transcript text.
    """

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._tasks: set[asyncio.Task[rtc.TextStreamInfo]] = set()

    def on_user_state(self, ev: UserStateChangedEvent) -> None:
        self._send({"type": "user_state", "state": ev.new_state})

    def on_llm_metrics(self, m: LLMMetrics) -> None:
        # Token counts of the latest LLM request, shown under the LLM box.
        self._send(
            {
                "type": "llm_usage",
                "prompt_tokens": m.prompt_tokens,
                "cached_tokens": m.prompt_cached_tokens,
                "completion_tokens": m.completion_tokens,
                "reasoning_tokens": m.reasoning_tokens,
            }
        )

    def on_item_added(self, ev: ConversationItemAddedEvent) -> None:
        if not isinstance(ev.item, ChatMessage) or ev.item.role not in ("user", "assistant"):
            return
        role = "user" if ev.item.role == "user" else "agent"
        self._send({"type": "latency", "role": role, **turn_metrics(ev.item)})

    def _has_web_client(self) -> bool:
        return any(
            p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_STANDARD
            for p in self._room.remote_participants.values()
        )

    def _send(self, payload: dict[str, object]) -> None:
        # Nobody to show it to on phone calls (or when the room isn't connected).
        if not self._room.isconnected() or not self._has_web_client():
            return
        task = asyncio.create_task(
            self._room.local_participant.send_text(json.dumps(payload), topic=PIPELINE_TOPIC)
        )
        self._tasks.add(task)
        task.add_done_callback(self._on_sent)

    def _on_sent(self, task: asyncio.Task[rtc.TextStreamInfo]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.debug("pipeline event not sent: %s", task.exception())


def is_phone_caller(participant: rtc.Participant) -> bool:
    return participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP


def log_call_start(room_name: str, participant: rtc.RemoteParticipant) -> None:
    if not is_phone_caller(participant):
        logger.info("call started: type=web room=%s", room_name)
        return
    # sip.trunkPhoneNumber is the number that was dialed (ours); sip.phoneNumber is the
    # caller's number, which is PII.
    dialed = participant.attributes.get("sip.trunkPhoneNumber", "unknown")
    caller = participant.attributes.get("sip.phoneNumber", "unknown") if LOG_TRANSCRIPTS else "hidden"
    logger.info("call started: type=phone room=%s dialed=%s caller=%s", room_name, dialed, caller)


def select_noise_cancellation(params: NoiseCancellationParams) -> rtc.FrameProcessor[rtc.AudioFrame]:
    if is_phone_caller(params.participant):
        # Voice isolation tuned for telephony audio, recommended for SIP participants.
        # Uses the Build plan's voice isolation allowance (100 min/month).
        return krisp.voice_isolation_telephony()
    # Background noise suppression: no extra cost on the Build plan.
    return ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_L)


def log_llm_error(ev: LLMError) -> None:
    # Fires per attempt, so failures the fallback below recovers from are still visible.
    msg = str(ev.error)
    kind = "tool-call error" if "tool" in msg.lower() else "error"
    # Groq error bodies can echo the model's output, so the detail follows LOG_TRANSCRIPTS.
    detail = msg if LOG_TRANSCRIPTS else type(ev.error).__name__
    logger.warning("groq %s (%s, recoverable=%s): %s", kind, ev.label, ev.recoverable, detail)


def build_llm(tracker: CallTracker, on_metrics: Callable[[LLMMetrics], None]) -> llm.LLM:
    # gpt-oss on Groq sometimes emits a tool call Groq rejects (for example when no tools
    # are sent) with a 400 that the SDK won't retry, so the turn goes silent.
    # A second instance of the same model gives that turn one more attempt.
    # The cap bounds runaway replies. On Groq, gpt-oss reasoning tokens count toward it, and at
    # reasoning_effort=low they reached ~160 in tests, so 256 leaves room for a 1-2 sentence reply.
    instances = [
        groq.LLM(model=LLM_MODEL, max_completion_tokens=MAX_COMPLETION_TOKENS, client=groq_client()) for _ in range(2)
    ]

    def record_error(ev: LLMError) -> None:
        status = getattr(ev.error, "status_code", None)
        kind = "tool-call error" if "tool" in str(ev.error).lower() else "error"
        # Only the error type and status, never the response body (it can echo model output).
        tracker.add_event("llm_error", f"{kind} {type(ev.error).__name__} status={status} recoverable={ev.recoverable}")

    # Per-plugin metrics (not deprecated) arrive on the instance that served the request.
    for instance in instances:
        instance.on("error", log_llm_error)
        instance.on("error", record_error)
        instance.on("metrics_collected", on_metrics)

    adapter = llm.FallbackAdapter(instances)

    @adapter.on("llm_availability_changed")
    def on_availability(ev: Any) -> None:
        index = instances.index(ev.llm) if ev.llm in instances else -1
        tracker.add_event("llm_fallback", f"llm instance {index} {'recovered' if ev.available else 'marked unavailable'}")

    return adapter


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer()
server.setup_fnc = prewarm

# One call per job; keyed by job id so on_session_end can find the call it belongs to.
_trackers: dict[str, CallTracker] = {}
_recording_tasks: dict[str, asyncio.Task[None]] = {}


async def finalize_call(ctx: JobContext) -> None:
    """on_session_end: runs after the session closes for every ending (goodbye, hangup, crash)."""
    tracker = _trackers.pop(ctx.job.id, None)
    recording_task = _recording_tasks.pop(ctx.job.id, None)
    if tracker is None:
        return
    if recording_task is not None:
        try:
            await asyncio.wait_for(recording_task, timeout=10)
        except Exception:
            pass  # start_recording never raises; this only covers a timeout
    if tracker.egress_id:
        tracker.recording_status = await final_recording_status(ctx.api, tracker.egress_id)
        tracker.add_event("recording_status_change", f"at call end: {tracker.recording_status}")
    await tracker.finalize()


@server.rtc_session(agent_name=AGENT_NAME, on_session_end=finalize_call)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    try:
        await run_session(ctx)
    except Exception:
        # Without this a failed setup leaves the caller in a silent room until they hang up.
        logger.exception("entrypoint failed, deleting room to end the call")
        if tracker := _trackers.get(ctx.job.id):
            tracker.end_reason = "error"
        await ctx.delete_room()
        ctx.shutdown(reason="entrypoint error")


async def record_call(ctx: JobContext, tracker: CallTracker) -> None:
    result = await start_recording(ctx.api, ctx.room.name)
    tracker.recording_status = result.status
    tracker.recording_key = result.key
    tracker.egress_id = result.egress_id
    tracker.add_event("recording_status_change", f"starting -> {result.status}")


async def run_session(ctx: JobContext) -> None:
    call_started_at = time.monotonic()
    logger.info("using LLM model %s", LLM_MODEL)

    # The caller: a SIP participant for phone calls, a standard one for the web frontend.
    caller = await ctx.wait_for_participant()
    log_call_start(ctx.room.name, caller)

    # Console mode has no real LiveKit room to record.
    record = RECORDING_ENABLED and not ctx.is_fake_job()
    tracker = CallTracker(
        room_name=ctx.room.name,
        call_type="phone" if is_phone_caller(caller) else "web",
        llm_model=LLM_MODEL,
        caller_number=caller.attributes.get("sip.phoneNumber"),
        recording_status="starting" if record else "disabled",
    )
    _trackers[ctx.job.id] = tracker
    tracker.start()
    if record:
        # In the background so a slow or failing egress never delays or breaks the call.
        _recording_tasks[ctx.job.id] = asyncio.create_task(record_call(ctx, tracker))

    pipeline = PipelinePublisher(ctx.room)

    def on_llm_metrics(m: LLMMetrics) -> None:
        tracker.on_llm_metrics(m)
        pipeline.on_llm_metrics(m)

    # Per-plugin metrics_collected (the documented, non-deprecated surface), buffered by the tracker.
    stt = deepgram.STT(model="nova-3", language="en")
    tts = deepgram.TTS(model="aura-2-asteria-en")
    stt.on("metrics_collected", tracker.on_stt_metrics)
    tts.on("metrics_collected", tracker.on_tts_metrics)

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=stt,
        llm=build_llm(tracker, on_llm_metrics),
        tts=tts,
        turn_handling=TurnHandlingOptions(
            # Full v1 model on LiveKit Inference while the free dev allowance lasts,
            # then falls back to the local v1-mini model automatically.
            turn_detection=inference.TurnDetector(),
            # Turns the detector is sure about commit at min_delay (0.3s); unsure ones wait up to
            # max_delay. 4.0s made every unsure turn ("Hey. Thank you.") cost 4s. Dynamic mode
            # adapts the wait to this caller's own pauses within 0.3-3.0s; 3.0s still covers more
            # than the 2.5s default for mid-sentence pauses.
            endpointing={"mode": "dynamic", "max_delay": 3.0},
            # Each preemptive attempt is a full LLM request, discarded when the transcript changes.
            # The default 3 per turn spent the free tier's 8k tokens/minute on drafts; one keeps
            # the head start for the common case of a turn that doesn't change.
            preemptive_generation={"max_retries": 1},
        ),
    )

    session.on("conversation_item_added", log_turn_latency)

    session.on("user_state_changed", pipeline.on_user_state)
    session.on("conversation_item_added", pipeline.on_item_added)

    session.on("conversation_item_added", tracker.on_item_added)
    session.on("close", tracker.on_close)

    await session.start(
        room=ctx.room,
        agent=Talkbalt(call_started_at, tracker),
        room_options=room_io.RoomOptions(
            # Errors after start close the session rather than reaching the except above;
            # deleting the room then hangs up instead of leaving the caller in silence.
            delete_room_on_close=True,
            audio_input=room_io.AudioInputOptions(
                # Picked per participant: telephony model for phone callers, QUAIL_L for web.
                noise_cancellation=select_noise_cancellation,
            ),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
