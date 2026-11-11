"""The receptionist: its instructions and the code-level safeguards around the LLM.

- Booking notes (booking.py) are rebuilt from the history before every reply and the booking is
  made in code (bookings.py) once the caller has confirmed every detail.
- One question per reply: anything after the first question or line break is dropped.
- The hang-up tool only works after a clear goodbye from the caller.
- A Groq rate-limit wait is announced instead of leaving the caller in silence.
"""

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import AsyncIterable
from typing import Any

from livekit import rtc
from livekit.agents import Agent, ModelSettings, RunContext, llm, tts, utils
from livekit.agents.beta.tools import EndCallTool
from livekit.agents.llm import ChatItem, ChatMessage, Toolset

from booking import BookingNotes
from bookings import BookingResult, book, booking_note, requested
from business_hours import business_context
from call_history import CallTracker
from groq_client import rate_limit
from lang import gloss, script_language
from tenant import Business
from voices import voice_for

logger = logging.getLogger("talkbalt-agent")

INSTRUCTIONS = """You are a friendly, professional receptionist on a live call for the business named at the end. Your replies are spoken aloud.
Keep every reply to one or two short sentences of plain conversational language: no markdown, lists, emojis, asterisks or other symbols.
Be warm and helpful. If you don't know something, say so briefly and offer to take a message.

Every reply you write is one turn, and then the caller speaks. Write only your own words: never write what the caller says,
never guess or invent their answers, and never continue the conversation on their behalf.
Each reply asks at most one question, as the last thing in the reply; then stop and wait for the caller to finish and answer.
Listen to what the caller actually says and respond to that first, like a real receptionist: answer their question,
and if they are upset, apologize sincerely and show you understood. Never just repeat your previous question when
the caller said something else. Only start collecting booking details after the caller asks for an appointment.

Your only tool is end_call; use it only as its description says. Never call any other tool or function; otherwise always answer in plain spoken text.
You cannot book, cancel or change appointments yourself, look anything up, or send texts or emails, and must never say or
imply you did. When every detail is in, the system may book the appointment and the call notes then say it is confirmed:
only then may you tell the caller it is booked. Never promise a confirmation text or email.

For an appointment request you need the caller's name, the service, a preferred date, a preferred time and a phone number.
Ask only for the next missing detail, using what the caller already told you. When you have all of them, follow the call notes.

The system may add call notes before your reply: what is collected, what is missing, and checks on the phone number.
Follow them, and never ask again for a detail they list as collected. Read phone numbers digit by digit in small groups,
exactly as the notes give them, never formatted with dashes or as hundreds and thousands."""

# The language the business chose for its receptionist. Callers are understood in all three either way.
LANGUAGE_RULES = {
    "en": "This receptionist speaks only English. Write every reply in English, whatever language the caller uses.",
    "ur": "This receptionist speaks only Urdu. Write every reply in Urdu, in Urdu script (never Roman Urdu, never Devanagari), "
          "whatever language the caller uses. Keep service and product names as they are written. Say numbers, prices and "
          "times the way an Urdu speaker says them.",
    "hi": "This receptionist speaks only Hindi. Write every reply in Hindi, in Devanagari (never Roman Hindi, never Urdu "
          "script), whatever language the caller uses. Keep service and product names as they are written. Say numbers, "
          "prices and times the way a Hindi speaker says them.",
}
# Callers who speak another language are still understood, but told once which language this line uses.
OTHER_LANGUAGE_RULE = (
    "If the caller speaks a language other than {language} (for example {others}), you still understand them: the first "
    "time it happens, kindly tell them in one short sentence, in {language}, that this line is only in {language}; then "
    "answer what they asked, in {language}. Do not repeat that notice later in the call."
)
# The business writes its details (about, FAQs, address) in whichever language is easiest for it;
# they are facts to pass on, and are always spoken in the receptionist's language.
BUSINESS_INFO_RULE = (
    "The business details below may be written in English, Urdu or Hindi. They are facts, not wording to read out: "
    "always say them in {language}, in your own natural words."
)
_OTHERS = {"en": "Urdu or Hindi", "ur": "English or Hindi", "hi": "English or Urdu"}
_PLAIN_NAMES = {"en": "English", "ur": "Urdu", "hi": "Hindi"}
SPEAKER = {
    "female": "You speak as a woman: in Urdu and Hindi use feminine forms for yourself (for example sakti hoon, karti hoon).",
    "male": "You speak as a man: in Urdu and Hindi use masculine forms for yourself (for example sakta hoon, karta hoon).",
}
LANGUAGE_NAMES = {"en": "English", "ur": "Urdu, in Urdu script", "hi": "Hindi, in Devanagari"}

GREETING_INSTRUCTIONS = (
    "Greet the caller warmly in {language}, say the business name, introduce yourself as its receptionist, and ask how you can help."
)

# Added to EndCallTool's built-in description.
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

# Said while a Groq rate-limit wait holds up the reply, in the receptionist's language.
RATE_LIMIT_NOTICE = {
    "en": "Just a moment, please. ",
    "ur": "براہِ مہربانی ایک لمحہ انتظار کیجیے۔ ",
    "hi": "कृपया एक पल रुकिए। ",
}

# The caller's latest message clearly ends the call (English, plus Urdu closings as STT spells them).
_FINISHED = re.compile(
    r"\b(bye|goodbye|good bye|that's all|that is all|that's everything|that's it|that is it|i'm done|i am done"
    r"|all set|nothing else|nothing[.,!]? thanks?( you)?|no[.,!]? thanks?( you)?|have a (good|nice|great) (day|night|evening|one)"
    r"|(allah|khuda) ha(f|ph)(i|ee?)z|bas,? shukri?y?a|th(ee|i)k hai,? bas|bas itna hi)\b"
)
# The caller wants to keep going; wins over any closing phrase.
_CONTINUING = re.compile(r"\b(not (done|finished|all set)|wait|hold on|one more|one (sec|second|minute)|ek minute|ruko)\b")
# A plain "no" ends the call only right after "anything else?".
_NO_ANSWER = re.compile(r"^(no|nope|nah|not really|nothing|i'm good|i am good)\b")
_ASKED_ANYTHING_ELSE = re.compile(r"\b(anything|something) (else|more)\b")


def _normalize(text: str) -> str:
    return gloss(text).strip(" \t\n.,!?")


def caller_finished(items: list[ChatItem]) -> tuple[bool, str]:
    """Whether the caller's latest message ends the call, and why (never quoting the caller)."""
    messages = [i for i in items if isinstance(i, ChatMessage) and i.role in ("user", "assistant")]
    last_user = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].role == "user"), None)
    if last_user is None:
        return False, "caller has not said anything yet"
    said = _normalize(messages[last_user].text_content or "")
    asked = next((_normalize(m.text_content or "") for m in reversed(messages[:last_user]) if m.role == "assistant"), "")
    if _CONTINUING.search(said):
        return False, "caller wants to continue"
    if _FINISHED.search(said):
        return True, "caller said a closing phrase"
    if _NO_ANSWER.match(said) and _ASKED_ANYTHING_ELSE.search(asked):
        return True, "caller said no to 'anything else'"
    return False, "caller's last message is not a clear goodbye"


class GuardedEndCallTool(EndCallTool):
    """EndCallTool that hangs up only after the caller clearly said goodbye: hanging up can't be
    undone, so the LLM's decision alone isn't enough. Overrides the private EndCallTool._end_call;
    recheck it on SDK upgrades."""

    def __init__(self, tracker: CallTracker, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tracker = tracker

    async def _end_call(self, ctx: RunContext) -> Any | None:
        finished, reason = caller_finished(ctx.session.history.items)
        logger.info("end_call %s: %s", "allowed" if finished else "blocked", reason)
        self._tracker.add_event("end_call_allowed" if finished else "end_call_blocked", reason)
        return await super()._end_call(ctx) if finished else END_CALL_BLOCKED


class OneQuestionFilter:
    """Cuts a streaming reply after its first question mark (Latin or Urdu "؟"), or its first line break once text was
    written. gpt-oss sometimes keeps going with more questions or the caller's answers; a normal
    spoken reply never needs either, so it passes through unchanged."""

    def __init__(self) -> None:
        self.kept = ""
        self.dropped = ""
        self.cut = False

    def feed(self, delta: str) -> str:
        """The part of `delta` that may be spoken; empty once the reply was cut."""
        if self.cut:
            self.dropped += delta
            return ""
        for i, char in enumerate(delta):
            if char in "?؟" or (char == "\n" and self.kept.strip()):
                end = i + 1 if char in "?؟" else i  # the question mark itself is spoken
                self.cut = True
                self.kept += delta[i:end]
                self.dropped += delta[end:]
                return delta[:end]
            self.kept += char
        return delta


class Receptionist(Agent):
    def __init__(self, business: Business, tracker: CallTracker, speech: tts.TTS | None = None) -> None:
        end_call = GuardedEndCallTool(
            tracker,
            extra_description=END_CALL_WHEN,
            delete_room=True,
            end_instructions=END_CALL_GOODBYE,
            ignore_on_enter=True,  # can't hang up during the greeting
            on_tool_called=self._on_end_call,
        )
        language = business.language if business.language in LANGUAGE_RULES else "en"
        speaker = SPEAKER[voice_for(business.voice).gender]
        other = OTHER_LANGUAGE_RULE.format(language=_PLAIN_NAMES[language], others=_OTHERS[language])
        info = BUSINESS_INFO_RULE.format(language=_PLAIN_NAMES[language])
        super().__init__(
            instructions=f"{INSTRUCTIONS}\n\n{LANGUAGE_RULES[language]} {other} {speaker} {info}\n\n{business_context()}",
            tools=end_call.tools,
        )
        self._language = language
        self._business = business
        self._speech = speech
        self._speech_language = language
        self._tracker = tracker
        self._started = time.monotonic()
        self._booking: BookingResult | None = None

    async def _on_end_call(self, ev: Toolset.ToolCalledEvent) -> None:
        self._tracker.end_reason = "caller_goodbye"
        logger.info("ending call after goodbye, duration=%.1fs", time.monotonic() - self._started)

    async def on_enter(self) -> None:
        # The business's own greeting is spoken word for word, with no LLM request, when it is written in
        # the receptionist's language; otherwise the receptionist greets in its own words, so the call
        # never starts in one language and continues in another.
        greeting = self._business.greeting
        if greeting and script_language(greeting) in (self._language, None):
            self.session.say(greeting)
        else:
            self.session.generate_reply(instructions=GREETING_INSTRUCTIONS.format(language=LANGUAGE_NAMES[self._language]))

    async def _booking_line(self, notes: BookingNotes) -> str | None:
        """Books once the caller confirmed every detail; the line telling the model how it went."""
        if not notes.ready_to_book or notes.summarized:
            return None
        service, start, _ = requested(self._business, notes)
        # Each slot is tried once; another slot after a conflict is tried again.
        if self._booking is None or (self._booking.status != "confirmed" and self._booking.key != (service, start)):
            self._booking = await book(self._business, notes, self._tracker.call_id, self._tracker.call_type)
            status = "created" if self._booking.status == "confirmed" else self._booking.status
            self._tracker.add_event(f"booking_{status}", f"{service} at {start.isoformat()}")
        return booking_note(self._booking, notes)

    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk]:
        # Notes go into a copy of the context, so they reach the model but never the stored history,
        # and preemptive generation stays valid.
        notes = BookingNotes.from_history(chat_ctx.items)
        note = "\n".join(line for line in (notes.render(), await self._booking_line(notes)) if line)
        if note:
            chat_ctx = chat_ctx.copy()
            chat_ctx.add_message(role="system", content=note)

        # Groq drops its prompt cache when tool_choice changes, and the SDK only sends it on tool
        # follow-ups: sending it on every request keeps the prefix cacheable.
        if tools and not utils.is_given(model_settings.tool_choice):
            model_settings = ModelSettings(tool_choice="auto")

        guard = OneQuestionFilter()
        reply = self._announce_rate_limit_wait(Agent.default.llm_node(self, chat_ctx, tools, model_settings))
        async with contextlib.aclosing(reply) as stream:
            async for chunk in stream:
                if isinstance(chunk, str):
                    chunk = llm.ChatChunk(id="", delta=llm.ChoiceDelta(role="assistant", content=chunk))
                if self._tracker.end_reason == "caller_goodbye" and isinstance(chunk, llm.ChatChunk) and chunk.delta and chunk.delta.tool_calls:
                    # Already hanging up: gpt-oss answers the goodbye instruction by calling end_call
                    # again, which would loop until the tool-step limit.
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
                # Cut at a chunk boundary: read on until the model stops or writes more, so a real
                # truncation is always recorded; closing the stream then stops the generation.
                async for chunk in stream:
                    delta = chunk.delta if isinstance(chunk, llm.ChatChunk) else None
                    if delta and (delta.content or delta.tool_calls):
                        guard.dropped += delta.content or "[tool call]"
                        if guard.dropped.strip():
                            break

        if guard.dropped.strip():
            logger.info("reply truncated after the first question (%d chars dropped)", len(guard.dropped))
            self._tracker.add_event("truncation_net", f"{len(guard.dropped)} chars dropped after the first question")

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings) -> AsyncIterable[rtc.AudioFrame]:
        """Speaks each reply in the language it is written in (Urdu script, Devanagari or Latin).
        The TTS stream copies its options when it opens, so the language is set from the first
        words of the reply before the stream starts; the rest streams through unchanged."""
        head: list[str] = []
        stream = aiter(text)
        language = None
        async for chunk in stream:
            head.append(chunk)
            if language := script_language("".join(head)):
                break
        if language and language != self._speech_language and self._speech is not None:
            self._speech.update_options(language=language)  # type: ignore[attr-defined]
            self._speech_language = language
            self._tracker.add_event("speech_language", language)

        async def replay() -> AsyncIterable[str]:
            for chunk in head:
                yield chunk
            async for chunk in stream:
                yield chunk

        async for frame in Agent.default.tts_node(self, replay(), model_settings):
            yield frame

    async def _announce_rate_limit_wait(self, stream: AsyncIterable[Any]) -> AsyncIterable[Any]:
        """Passes the reply through, saying a RATE_LIMIT_NOTICE first if its first chunk is held up by
        a Groq rate-limit wait (groq_client waits out retry-after and retries once)."""
        hits = rate_limit.hits
        async with contextlib.aclosing(stream) as it:  # type: ignore[type-var]
            first = asyncio.ensure_future(anext(aiter(it), None))
            announced = False
            try:
                while not first.done():
                    await asyncio.wait({first}, timeout=0.25)
                    if not first.done() and not announced and (wait := rate_limit.seconds_to_wait()):
                        announced = True
                        logger.warning("groq rate limit: waiting %.1fs before replying", wait)
                        self._tracker.add_event("rate_limited", f"groq 429, waiting {wait:.0f}s (retry-after)")
                        yield RATE_LIMIT_NOTICE[self._language]
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
