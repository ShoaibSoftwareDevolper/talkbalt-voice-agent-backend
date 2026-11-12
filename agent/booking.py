"""Booking notes rebuilt from the conversation on every LLM call.

The model is bad at remembering which details it already has and at reading phone numbers
back, so code tracks both: which detail each caller answer belongs to, and whether the phone
number heard so far is complete and valid. The result is a short system note added to the
context for the next reply only (it never enters the stored chat history).

Callers and the agent may speak English, Urdu or Hindi: every text is matched in its glossed form
(lang.gloss), where the Urdu and Hindi keywords these checks look for read as English.
"""

import difflib
import re
from dataclasses import dataclass, field
from datetime import datetime, time

from livekit.agents.llm import ChatItem, ChatMessage

import conduct
from business_hours import (
    DAY_OR_TIME_WORDS,
    business_now,
    check_day,
    check_time,
    format_date,
    nearest_open_days,
    resolve_day,
    resolve_times,
)
from lang import gloss, romanize, skeleton
from tenant import ServiceInfo, current_business

# Words that make a number a clock time ("nine in the evening"), not a day of the month.
_TIME_WORDS = re.compile(r"\b(am|pm|a\.m|p\.m|morning|afternoon|evening|night|tonight|noon|midday|o'?clock|half past)\b")

FIELDS = ("name", "service", "date", "time", "phone")

# Which detail the agent's question asks for, judged from its wording.
_ASKS = {
    "phone": re.compile(r"\b(phone|number)\b"),
    "time": re.compile(r"\btime\b"),
    "date": re.compile(r"\b(date|day)\b|\bwhen would\b"),
    "service": re.compile(r"\b(service|kind of|type of)\b"),
    "name": re.compile(r"\bname\b"),
}
# A reply that only agrees ("Yeah. Sure.") instead of answering.
_ACK_ONLY = re.compile(r"^(?:(?:yeah|yes|yep|sure|okay|ok|alright|go ahead|of course)[\s.,!]*)+$")
_DENIAL = re.compile(r"^(no|nope|wrong|incorrect|that's wrong|not correct)\b")
# Replies that answer nothing: a greeting or a line check ("Hello? Hello?", "can you hear me").
_LINE_CHECK = re.compile(r"\b(can you hear me|are you there|is anyone there|anybody there|is this working)\b")
_GREETING_WORDS = {"hello", "hi", "hey", "hallo", "helo", "hullo", "yo", "there", "salam", "salaam", "assalam",
                   "asalam", "o", "u", "wa", "alaikum", "aleikum", "assalamualaikum", "aoa"}
# The caller didn't catch the agent's last message.
_REPEAT_REQUEST = re.compile(
    r"^(?:(?:sorry|pardon(?: me)?|what|huh|excuse me|come again|say (?:that|it) again|repeat(?: that)?|please"
    r"|(?:can|could) you (?:please )?(?:repeat|say) (?:that|it)(?: again)?|i didn't (?:hear|catch) (?:you|that))\s*)+$"
)
# The caller doesn't want an appointment ("I don't want any service", "not interested").
_DECLINE = re.compile(
    r"\b(don't|do not|dont) (want|need) (any |an |a )?(service|appointment|booking)s?\b"
    r"|\bnot interested\b|\bno (service|appointment|booking)\b"
)
_BOOKING_INTENT = re.compile(r"\b(book|booking|appointment|schedule|reserve|haircut|slot)\b")
# A bare "no" (or "no no no") when the agent asked for a detail: a refusal, not the detail.
_BARE_NO = re.compile(r"^(?:(?:no|nope|nah|nahi|nahin)\s*)+$")
# Lead-ins before a name ("my name is ...", "this is ...").
_NAME_LEAD_IN = re.compile(r"^(?:(?:my name is|my name's|name is|this is|it's|it is|i am|i'm|call me)\s+)")
# A name given while upset ("I said my name is Alex!").
_UPSET_NAME = re.compile(r"^(?:my name is|my name's|name is|call me)\s+")
# The caller is asking something ("How many services do you have?", "Who are you?").
_QUESTION_START = re.compile(
    r"^(?:i said,? |i asked,? |but |so |and |okay,? |ok,? |wait,? )*"
    r"(how|what|what's|who|who's|why|where|when|which|do you|does|did you|are you|is it|is there|is this|are there"
    r"|can you|could you|can i|will you|would you|should i|have you)\b"
)
# The caller is upset with the agent: insults, swearing, "I said ...", "you're not listening".
_FRUSTRATED = re.compile(
    r"\b(fuck\w*|shit|damn|stupid|idiot|dumb|disgusting|useless|ridiculous|annoying|terrible|pathetic|nonsense"
    r"|oh my god|are you (deaf|stupid|crazy|a robot|even listening)|not listening|listen to me|i (already )?(said|told you|asked)"
    r"|what (the hell|are you doing))\b"
)
# The agent told the caller the booking went through.
_CONFIRMED = re.compile(r"\b(is confirmed|are confirmed|booked and confirmed|you're booked|you are booked|is booked)\b")
# The agent asked whether something is right ("Is that correct?", "क्या यह सही है?").
_ASKS_CORRECT = re.compile(r"\b(correct|right)\b")
# How to name each detail when asking for it again.
_FIELD_WORDS = {"name": "name", "service": "service", "date": "preferred date", "time": "preferred time",
                "phone": "phone number"}


def match_service(text: str, services: tuple[ServiceInfo, ...]) -> ServiceInfo | None:
    """The offered service the caller named ("a regular haircut" -> Haircut), or None."""
    if (found := _match_service_words(text, services)) is not None:
        return found
    # Urdu and Hindi transcripts write English service names in their own script ("ہیئر کٹ", "हेयर कट"):
    # compare the consonants of each run of words with the service's ("hrkt" ~ "hrkt").
    if re.search(r"[\u0600-\u06FF\u0900-\u097F]", text):
        return _match_service_sound(text, services)
    return None


def _match_service_sound(text: str, services: tuple[ServiceInfo, ...]) -> ServiceInfo | None:
    said = text.split()
    best, best_score = None, 0.0
    for service in services:
        target = skeleton(service.name)
        if len(target) < 2:
            continue
        size = max(1, len(service.name.split()))
        for n in {size, size + 1, max(1, size - 1)}:
            for i in range(max(1, len(said) - n + 1)):
                window = skeleton(" ".join(said[i:i + n]))
                score = difflib.SequenceMatcher(None, window, target).ratio() if window else 0.0
                if score > best_score:
                    best, best_score = service, score
    return best if best_score >= 0.75 else None


def _match_service_words(text: str, services: tuple[ServiceInfo, ...]) -> ServiceInfo | None:
    said = set(_words(_normalize(text)))
    best, best_score = None, 0
    for service in services:
        name_words = [w for w in _words(service.name.lower()) if len(w) > 2] or _words(service.name.lower())
        joined = " ".join(_words(_normalize(text)))
        if service.name.lower() in joined:
            score = 100 + len(name_words)
        else:
            # "haircuts" still matches "haircut".
            score = sum(1 for w in name_words if w in said or w + "s" in said or w.rstrip("s") in said)
        if score > best_score:
            best, best_score = service, score
    return best


def clean_name(text: str) -> str:
    """'My name is alex.' -> 'Alex'; 'मेरा नाम अहमद है' -> 'Ahmad' (in Latin letters for the dashboard)."""
    name = _NAME_LEAD_IN.sub("", _normalize(text)).strip(" .,!?")
    name = re.sub(r"^(yes|yeah|sure|okay|ok),?\s+", "", name)
    name = re.sub(r"\s+(yes|okay|sure)$", "", name)  # "अहमद जी": the respectful "ji" is not part of the name
    name = _NAME_LEAD_IN.sub("", name)
    name = romanize(name)
    return " ".join(part.capitalize() for part in name.split()[:4]) or text.strip()


def _words(text: str) -> list[str]:
    return re.sub(r"[^\w\s']", " ", text).split()


def _non_answer(said: str) -> str | None:
    """'hello' for a greeting or line check, 'repeat' for "sorry, what?", None for a real reply."""
    if not said.strip(" .,!?"):
        return None
    words = _words(_LINE_CHECK.sub(" ", said))
    if (words or _LINE_CHECK.search(said)) and set(words) <= _GREETING_WORDS:
        return "hello"
    if _REPEAT_REQUEST.match(" ".join(_words(said))):
        return "repeat"
    return None

_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_REPEAT = {"double": 2, "triple": 3}
_SPOKEN = {d: w for w, d in _DIGIT_WORDS.items() if w not in ("oh", "o")}


def _normalize(text: str) -> str:
    return gloss(text)


def extract_digits(text: str) -> tuple[str, bool]:
    """Digits spoken or written in `text` ("zero three double five ..."), and whether "plus" led them."""
    tokens = re.findall(r"\+|[a-z']+|\d+", _normalize(text))
    digits, plus, repeat = "", False, 1

    def is_digit_token(i: int) -> bool:
        return 0 <= i < len(tokens) and (tokens[i].isdigit() or tokens[i] in _DIGIT_WORDS or tokens[i] in _REPEAT)

    for i, tok in enumerate(tokens):
        if tok in ("+", "plus"):
            plus = plus or not digits
        elif tok in _REPEAT:
            repeat = _REPEAT[tok]
        elif tok.isdigit():
            digits += tok[0] * repeat + tok[1:]
            repeat = 1
        elif tok in _DIGIT_WORDS and (tok not in ("oh", "o") or is_digit_token(i - 1) or is_digit_token(i + 1)):
            # "oh"/"o" only count as zero next to other digits ("oh, my number is..." is not a digit).
            digits += _DIGIT_WORDS[tok] * repeat
            repeat = 1
        else:
            repeat = 1
    return digits, plus


@dataclass
class PhoneCheck:
    digits: str
    valid: bool
    kind: str = ""
    spoken: str = ""  # digit-by-digit read-back in small groups


def _speak(groups: list[str]) -> str:
    return ", ".join(" ".join(_SPOKEN[d] for d in g) for g in groups)


def check_phone(digits: str) -> PhoneCheck:
    """Pakistani mobile: 11 digits starting 03, or +92 and 10 digits. US: 10 digits, or +1 and 10."""
    n = len(digits)
    if n == 11 and digits.startswith("03"):
        return PhoneCheck(digits, True, "Pakistani mobile", _speak([digits[:4], digits[4:7], digits[7:]]))
    if n == 12 and digits.startswith("923"):
        rest = digits[2:]
        return PhoneCheck(digits, True, "Pakistani mobile", "plus nine two, " + _speak([rest[:3], rest[3:6], rest[6:]]))
    if n == 10 and digits[0] in "23456789":
        return PhoneCheck(digits, True, "US", _speak([digits[:3], digits[3:6], digits[6:]]))
    if n == 11 and digits[0] == "1" and digits[1] in "23456789":
        rest = digits[1:]
        return PhoneCheck(digits, True, "US", "plus one, " + _speak([rest[:3], rest[3:6], rest[6:]]))
    return PhoneCheck(digits, False)


@dataclass
class BookingNotes:
    collected: dict[str, str] = field(default_factory=dict)
    phone_digits: str = ""  # digits heard while the agent is collecting the number
    phone_check: PhoneCheck | None = None
    phone_confirmed: bool = False
    acknowledged_field: str | None = None  # the caller just agreed to give this, without giving it
    non_answer: str | None = None  # the caller's latest reply answered nothing ("hello" or "repeat")
    non_answer_streak: int = 0  # how many replies in a row answered nothing
    intent_seen: bool = False  # the caller has asked for an appointment (or started giving details)
    refused_field: str | None = None  # the caller just said a bare "no" when asked for this detail
    caller_question: bool = False  # the caller's latest reply was a question, not an answer
    frustrated: bool = False  # the caller's latest reply sounded upset with the agent
    misconduct: str | None = None  # the caller's latest reply was abusive or sexual (conduct.py)
    misconduct_count: int = 0  # how many such replies on this call
    summarized: bool = False  # the agent already told the caller the booking is confirmed
    time_value: time | None = None  # the accepted appointment time
    service: ServiceInfo | None = None  # the business's service the caller chose
    declined: bool = False  # the caller said they don't want an appointment
    # Details the agent is waiting for. Survives replies that aren't questions ("Go ahead.",
    # "Go on.") and changes only when the agent asks something else.
    awaiting: list[str] = field(default_factory=list)
    now: datetime = field(default_factory=business_now)
    day: datetime | None = None  # the resolved appointment day, once accepted
    problem: str | None = None  # why the caller's latest requested day or time can't be booked

    @classmethod
    def from_history(cls, items: list[ChatItem], now: datetime | None = None) -> "BookingNotes":
        notes = cls(now=now) if now else cls()
        last_agent = ""
        for item in items:
            if not isinstance(item, ChatMessage) or item.role not in ("user", "assistant"):
                continue
            text = item.text_content or ""
            if item.role == "assistant":
                last_agent = _normalize(text)
                notes.summarized = notes.summarized or bool(_CONFIRMED.search(last_agent))
            else:
                notes._observe(last_agent, text)
        return notes

    def _asked_fields(self, agent_text: str) -> list[str]:
        if not agent_text.rstrip().endswith("?"):
            return []
        question = agent_text.rsplit(".", 1)[-1] if "?" in agent_text else agent_text
        return [f for f in FIELDS if _ASKS[f].search(question)]

    def _observe(self, agent_text: str, user_text: str) -> None:
        said = _normalize(user_text)
        if fields := self._asked_fields(agent_text):
            self.awaiting = fields
        elif "?" in agent_text:
            # A question that names no detail ("Which would you like?", "Shall we continue?") still waits for
            # the detail asked before, unless it moves on ("Anything else?").
            moving_on = re.search(r"\b(anything|something) (else|more)\b", agent_text)
            self.awaiting = [] if moving_on else [f for f in self.awaiting if f not in self.collected]
        asked = self.awaiting
        self.acknowledged_field = None
        self.problem = None

        refused_before, self.refused_field = self.refused_field, None
        self.caller_question = False
        # Swearing at the agent or sexual talk: nothing in it is a detail, and conduct.py decides the reply.
        self.misconduct = conduct.classify(user_text)
        if self.misconduct:
            self.misconduct_count += 1
            self.frustrated = False
            self.non_answer = None
            return
        self.frustrated = bool(_FRUSTRATED.search(said))

        # A greeting, line check or "sorry, what?" answers nothing: record nothing, keep waiting.
        self.non_answer = _non_answer(said)
        if self.non_answer:
            self.non_answer_streak += 1
            return
        self.non_answer_streak = 0
        if _DECLINE.search(said):
            self.declined = True
            self.awaiting = []
            return
        if _BOOKING_INTENT.search(said):
            self.intent_seen = True
            self.declined = False
            # "I'd like to book a haircut" already names the service.
            business = current_business.get()
            if business and "service" not in self.collected and (chosen := match_service(user_text, business.services)):
                self.service = chosen
                self.collected["service"] = chosen.name
        # A bare "no" when asked for a detail is a refusal: check once whether they still want to book,
        # and treat a second "no" in a row as declining.
        bare_no = bool(_BARE_NO.match(" ".join(_words(said))))
        if refused_before:
            if bare_no:  # second "no" in a row: they don't want to book
                self.declined = True
                self.awaiting = []
                return
            # They still want to book: go back to the detail they refused.
            self.intent_seen = True
            self.awaiting = asked = [refused_before]
            if _ACK_ONLY.match(said.strip(" .!")):
                return  # a plain "yes": the agent simply asks for that detail again
        elif asked and bare_no and not _ASKS_CORRECT.search(agent_text):
            if _BOOKING_INTENT.search(agent_text) or re.search(r"\b(continue|go ahead|proceed)\b", agent_text):
                self.declined = True  # "No" to "Would you like to continue booking?"
                self.awaiting = []
            else:
                self.refused_field = asked[0]
            return

        # A read-back of the number was answered.
        if self.phone_check and self.phone_check.valid and not self.phone_confirmed and _ASKS_CORRECT.search(agent_text):
            if not _DENIAL.match(said):
                self.phone_confirmed = bool(said)
                return
            # Wrong number: start over, using any digits given with the correction.
            self.phone_digits, self.phone_check = "", None
            self.collected.pop("phone", None)
            self.awaiting = asked = ["phone"]

        if asked and _ACK_ONLY.match(said.strip(" .!")):
            self.acknowledged_field = asked[0]
            return

        # Days and times are checked in code: callers answer them out of order ("Monday then"
        # when asked for a time), and the model doesn't reliably refuse closed days or hours.
        if self._observe_day_and_time(asked, said):
            return

        # "At nine in the evening" is a time, not part of a phone number.
        digits = "" if DAY_OR_TIME_WORDS.search(said) else extract_digits(user_text)[0]
        collecting_phone = "phone" in asked or (self.phone_digits and not self.phone_check)
        if collecting_phone and digits:
            if check_phone(digits).valid:
                self.phone_digits = digits  # a complete number on its own replaces any earlier pieces
            else:
                self.phone_digits += digits
            self.phone_check = check_phone(self.phone_digits) if len(self.phone_digits) >= 10 else None
            if self.phone_check and self.phone_check.valid:
                self.collected["phone"] = self.phone_check.digits
            elif self.phone_check:  # complete but invalid: start over on the next attempt
                self.phone_digits = ""
            return
        if collecting_phone and not digits:
            self.phone_digits = ""

        # "How many services do you have?" is not a service: nothing is saved and the agent answers it first.
        if (user_text.rstrip().endswith("?") or _QUESTION_START.match(said)) and not extract_digits(user_text)[0]:
            self.caller_question = True
            return
        for f in asked:
            if f not in ("phone", "date", "time") and f not in self.collected and said:
                # An angry reply ("You are so disgusting") is not a detail; "I said my name is Alex!" still is.
                # ("This is useless" starts like "This is Alex", so an upset reply needs an explicit "my name is".)
                if self.frustrated and not (f == "name" and _UPSET_NAME.match(re.sub(r"^i (already )?(said|told you),? ", "", said))):
                    continue
                if f == "name" and _BOOKING_INTENT.search(said):
                    continue  # "I want to book a haircut" answers what they want, not their name
                if f == "name" and _non_answer(_NAME_LEAD_IN.sub("", said)) == "hello":
                    # "My name is Hello": almost certainly a mishearing, so check instead of saving it.
                    self.problem = (
                        "The name you heard was only 'hello', which is probably not their real name. Say kindly "
                        "that you did not catch their name, and ask for it again."
                    )
                    return
                if f == "service" and (business := current_business.get()) and business.services:
                    chosen = match_service(user_text, business.services)
                    if chosen is None:
                        offered = ", ".join(s.name for s in business.services)
                        self.problem = (
                            f"The caller asked for something that is not one of our services. We offer: {offered}. "
                            "Say so briefly and ask which one they would like."
                        )
                        return
                    self.service = chosen
                    self.collected[f] = chosen.name
                else:
                    self.collected[f] = user_text.strip()
                self.intent_seen = True
        if any(f != "phone" and f in self.collected for f in asked) and said:
            self.awaiting = []  # answered; the next question decides what comes next

    def _observe_day_and_time(self, asked: list[str], said: str) -> bool:
        """Handles a day and/or time in the caller's reply. True if the reply was about them."""
        day = resolve_day(said, self.now) if ("date" in asked or DAY_OR_TIME_WORDS.search(said)) else None
        times = resolve_times(said) if ("time" in asked or _TIME_WORDS.search(said)) else []
        if day is None and not times:
            return False

        if day is not None:
            if reason := check_day(day, self.now):
                self.problem = f"The caller asked for {format_date(day)}, but {reason}. Say so briefly and offer {nearest_open_days(self.now, after=day)}."
                self.collected.pop("date", None)
                self.day = None
                return True
            self.day = day
            self.collected["date"] = format_date(day)
        if times:
            ok, label, chosen = check_time(self.day, times, self.now)
            if ok:
                self.collected["time"] = label
                self.time_value = chosen
            else:
                self.time_value = None
                self.problem = f"The caller asked for a time we can't book: {label}. Say so briefly and suggest a time within opening hours."
                self.collected.pop("time", None)
        self.awaiting = []
        return True

    @property
    def ready_to_book(self) -> bool:
        """Every detail is in, the number was confirmed, and the day and time are bookable."""
        return (
            all(f in self.collected for f in FIELDS)
            and self.phone_confirmed
            and self.day is not None
            and self.time_value is not None
            and not self.declined
        )

    def render(self) -> str | None:
        """The note for the model, or None when there is nothing booking-related yet."""
        if not (self.collected or self.phone_digits or self.phone_check or self.acknowledged_field or self.problem
                or self.non_answer or self.declined or self.refused_field or self.frustrated or self.caller_question
                or self.misconduct):
            return None
        lines = ["Call notes from the system (the caller cannot see these):"]
        given = [f"{f}: \"{v}\"" for f, v in self.collected.items() if f != "phone"]
        if "phone" in self.collected and self.phone_check:
            given.append(f"phone: {self.phone_check.spoken} ({self.phone_check.kind}{', confirmed' if self.phone_confirmed else ''})")
        if given:
            lines.append("Already collected (do not ask again): " + "; ".join(given) + ".")
        missing = [f for f in FIELDS if f not in self.collected]
        instruction = self._instruction()
        if self.declined or self.misconduct:
            pass  # no booking details are asked for after the caller said no, or while they are abusive
        elif missing:
            # A specific instruction below (wait, read back, ask again) takes priority over asking.
            lines.append("Still needed: " + ", ".join(missing) + ("." if instruction else ". Ask only for the first one, after responding to anything else the caller just said."))
        elif self.summarized:
            lines.append("The appointment is booked and you already told the caller. Do not repeat the details.")
        if instruction:
            lines.append(instruction)
        return "\n".join(lines)

    def _instruction(self) -> str | None:
        if self.misconduct:
            business = current_business.get()
            return conduct.instruction(self.misconduct, self.misconduct_count, business.name if business else "this business")
        if self.frustrated:
            return (
                "The caller is upset with you. Apologize sincerely in your own words and show you understood what "
                "they said; if they asked something, answer it now. Do not repeat your previous question."
            )
        if self.caller_question:
            return (
                "The caller asked a question instead of answering. Answer it first in one short sentence (if you "
                "do not know, say so honestly and offer to take a message). Then, only if they still want to book, "
                "continue with the next missing detail."
            )
        if self.non_answer == "hello":
            if not self.intent_seen:
                if self.non_answer_streak >= 2:
                    return (
                        "The caller keeps saying only hello and may not be able to hear you. Say clearly that you can "
                        "hear them, ask if they can hear you, then ask how you can help. Do not ask for booking details."
                    )
                return (
                    "The caller only said hello or checked the line, and has not said what they need yet. Say warmly "
                    "that you can hear them and ask how you can help. Do not ask for any booking details yet."
                )
            again = (
                f"ask again for their {_FIELD_WORDS[self.awaiting[0]]}" if self.awaiting else "repeat your last question"
            )
            if self.non_answer_streak >= 3:
                return (
                    "The caller has only said hello several times in a row, so they may not be able to hear you. "
                    "Say clearly that you can hear them, ask if they can hear you, and say they can call back if the "
                    f"line is bad. Then {again}."
                )
            if self.non_answer_streak == 2:
                return (
                    "The caller said only hello again instead of answering. Like a real receptionist, point it out "
                    f"kindly (for example, that you heard hello but not their answer), then {again}."
                )
            return f"The caller only said hello; nothing new was collected. Say briefly that you can hear them, then {again}."
        if self.non_answer == "repeat":
            return "The caller did not catch your last message. Repeat your last question briefly, in simpler words."
        if self.refused_field:
            return (
                f"The caller just said no instead of giving their {_FIELD_WORDS[self.refused_field]}. Do not ask for it "
                "again yet: ask kindly whether they still want to book an appointment."
            )
        if self.declined:
            return (
                "The caller said they do not want an appointment. Do not ask for any booking details. "
                "Reply briefly and ask if there is anything else you can help with."
            )
        if self.problem:
            return self.problem
        if self.acknowledged_field:
            return (
                f"The caller just agreed to give their {self.acknowledged_field} but has not said it yet. "
                "Reply only with something very short like 'Go ahead.' Do not repeat the question."
            )
        if self.phone_check and not self.phone_check.valid:
            return (
                f"The phone number heard has {len(self.phone_check.digits)} digits, which is not a valid Pakistani "
                "mobile (11 digits starting with 0 3, or plus 9 2 and 10 digits) or US number (10 digits, or plus 1 "
                "and 10 digits). Politely say you did not get a complete number and ask them to say the full number again."
            )
        if self.phone_check and not self.phone_confirmed:
            return (
                f"Read the phone number back exactly like this, digit by digit: \"{self.phone_check.spoken}\" "
                f"(digits {' '.join(self.phone_check.digits)}; when speaking Urdu or Hindi, say each digit in that "
                "language), and ask if it is correct. Never say it as a formatted number."
            )
        if self.phone_digits and not self.phone_check:
            return (
                f"The caller has given {len(self.phone_digits)} digits of the phone number so far. "
                "Wait for the rest: reply only with something very short like 'Go on.'"
            )
        return None
