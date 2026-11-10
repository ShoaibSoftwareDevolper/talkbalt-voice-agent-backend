"""Booking notes rebuilt from the conversation on every LLM call.

The model is bad at remembering which details it already has and at reading phone numbers
back, so code tracks both: which detail each caller answer belongs to, and whether the phone
number heard so far is complete and valid. The result is a short system note added to the
context for the next reply only (it never enters the stored chat history).
"""

import re
from dataclasses import dataclass, field
from datetime import datetime

from business_hours import (
    DAY_OR_TIME_WORDS,
    format_date,
    business_now,
    check_day,
    check_time,
    nearest_open_days,
    resolve_day,
    resolve_times,
)
from livekit.agents.llm import ChatItem, ChatMessage

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

_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_REPEAT = {"double": 2, "triple": 3}
_SPOKEN = {d: w for w, d in _DIGIT_WORDS.items() if w not in ("oh", "o")}


def _normalize(text: str) -> str:
    return text.lower().replace("’", "'").strip()


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
            self.awaiting = []  # a question about something else, e.g. "anything else?"
        asked = self.awaiting
        self.acknowledged_field = None
        self.problem = None

        # A read-back of the number was answered.
        if self.phone_check and self.phone_check.valid and not self.phone_confirmed and "correct" in agent_text:
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

        for f in asked:
            if f not in ("phone", "date", "time") and f not in self.collected and said:
                self.collected[f] = user_text.strip()
        if any(f != "phone" for f in asked) and said:
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
            ok, label = check_time(self.day, times, self.now)
            if ok:
                self.collected["time"] = label
            else:
                self.problem = f"The caller asked for a time we can't book: {label}. Say so briefly and suggest a time within opening hours."
                self.collected.pop("time", None)
        self.awaiting = []
        return True

    def render(self) -> str | None:
        """The note for the model, or None when there is nothing booking-related yet."""
        if not (self.collected or self.phone_digits or self.phone_check or self.acknowledged_field or self.problem):
            return None
        lines = ["Call notes from the system (the caller cannot see these):"]
        given = [f"{f}: \"{v}\"" for f, v in self.collected.items() if f != "phone"]
        if "phone" in self.collected and self.phone_check:
            given.append(f"phone: {self.phone_check.spoken} ({self.phone_check.kind}{', confirmed' if self.phone_confirmed else ''})")
        if given:
            lines.append("Already collected (do not ask again): " + "; ".join(given) + ".")
        missing = [f for f in FIELDS if f not in self.collected]
        instruction = self._instruction()
        if missing:
            # A specific instruction below (wait, read back, ask again) takes priority over asking.
            lines.append("Still needed: " + ", ".join(missing) + ("." if instruction else ". Ask only for the first one."))
        elif not instruction:
            lines.append("All details collected. Repeat them back in one or two sentences, then say a team member will call to confirm.")
        if instruction:
            lines.append(instruction)
        return "\n".join(lines)

    def _instruction(self) -> str | None:
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
                f"Read the phone number back exactly like this, digit by digit: \"{self.phone_check.spoken}\", "
                "and ask if it is correct. Never say it as a formatted number."
            )
        if self.phone_digits and not self.phone_check:
            return (
                f"The caller has given {len(self.phone_digits)} digits of the phone number so far. "
                "Wait for the rest: reply only with something very short like 'Go on.'"
            )
        return None
