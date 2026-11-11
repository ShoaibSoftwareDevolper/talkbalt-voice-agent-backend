"""Understanding days and times callers say, checking them against the business's opening hours,
and the per-call business context for the agent's instructions."""

import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from tenant import Business, current_business

_WEEKDAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8,
             "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14,
             "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18, "nineteenth": 19, "twentieth": 20,
             "thirtieth": 30}
# Words that show a caller is talking about a day or time (used to keep them out of phone numbers).
DAY_OR_TIME_WORDS = re.compile(
    r"\b(today|tonight|tomorrow|morning|afternoon|evening|night|noon|midday|o'?clock|am|pm|a\.m|p\.m|half past|quarter"
    r"|" + "|".join(_WEEKDAY_NAMES + _MONTHS) + r")\b"
)


def _business() -> Business:
    business = current_business.get()
    assert business is not None, "a call always runs for a business"
    return business


def business_now() -> datetime:
    return datetime.now(ZoneInfo(_business().timezone))


def business_hours() -> dict[int, tuple[time, time]]:
    return _business().hours


def _number_words(text: str) -> str:
    """'twenty second' -> '22nd', 'twenty two' -> '22', 'thirty' -> '30' (enough for days and minutes)."""
    def repl(m: re.Match[str]) -> str:
        tens = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}[m[1]]
        unit = m[2]
        if not unit:
            return str(tens)
        if unit in _ORDINALS:
            return f"{tens + _ORDINALS[unit]}th"
        return str(tens + _ONES.index(unit))

    text = re.sub(r"\b(twenty|thirty|forty|fifty)(?:[\s-](" + "|".join(_ONES[1:10] + list(_ORDINALS)[:9]) + r"))?\b", repl, text)
    text = re.sub(r"\b(" + "|".join(_ORDINALS) + r")\b", lambda m: f"{_ORDINALS[m[1]]}th", text)
    return re.sub(r"\b(" + "|".join(_ONES[1:]) + r")\b", lambda m: str(_ONES.index(m[1])), text)


def resolve_day(text: str, now: datetime) -> datetime | None:
    """The day a caller means ("tomorrow", "this Sunday", "next Tuesday", "22 October"), or None."""
    t = _number_words(text.lower().replace("’", "'"))
    if "day after tomorrow" in t:
        return now + timedelta(days=2)
    if "tomorrow" in t:
        return now + timedelta(days=1)
    if re.search(r"\b(today|tonight)\b", t):
        return now
    for i, name in enumerate(_WEEKDAY_NAMES):
        if re.search(rf"\b{name}\b", t):
            ahead = (i - now.weekday()) % 7
            if ahead == 0 and not re.search(r"\b(this|today)\b", t):
                ahead = 7  # "Monday" said on a Monday means next week
            return now + timedelta(days=ahead)
    month_pat = "|".join(_MONTHS)
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?: of)? ({month_pat})\b|\b({month_pat})(?: the)? (\d{{1,2}})(?:st|nd|rd|th)?\b", t)
    if m:
        day = int(m[1] or m[4])
        month = _MONTHS.index(m[2] or m[3]) + 1
        for year in (now.year, now.year + 1):
            try:
                candidate = now.replace(year=year, month=month, day=day)
            except ValueError:
                return None
            if candidate.date() >= now.date():
                return candidate
    return None


def resolve_times(text: str) -> list[time]:
    """Possible clock times in what the caller said; both AM and PM when they didn't say which."""
    t = _number_words(text.lower().replace("’", "'"))
    if re.search(r"\b(noon|midday)\b", t):
        return [time(12, 0)]
    minute = 0
    m = re.search(r"\bhalf past (\d{1,2})\b", t)
    if m:
        hour, minute = int(m[1]), 30
    else:
        m = re.search(r"\b(\d{1,2})(?:[:.](\d{2})| (\d{2})\b)?\s*(?:o'?clock)?", t)
        if not m:
            return []
        hour, minute = int(m[1]), int(m[2] or m[3] or 0)
    if hour > 23 or minute > 59:
        return []
    if hour > 12:
        return [time(hour, minute)]
    pm = re.search(r"\b(pm|p\.m|afternoon|evening|night|tonight)\b", t)
    am = re.search(r"\b(am|a\.m|morning)\b", t)
    h12 = hour % 12
    if pm:
        return [time(h12 + 12, minute)]
    if am:
        return [time(h12, minute)]
    return [time(h12, minute), time(h12 + 12, minute)]


def check_day(day: datetime, now: datetime) -> str | None:
    """Why the day can't be booked (closed or past), or None if it's fine."""
    hours = business_hours()
    if day.date() < now.date():
        return f"{format_date(day)} is in the past"
    span = hours.get(day.weekday())
    if span is None:
        return f"we are closed on {format_date(day)}"
    if day.date() == now.date() and now.time() >= span[1]:
        return "we are already closed for today"
    return None


def check_time(day: datetime | None, times: list[time], now: datetime) -> tuple[bool, str, time | None]:
    """(ok, explanation, accepted time) for the requested time on `day` (or any open day when the
    day is unknown)."""
    hours = business_hours()
    span = hours.get(day.weekday()) if day else None
    spans = [span] if span else list(hours.values())
    for t in times:
        if any(open_ <= t < close for open_, close in spans):
            if day and day.date() == now.date() and t <= now.time():
                continue
            return True, clock(t), t
    asked = " or ".join(clock(t) for t in times)
    if span:
        return False, f"{asked} is outside opening hours ({clock(span[0])} to {clock(span[1])}) or has already passed", None
    return False, f"{asked} is outside opening hours", None


def nearest_open_days(now: datetime, after: datetime | None = None, count: int = 2) -> str:
    """The next open days from now (or after `after`), e.g. 'Monday, 26 October or Tuesday, 27 October'."""
    start = after or now
    found = []
    for offset in range(1 if after else 0, 15):
        day = start + timedelta(days=offset)
        if check_day(day, now) is None:
            found.append(format_date(day))
            if len(found) == count:
                break
    return " or ".join(found)


def clock(t: time | datetime) -> str:
    return t.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")


def format_date(d: datetime) -> str:
    return f"{d:%A}, {d.day} {d:%B}"


def business_context(now: datetime | None = None) -> str:
    """The per-call part of the instructions. It goes after the static instructions, so those stay an
    identical prefix across calls for Groq's prompt cache."""
    business = _business()
    now = now or business_now()
    hours = business.hours

    today = hours.get(now.weekday())
    if today is None:
        today_status = "closed today"
    elif now.time() >= today[1]:
        today_status = "already closed for today"
    elif now.time() >= today[0]:
        today_status = f"open now until {clock(today[1])}"
    else:
        today_status = f"opens today at {clock(today[0])}"

    def day_entry(offset: int) -> str:
        day = now + timedelta(days=offset)
        span = hours.get(day.weekday())
        label = f"{'tomorrow ' if offset == 1 else ''}{day:%a} {day.day} {day:%b}"
        return f"{label} {clock(span[0])}-{clock(span[1])}" if span else f"{label} closed"

    lines = [f"You answer calls for {business.name}; introduce yourself as its receptionist."]
    if business.about:
        lines[0] += f" About the business: {business.about.strip()}"
    lines.append(business.location())
    if business.services:
        lines.append(f"Bookable services: {'; '.join(s.describe() for s in business.services)}. Answer questions about services from this list only.")
    else:
        lines.append("You do not have the list of services or prices. If asked, say a team member can share the details, and offer to take a message.")
    if business.products:
        lines.append(f"Products for sale (callers can ask, not book): {'; '.join(p.describe(with_duration=False) for p in business.products)}.")
    contact = "; ".join(f"{k}: {v}" for k, v in (("website", business.website), ("email", business.contact_email)) if v)
    if contact:
        lines.append(f"Contact details you may share: {contact}.")
    if business.faqs:
        lines.append("Answers to common questions (use them, in your own words): " + " ".join(f"Q: {q} A: {a}" for q, a in business.faqs))
    lines += [
        f"Right now it is {format_date(now)} {now.year}, {clock(now)} ({now.tzinfo}); {today_status}.",
        f"Opening hours for the coming days: {'; '.join(day_entry(i) for i in range(1, 8))}.",
        "Only accept appointment times on open days, within opening hours (an appointment must start before closing "
        "time), and not in the past. If the caller asks for a closed day, a time outside opening hours, or a past time, "
        "say so briefly and suggest the nearest open time. Use the dates above for words like tomorrow or next Tuesday, "
        "and always say a date with its weekday, for example Friday, 24 October.",
    ]
    return "\n".join(lines)
