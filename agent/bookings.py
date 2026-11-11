"""Booking the appointment the caller agreed to, through the platform API.

The platform's database refuses overlapping bookings (exclusion constraint), so two callers asking
for the same slot at once can't both get it; the loser is offered the nearest free times.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import platform_client
from booking import BookingNotes, clean_name
from business_hours import clock, format_date
from tenant import Business

logger = logging.getLogger("talkbalt-agent")
DEFAULT_DURATION_MIN = 30


@dataclass
class BookingResult:
    status: str  # confirmed | conflict | error
    key: tuple[str, datetime]  # (service, start): the request this result answers
    booking_id: uuid.UUID | None = None
    alternatives: list[str] = field(default_factory=list)  # spoken clock times


def to_e164(digits: str) -> str:
    """Digits accepted by booking.check_phone, in E.164 (+923001234567, +12025550123)."""
    if digits.startswith("03") and len(digits) == 11:
        return "+92" + digits[1:]
    if digits.startswith("92") or (digits.startswith("1") and len(digits) == 11):
        return "+" + digits
    return "+1" + digits


def requested(business: Business, notes: BookingNotes) -> tuple[str, datetime, timedelta]:
    assert notes.day is not None and notes.time_value is not None
    service = notes.service.name if notes.service else notes.collected.get("service", "Appointment")
    duration = notes.service.duration_min if notes.service else DEFAULT_DURATION_MIN
    start = datetime.combine(notes.day.date(), notes.time_value, ZoneInfo(business.timezone))
    return service, start, timedelta(minutes=duration)


async def book(business: Business, notes: BookingNotes, call_id: uuid.UUID | None, source: str) -> BookingResult:
    service_name, start, length = requested(business, notes)
    key = (service_name, start)
    try:
        result = await platform_client.book(
            {
                "org_id": str(business.org_id),
                "call_id": str(call_id) if call_id else None,
                "service_id": str(notes.service.id) if notes.service else None,
                "service_name": service_name,
                "starts_at": start.isoformat(),
                "duration_min": int(length.total_seconds() // 60),
                "customer_name": clean_name(notes.collected["name"]),
                "customer_phone": to_e164(notes.collected["phone"]),
                "source": source,
            }
        )
    except Exception as e:
        logger.warning("booking failed: %s", type(e).__name__)
        return BookingResult("error", key)
    alternatives = [clock(time.fromisoformat(t)) for t in result.get("alternatives", [])]
    booking_id = uuid.UUID(result["booking_id"]) if result.get("booking_id") else None
    if result["status"] == "confirmed":
        logger.info("booking confirmed: id=%s start=%s", booking_id, start.isoformat())
    return BookingResult(result["status"], key, booking_id=booking_id, alternatives=alternatives)


def booking_note(result: BookingResult, notes: BookingNotes) -> str:
    """The line the agent reads about the booking it just tried."""
    service, start = result.key
    when = f"{format_date(start)} at {clock(start)}"
    if result.status == "confirmed":
        return (
            f"The appointment is now booked and confirmed in the system: {service} on {when} for "
            f"{clean_name(notes.collected['name'])}. Tell the caller it is confirmed, repeating these details in one "
            "sentence, then ask if there is anything else."
        )
    if result.status == "conflict":
        free = " or ".join(result.alternatives)
        offer = f"Free times that day: {free}. Apologize and offer them." if free else (
            "There are no other free times that day. Apologize and ask for another day."
        )
        return f"Someone else just booked {when}, so it is no longer free. {offer}"
    return (
        "The booking could not be saved because of a system problem. Apologize, and say a team member will "
        "call to confirm the appointment."
    )
