"""The business a call belongs to, and its setup, loaded once when the call starts.

Every call carries `{"org_id": ...}` in its agent dispatch metadata (set by the platform API when
it issues the call token). The setup comes from the platform API; the agent has no access to its
database.

`current_business` is a ContextVar set before the session starts, so every task the session
creates (LLM node, booking notes, tools) sees the same business without passing it around.
"""

import json
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import time

import platform_client


@dataclass(frozen=True)
class ServiceInfo:
    id: uuid.UUID
    name: str
    duration_min: int
    price_cents: int | None
    currency: str

    def describe(self, with_duration: bool = True) -> str:
        parts = [f"{self.duration_min} min"] if with_duration else []
        if self.price_cents is not None:
            parts.append(f"{self.price_cents / 100:g} {self.currency}")
        return f"{self.name} ({', '.join(parts)})" if parts else self.name


@dataclass(frozen=True)
class Business:
    org_id: uuid.UUID
    name: str
    timezone: str
    greeting: str | None
    about: str | None
    hours: dict[int, tuple[time, time]] = field(default_factory=dict)  # weekday -> (opens, closes)
    services: tuple[ServiceInfo, ...] = ()  # bookable
    products: tuple[ServiceInfo, ...] = ()  # callers can ask about these, not book them
    location_type: str = "in_person"  # in_person | remote | both
    address: str | None = None
    website: str | None = None
    contact_email: str | None = None
    faqs: tuple[tuple[str, str], ...] = ()
    language: str = "en"  # what the receptionist speaks: en | ur | hi
    voice: str = "kiara"  # a key in voices.VOICES

    def location(self) -> str:
        """Where the business serves customers, as one sentence for the agent."""
        if self.location_type == "remote":
            return "The business works remotely; there is no address to visit."
        both = " It also offers remote appointments." if self.location_type == "both" else ""
        visit = f"Customers visit at: {self.address}." if self.address else "Customers visit in person; no address is on file."
        return visit + both


current_business: ContextVar[Business | None] = ContextVar("current_business", default=None)


def org_id_from_metadata(metadata: str | None) -> uuid.UUID | None:
    try:
        value = json.loads(metadata or "{}").get("org_id")
        return uuid.UUID(value) if value else None
    except (ValueError, TypeError, AttributeError):
        return None


async def load_business(org_id: uuid.UUID) -> Business | None:
    """The business's profile, hours, services and products, from the platform API."""
    if not platform_client.configured():
        return None
    runtime = await platform_client.business_runtime(org_id)
    if runtime is None:
        return None
    hours: dict[int, tuple[time, time]] = {}
    for h in runtime["hours"]:
        # One window per day: the earliest opening and the latest closing.
        opens, closes = time.fromisoformat(h["opens"]), time.fromisoformat(h["closes"])
        if h["weekday"] in hours:
            opens, closes = min(opens, hours[h["weekday"]][0]), max(closes, hours[h["weekday"]][1])
        hours[h["weekday"]] = (opens, closes)

    def info(item: dict) -> ServiceInfo:
        return ServiceInfo(uuid.UUID(item["id"]), item["name"], item["duration_min"], item["price_cents"], item["currency"])

    return Business(
        org_id=org_id,
        name=runtime["name"],
        timezone=runtime["timezone"],
        greeting=runtime["greeting"],
        about=runtime["about"],
        hours=hours,
        services=tuple(info(i) for i in runtime["items"] if i["kind"] == "service"),
        products=tuple(info(i) for i in runtime["items"] if i["kind"] == "product"),
        location_type=runtime["location_type"],
        address=runtime["address"],
        website=runtime["website"],
        contact_email=runtime["contact_email"],
        faqs=tuple((f["q"], f["a"]) for f in runtime["faqs"]),
        # .get: a last-good copy cached before these settings existed has neither.
        language=runtime.get("language") or "en",
        voice=runtime.get("voice") or "kiara",
    )
