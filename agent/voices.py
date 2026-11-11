"""The receptionist voices a business can choose in its dashboard (Profile → Receptionist voice).

Cartesia voices through LiveKit Inference; every one speaks English, Urdu and Hindi. The keys match
the platform's profile setting, and the frontend plays a sample of each (public/voices/), made with
the same model, voice and language settings as a call. IDs checked against LiveKit Inference on
2026-11-11.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Voice:
    cartesia_id: str
    gender: str  # "female" | "male": Urdu and Hindi verbs change with the speaker's gender


VOICES = {
    "kiara": Voice("f8f5f1b2-f02d-4d8e-a40d-fd850a487b3d", "female"),
    "janvi": Voice("7ea5e9c2-b719-4dc3-b870-5ba5f14d31d8", "female"),
    "nora": Voice("f4c1a0b2-669d-403f-b440-4b34b34856aa", "female"),
    "devansh": Voice("1259b7e3-cb8a-43df-9446-30971a46b8b0", "male"),
    "krishna": Voice("c63361f8-d142-4c62-8da7-8f8149d973d6", "male"),
    "daniel": Voice("47c38ca4-5f35-497b-b1a3-415245fb35e1", "male"),
}
DEFAULT_VOICE = "kiara"


def voice_for(key: str | None) -> Voice:
    return VOICES.get(key or "", VOICES[DEFAULT_VOICE])
