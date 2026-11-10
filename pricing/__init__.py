"""Cost estimates from config/pricing.toml. Prices left as "TODO" are treated as unknown."""

import math
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "pricing.toml"


@lru_cache(maxsize=1)
def load_pricing() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as f:
        return tomllib.load(f)


def price(*path: str) -> float | None:
    """A configured number, or None when it's missing or still "TODO"."""
    node: Any = load_pricing()
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return float(node) if isinstance(node, (int, float)) and not isinstance(node, bool) else None


def free_tier(key: str) -> float | None:
    return price("free_tier", key)


def estimate_call_cost(
    *,
    llm_model: str | None,
    prompt_tokens: int,
    cached_tokens: int,
    completion_tokens: int,
    stt_audio_seconds: float,
    tts_characters: int,
    duration_seconds: float,
    call_type: str,
    recorded: bool,
) -> tuple[float | None, dict[str, Any]]:
    """(estimated total USD or None, breakdown). Components without a configured price are listed
    as missing instead of being guessed; the total covers only the priced components."""
    parts: dict[str, float] = {}
    missing: list[str] = []

    def add(name: str, unit_price: float | None, amount: float) -> None:
        if amount <= 0:
            return
        if unit_price is None:
            missing.append(name)
        else:
            parts[name] = round(unit_price * amount, 6)

    model = llm_model or ""
    uncached = max(prompt_tokens - cached_tokens, 0)
    add("llm_input", price("llm", model, "input_per_million"), uncached / 1e6)
    add("llm_cached_input", price("llm", model, "cached_input_per_million"), cached_tokens / 1e6)
    add("llm_output", price("llm", model, "output_per_million"), completion_tokens / 1e6)
    add("stt", price("stt", "nova-3", "per_minute"), stt_audio_seconds / 60)
    add("tts", price("tts", "aura-2", "per_1k_characters"), tts_characters / 1000)
    minutes = math.ceil(duration_seconds / 60) if duration_seconds else 0
    if call_type == "phone":
        add("phone_inbound", price("livekit", "inbound_phone_per_minute"), minutes)
    if recorded:
        add("egress", price("livekit", "egress_audio_per_minute"), minutes)

    total = round(sum(parts.values()), 6) if parts else None
    return total, {"components": parts, "missing_prices": sorted(set(missing)), "estimated": True}
