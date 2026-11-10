"""The HTTP client for Groq, built like the LiveKit plugin's default plus rate-limit handling:

- a 429 is retried once inside the transport after Groq's `retry-after`, so the SDK never sees
  it: no FallbackAdapter failover, recovery probes or session-closing error count, which on the
  free tier turned one 429 into a burst of 15 more;
- while a retry-after window is open, new requests wait for it instead of being sent to fail.
"""

import asyncio
import os
import time

import httpx2 as httpx
import openai

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
# Waits longer than this are not hidden: the 429 goes back to the SDK as before.
MAX_RATE_LIMIT_WAIT = 20.0


class RateLimitState:
    """The latest 429 seen on this process's Groq client (one call per job process)."""

    def __init__(self) -> None:
        self.retry_after: float | None = None
        self.at: float = 0.0
        self.hits = 0  # 429s seen so far; the agent compares it to log one event per wait

    def record(self, retry_after: float | None) -> None:
        self.retry_after = retry_after
        self.at = time.monotonic()
        self.hits += 1

    def seconds_to_wait(self) -> float | None:
        """Remaining wait from the most recent 429, or None if there's nothing to wait for."""
        if self.retry_after is None:
            return None
        remaining = self.retry_after - (time.monotonic() - self.at)
        return remaining if remaining > 0 else None


rate_limit = RateLimitState()


def _retry_after(response: httpx.Response) -> float | None:
    try:
        return float(response.headers.get("retry-after", "") or 0) or None
    except ValueError:
        return None


class RateLimitTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if wait := rate_limit.seconds_to_wait():
            await asyncio.sleep(min(wait, MAX_RATE_LIMIT_WAIT))
        response = await self._inner.handle_async_request(request)
        if response.status_code != 429:
            return response
        retry_after = _retry_after(response)
        rate_limit.record(retry_after)
        if retry_after is None or retry_after > MAX_RATE_LIMIT_WAIT:
            return response
        await response.aclose()
        await asyncio.sleep(retry_after)
        response = await self._inner.handle_async_request(request)
        if response.status_code == 429:
            rate_limit.record(_retry_after(response))
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def groq_client() -> openai.AsyncClient:
    # Same settings as the plugin's own default client (livekit.plugins.openai.LLM).
    limits = httpx.Limits(max_connections=50, max_keepalive_connections=50, keepalive_expiry=120)
    return openai.AsyncClient(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=GROQ_BASE_URL,
        max_retries=0,
        http_client=httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=5.0, write=5.0, pool=5.0),
            follow_redirects=True,
            transport=RateLimitTransport(httpx.AsyncHTTPTransport(limits=limits)),
        ),
    )
