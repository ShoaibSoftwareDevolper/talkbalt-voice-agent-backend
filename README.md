# Talkbalt voice agent

The voice side of Talkbalt, built on [LiveKit Agents](https://docs.livekit.io/agents/). It does one thing: answer each business's browser calls (dashboard test call, public link, website widget) as that business's receptionist, and book appointments. Accounts, business data, bookings and call history live in the separate platform API (`../platform-api`), reached only through its internal API.

| Stage | Provider |
| --- | --- |
| VAD | Silero (local) |
| STT | Deepgram `nova-3` (streaming) |
| LLM | Groq `openai/gpt-oss-20b` |
| TTS | Deepgram Aura-2 `aura-2-asteria-en` |
| Turn detection | LiveKit audio `TurnDetector` |
| Noise cancellation | ai-coustics `QUAIL_L` |

The worker registers as `talkbalt-agent` (explicit dispatch).

## Call flow

1. The platform API issues the call token; its agent dispatch metadata carries `{"org_id": ...}`.
2. `agent.py` loads that business's setup from the platform API **before** joining the room, so the agent only appears once it can listen. A call without a known business is ended.
3. `receptionist.py` runs the conversation; `booking.py` tracks the details, `bookings.py` books through the platform API.
4. At session end `call_history.py` sends one report (transcript, per-turn metrics, events, usage) to the platform API.

```
agent/
├── agent.py            # entrypoint: business -> session -> report
├── receptionist.py     # instructions and conversation safeguards
├── booking.py          # booking notes: details, phone checks, non-answers
├── business_hours.py   # days and times callers say, opening hours, per-call context
├── bookings.py         # booking through the platform API
├── call_history.py     # per-turn metrics, events, the call report
├── pipeline_events.py  # live VAD/latency/token events for the browser
├── platform_client.py  # the platform's internal API, last-good cache, outbox
├── tenant.py           # the call's business
├── groq_client.py      # Groq HTTP client with rate-limit handling
├── recording.py        # audio-only egress to R2
└── voice_db.py         # the agent's own database (outbox)
```

## Setup and run

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), Docker. The platform API must be running.

```bash
uv sync
cp .env.example .env.local      # then fill in the values
uv run agent/agent.py download-files

# one time: the agent's own Postgres and Redis (idempotent, 127.0.0.1 only, passwords never printed)
PG_CONTAINER=talkbalt-voice-postgres PG_VOLUME=talkbalt-voice-pgdata PG_DB=talkbalt_voice \
  PG_URL_VAR=VOICE_DATABASE_URL bash scripts/setup_docker_db.sh
bash scripts/setup_docker_redis.sh talkbalt-voice-redis 6383

scripts/dev.sh                  # containers, migrations, agent worker
```

| Variable | Description |
| --- | --- |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit Cloud project → Settings → API keys |
| `DEEPGRAM_API_KEY`, `GROQ_API_KEY` | Speech and LLM providers |
| `PLATFORM_API_URL`, `INTERNAL_API_TOKEN` | The platform API and the shared service token (same value in `platform-api/.env.local`) |
| `VOICE_DATABASE_URL`, `REDIS_URL` | The agent's own Postgres and Redis (written by the setup scripts) |
| `R2_*`, `RECORDING_ENABLED` | Call recordings (audio-only egress to Cloudflare R2) |
| `LLM_MODEL` | Optional, default `openai/gpt-oss-20b` |
| `TTS_MODEL` | Optional, default `cartesia/sonic-3.6`, through LiveKit Inference |
| `LOG_TRANSCRIPTS` | Optional, default `true`; `false` strips what callers said from SDK logs |

## Languages

Each business chooses in its dashboard which language its receptionist speaks (English, Urdu or
Hindi) and its voice (`agent/voices.py`; samples in the frontend's `public/voices/`). Callers are
understood in all three either way: STT is Deepgram Nova-3 with `language="multi"` and keyterms (the
business, service and product names), which writes Hindi and spoken Urdu in Devanagari. Each reply
is spoken in the language its script shows (`Receptionist.tts_node`). The booking checks match a glossed copy of each message, with Urdu and
Hindi keywords read as English (`agent/lang.py`), and Devanagari names are saved in Latin letters.

## Resilience

- **Business setup:** fetched fresh for every call; the voice Redis keeps the last good copy, used when the platform API is unreachable.
- **Call reports:** a report the platform can't take waits in the voice database's outbox (`pending_reports`) and is retried after the next call.
- **Recording:** if egress can't start (R2 not configured, egress allowance used up), the call goes on and the recording is marked `failed`.

## Conversation safeguards

These run in code, because prompts alone aren't reliable with gpt-oss on Groq:

- **One question per reply:** everything after the first question mark or line break is dropped before it is spoken (`truncation_net` event).
- **Booking notes:** rebuilt from the conversation before every reply and added as a system note (never stored). They track name, service, date, time and phone; validate Pakistani and US numbers with a digit-by-digit read-back; resolve "tomorrow", "this Sunday", "nine in the evening"; refuse closed days, past times and out-of-hours times.
- **Callers first:** a question, a complaint, "hello?", "sorry, what?" or a bare "no" is never saved as a detail; the agent answers, apologizes, repeats or checks whether the caller still wants to book.
- **Booking in code:** once every detail is confirmed, the booking is made through the platform API; if the slot was just taken, the agent offers the nearest free times.
- **Goodbye guard:** `end_call` only works after a clear closing ("bye", "that's all", "Allah hafiz", or "no" after "anything else?"); "wait", "one more", "ek minute" always win.
- **Groq limits:** a 429 is retried once after `retry-after` inside the HTTP client while the agent says "One moment please."; every request sends the same `tool_choice` (cacheable prefix), and preemptive drafts are capped at one per turn.
- **Turn-taking:** at most 2 seconds of silence before replying (dynamic endpointing).
- **Token cap:** 256 completion tokens per reply (reasoning included).

## Pipeline events and latency logs

During a call the agent sends small JSON messages on the text stream topic `talkbalt.pipeline` (states and numbers only, never what was said), and logs the same per-turn latency:

```json
{"type": "user_state", "state": "speaking"}
{"type": "latency", "role": "agent", "llm_ttft_ms": 180, "tts_ttfb_ms": 120, "e2e_ms": 950}
{"type": "llm_usage", "prompt_tokens": 1200, "cached_tokens": 768, "completion_tokens": 40, "reasoning_tokens": 12}
```
