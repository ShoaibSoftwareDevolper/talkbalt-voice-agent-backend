# Talkbalt backend

Python backend for Talkbalt, a voice AI receptionist built on [LiveKit Agents](https://docs.livekit.io/agents/).

| Stage | Provider |
| --- | --- |
| VAD | Silero (local) |
| STT | Deepgram `nova-3` (streaming) |
| LLM | Groq `openai/gpt-oss-20b` |
| TTS | Deepgram Aura-2 `aura-2-asteria-en` |
| Turn detection | LiveKit audio `TurnDetector` (falls back to local `v1-mini` model) |
| Noise cancellation | ai-coustics `QUAIL_L` background noise suppression (LiveKit Cloud rooms only) |

The agent registers with the dispatch name `talkbalt-agent` for explicit dispatch.

## Layout

```
backend-web/
├── agent/          # LiveKit voice agent (worker process)
│   └── agent.py
├── api/            # FastAPI server: browser tokens with agent dispatch
│   └── main.py
├── .env.example
└── pyproject.toml
```

## Setup

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/).

```bash
cd backend-web
uv sync
cp .env.example .env.local   # then fill in the values
```

`.env.local` needs:

| Variable | Where to get it |
| --- | --- |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | LiveKit Cloud project → Settings → API keys (or `lk app env -w`) |
| `DEEPGRAM_API_KEY` | https://console.deepgram.com |
| `GROQ_API_KEY` | https://console.groq.com/keys |
| `LLM_MODEL` | Optional, default `openai/gpt-oss-20b`. Any Groq model ID, for A/B testing without code changes. |
| `LOG_TRANSCRIPTS` | Optional, default `true`. Set `false` to strip transcripts, spoken text and tool arguments (the SDK's `lk.pii.*` fields) from logs. Latency lines are unaffected. |
| `CORS_ORIGINS` | API only. Comma-separated browser origins allowed to call it, default `http://localhost:3000`. Add your domain later, e.g. `http://localhost:3000,https://talkbalt.online`. |

## Download model files

Run this once (and again after you upgrade dependencies). It downloads the local models (Silero VAD and so on):

```bash
uv run agent/agent.py download-files
```

## Console mode (local, in your terminal)

Talk to the agent through your mic and speakers without joining a LiveKit room:

```bash
uv run agent/agent.py console
```

Useful flags: `--text` (text-only, or toggle with `Ctrl+T`), `--list-devices`, `--input-device`, `--output-device`.

## Dev mode (connects to LiveKit Cloud)

Registers the worker with your LiveKit project, with auto-reload and debug logging:

```bash
uv run agent/agent.py dev
```

To run everything at once (Postgres container, migrations, API on 8001 and the agent; Ctrl+C stops both):

```bash
scripts/dev.sh
```

The agent uses explicit dispatch, so it won't join rooms by itself. To test it, start a call from the web frontend (see the API below), open the [Agent Console](https://cloud.livekit.io) for your project and set the agent name to `talkbalt-agent`, or use a SIP dispatch rule.

## API (FastAPI)

Issues LiveKit tokens to the browser frontend. Run it from `backend-web`:

```bash
uv run uvicorn api.main:app --port 8001 --reload
```

| Endpoint | Description |
| --- | --- |
| `GET /api/health` | Returns `{"status": "ok"}`. |
| `POST /api/token` | Returns `{ serverUrl, participantToken, roomName, participantName }`. |

Each token request creates a new room name (`talkbalt-<uuid>`) and embeds an explicit dispatch of `talkbalt-agent` in the token's `RoomConfiguration`. LiveKit only applies that configuration when the room is created, which is why every call gets a fresh room. Tokens expire after 10 minutes, which only limits the initial connection. The API secret stays on the server. The endpoint has no authentication yet, so anyone who can reach it can start a call.

## Phone calls (LiveKit Phone Number)

Inbound calls to `+12678019838` reach the agent through the dispatch rule `talkbalt-inbound`, a callee rule that puts every call in its own room named `call-<dialed number>_<random>` and explicitly dispatches `talkbalt-agent`. Web calls use `talkbalt-<uuid>` rooms, so the two never clash. A callee rule is used instead of an individual rule because an individual rule can put the caller's phone number in the room name, and LiveKit logs room names without redaction.

Set it up (idempotent; read-only unless you pass `--apply`):

```bash
uv run scripts/setup_inbound_number.py           # show the current state
uv run scripts/setup_inbound_number.py --apply   # create the rule and assign it to the number
```

If the API refuses the assignment, the script prints dashboard steps: Telephony → Phone Numbers → the number's ⋮ menu → Assign dispatch rule → `talkbalt-inbound`.

For phone callers the agent uses Krisp VIVA telephony voice isolation (counts toward the Build plan's 100 voice isolation minutes per month); web callers keep ai-coustics `QUAIL_L`. Each call logs `call started: type=phone|web room=...`, plus the dialed number for phone calls. The caller's number is only logged when `LOG_TRANSCRIPTS=true`.

## Conversation safeguards

These run in code, because prompts alone weren't reliable with gpt-oss on Groq:

- **One question per turn:** `Talkbalt.llm_node` drops everything after the first question mark or line break before it is spoken, so the model can't ask several questions or write the caller's answers. Logged as `reply truncated after first question`.
- **Token cap:** `max_completion_tokens=256` per reply. Reasoning tokens count toward it, so it isn't set lower.
- **Booking notes** (`agent/booking.py`): rebuilt from the conversation before every reply and added as a system note (never stored). They track the name, service, date, time and phone collected so far, so the agent doesn't re-ask, and:
  - phone numbers: accepts Pakistani mobiles (11 digits starting 03, or +92 and 10 digits) and US numbers (10 digits, or +1 and 10), otherwise asks again; gives the exact digit-by-digit read-back, e.g. "zero three zero zero, one two three, four five six seven";
  - days and times (`agent/business_hours.py`): resolves "tomorrow", "this Sunday", "22 October", "nine in the evening"; refuses closed days, past times and times outside opening hours, and suggests the nearest open days.
- **Date and hours in the prompt:** the current date and time and the coming week's opening hours (`BUSINESS_TIMEZONE`, `BUSINESS_HOURS`, e.g. `Mon-Sat 10:00-20:00`) are appended after the static instructions, so the static part stays a cacheable prefix for Groq's automatic prompt cache.
- **Groq rate limits** (`agent/groq_client.py`): a 429 is retried once after Groq's `retry-after` inside the HTTP client, and the agent says "One moment please." while it waits (logged as a `rate_limited` event). Every request sends the same `tool_choice`, and preemptive drafts are capped at one per turn, to save the free tier's 8,000 tokens per minute.
- **Goodbye guard:** `end_call` only proceeds when the caller's last message is a clear closing (including "no thanks", "that's it", Allah hafiz, bas shukriya, ...), or a plain "no"/"nothing"/"I'm good" right after "anything else?". Phrases like "wait", "one more", "ek minute" always win.

## Call history and recordings

Every call (web or phone) gets a row in the `calls` table: type, timing, end reason, model, per-turn transcript with latency, averages, and recording status. The agent creates the row when the call starts and finalizes it in `on_session_end`, which runs for every ending (goodbye, hangup, crash). Database errors are logged and never affect the call. Phone numbers are stored masked (`***1234`).

When `RECORDING_ENABLED=true`, the agent starts an audio-only egress (StartEgress with an audio-only `TemplateSource`) that uploads `recordings/<room_name>.ogg` to the R2 bucket. If it can't start (R2 not configured, Build plan egress allowance used up), the call continues and `recording_status` is `failed`. Console mode never records.

Database setup (on Railway just set `DATABASE_URL`). Locally, Postgres runs in Docker, no sudo needed:

```bash
bash scripts/setup_docker_db.sh   # one time (idempotent): container + DATABASE_URL in .env.local
uv run alembic upgrade head       # create/upgrade tables; run after every pull with new migrations
```

`setup_docker_db.sh` starts `talkbalt-postgres` (`postgres:18`, volume `talkbalt-pgdata`, restart `unless-stopped`) on `127.0.0.1` at the first free port from 5433, with user and database `talkbalt` and a random password. It never prints the password. Re-running it reuses the existing container.

History API (needs the `X-Admin-Token: <ADMIN_TOKEN>` header):

| Endpoint | Description |
| --- | --- |
| `GET /api/calls?limit=20&offset=0` | Newest first, summary fields only. |
| `GET /api/calls/{id}` | Full details including transcript and per-turn latency. |
| `GET /api/calls/{id}/recording` | `{url, expires_in}`: a presigned R2 URL valid for 5 minutes. |

## Metrics, events and cost estimates

The agent buffers metrics in memory during a call and writes them once in `on_session_end`, in the same transaction that finalizes the call (nothing touches the audio path):

- `call_turn_metrics`: one row per user or agent turn. User turns have STT audio seconds, end-of-turn and transcription delays. Agent turns have LLM requests, first token, duration, prompt/cached/completion/reasoning tokens, tokens per second, TTS first audio, characters and audio seconds, end-to-end latency, `interrupted` and `cancelled`. Sources: per-plugin `metrics_collected` (LLM, STT, TTS) and `ChatMessage.metrics`. Reasoning tokens come from Groq's `completion_tokens_details` and are already included in completion tokens. A request cut short (truncation net, interruption) can end before Groq reports usage, so its tokens are null.
- `call_events`: truncation_net, end_call_blocked/allowed, llm_error, llm_fallback, interruption, recording_status_change. Details never contain caller content.
- Per-call totals and `estimated_cost_usd` on `calls`, from `config/pricing.toml`. **Fill in the TODO prices there** (Groq, Deepgram, LiveKit pricing pages); unset components are listed as missing instead of guessed.

API (X-Admin-Token): `GET /api/calls/{id}/metrics`, `GET /api/analytics?from=YYYY-MM-DD&to=YYYY-MM-DD` (daily aggregates in `BUSINESS_TIMEZONE`, latency avg and p95 per stage), `GET /api/usage` (this month vs free tiers in `config/pricing.toml`; counted by Talkbalt, not fetched from providers).

## Pipeline events (for the web visualizer)

During a call the agent sends small JSON messages to the browser on the text stream topic `talkbalt.pipeline`. They carry only states and numbers, never transcript text:

```json
{"type": "user_state", "state": "speaking"}
{"type": "latency", "role": "user", "stt_ms": 210, "eou_ms": 450}
{"type": "latency", "role": "agent", "llm_ttft_ms": 180, "tts_ttfb_ms": 120, "e2e_ms": 950}
```

`user_state` comes from Silero VAD; the latency values are the same per-turn metrics as the log lines below. In console mode there is no room, so nothing is sent.

## Latency logs

After each turn the agent logs per-stage latency, taken from LiveKit's per-turn `ChatMessage.metrics`:

```
latency user turn: stt=180ms end_of_turn=420ms
latency agent turn: llm_ttft=210ms tts_ttfb=150ms e2e=900ms
```

- `stt`: time from end of speech to the final transcript
- `end_of_turn`: time from end of speech to the end-of-turn decision
- `llm_ttft`: time to the LLM's first token
- `tts_ttfb`: time to the first TTS audio
- `e2e`: time from the user stopping to the agent starting to speak
