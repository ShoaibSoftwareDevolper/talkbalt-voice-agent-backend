"""Talkbalt HTTP API: issues LiveKit tokens that dispatch the voice agent into a fresh room."""

import logging
import os
import uuid
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit import api
from pydantic import BaseModel

# backend-web/.env.local, resolved from this file so the cwd doesn't matter.
load_dotenv(Path(__file__).resolve().parent.parent / ".env.local")

from api.analytics import router as analytics_router  # noqa: E402  (needs the env loaded)
from api.history import router as history_router  # noqa: E402

logger = logging.getLogger("talkbalt-api")

AGENT_NAME = "talkbalt-agent"
# Only the initial connection needs a valid token; LiveKit refreshes it for connected clients.
TOKEN_TTL = timedelta(minutes=10)

CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()
]

app = FastAPI(title="Talkbalt API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Admin-Token"],
)
app.include_router(history_router)
app.include_router(analytics_router)


class TokenResponse(BaseModel):
    serverUrl: str
    participantToken: str
    roomName: str
    participantName: str


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/token", status_code=201)
async def create_token() -> TokenResponse:
    server_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    if not (server_url and api_key and api_secret):
        logger.error("LIVEKIT_URL, LIVEKIT_API_KEY or LIVEKIT_API_SECRET is not set")
        raise HTTPException(status_code=500, detail="Server configuration error")

    # Every call gets a new room: RoomConfiguration (and so the agent dispatch) is only
    # applied when the room is created, and is ignored for a room that already exists.
    # The room, identity and agent are all chosen here, never by the browser.
    room_name = f"talkbalt-{uuid.uuid4().hex}"
    identity = f"caller-{uuid.uuid4().hex[:12]}"
    participant_name = "Caller"

    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(participant_name)
        .with_ttl(TOKEN_TTL)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
                can_publish_data=True,
            )
        )
        .with_room_config(
            api.RoomConfiguration(agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)])
        )
    )

    return TokenResponse(
        serverUrl=server_url,
        participantToken=token.to_jwt(),
        roomName=room_name,
        participantName=participant_name,
    )
