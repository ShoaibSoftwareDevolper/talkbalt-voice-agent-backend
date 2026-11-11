#!/usr/bin/env bash
# Runs the voice agent for development: its own Postgres and Redis containers, its migrations, and
# the LiveKit agent worker. The platform API (platform-api/scripts/dev.sh) must be running too.
#
# First time only:
#   PG_CONTAINER=talkbalt-voice-postgres PG_VOLUME=talkbalt-voice-pgdata PG_DB=talkbalt_voice \
#     PG_URL_VAR=VOICE_DATABASE_URL bash scripts/setup_docker_db.sh
#   bash scripts/setup_docker_redis.sh talkbalt-voice-redis 6383
set -euo pipefail
cd "$(dirname "$0")/.."

docker start talkbalt-voice-postgres talkbalt-voice-redis >/dev/null
uv run alembic upgrade head
exec uv run agent/agent.py dev
