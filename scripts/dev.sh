#!/usr/bin/env bash
# Runs the whole backend for development: Postgres container, API (port 8001) and the agent worker.
# Ctrl+C stops the API and the agent; the Postgres container keeps running.
#
# Usage (from anywhere):  backend-web/scripts/dev.sh
set -euo pipefail
cd "$(dirname "$0")/.."

docker start talkbalt-postgres >/dev/null
uv run alembic upgrade head

uv run uvicorn api.main:app --port 8001 --reload &
api=$!
uv run agent/agent.py dev &
agent=$!

trap 'kill $api $agent 2>/dev/null; wait' INT TERM EXIT
wait -n
