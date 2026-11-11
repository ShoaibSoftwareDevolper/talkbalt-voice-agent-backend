#!/usr/bin/env bash
# Local Postgres for Talkbalt in Docker, no sudo needed (idempotent).
#
# Starts a Postgres container (postgres:18, named volume, restart unless-stopped, bound to
# 127.0.0.1 on the first free port from 5433) with a random password, and writes its URL into
# .env.local (that line only). If the container already exists it is reused (started if stopped).
# Secrets are never printed.
#
# Defaults are the platform database; override for another service, e.g. the voice agent:
#   PG_CONTAINER=talkbalt-voice-postgres PG_VOLUME=talkbalt-voice-pgdata PG_DB=talkbalt_voice \
#   PG_URL_VAR=VOICE_DATABASE_URL bash scripts/setup_docker_db.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ENV_FILE=".env.local"
CONTAINER="${PG_CONTAINER:-talkbalt-postgres}"
VOLUME="${PG_VOLUME:-talkbalt-pgdata}"
IMAGE="postgres:18"
DB_USER="talkbalt"
DB_NAME="${PG_DB:-talkbalt}"
URL_VAR="${PG_URL_VAR:-DATABASE_URL}"
export URL_VAR

container_env() {
  # One environment value from the container's config (used to rebuild DATABASE_URL).
  docker inspect "$CONTAINER" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sed -n "s/^$1=//p"
}

if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "Container ${CONTAINER} already exists: reusing it."
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER")" != "true" ]]; then
    docker start "$CONTAINER" >/dev/null
    echo "Started ${CONTAINER}."
  fi
else
  port=""
  for candidate in $(seq 5433 5499); do
    if ! ss -ltn | grep -q ":${candidate} "; then port=$candidate; break; fi
  done
  [[ -n "$port" ]] || { echo "No free port between 5433 and 5499." >&2; exit 1; }

  password=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
  env_file=$(mktemp)
  chmod 600 "$env_file"
  # Passed via --env-file so the password never appears on a command line.
  printf 'POSTGRES_USER=%s\nPOSTGRES_PASSWORD=%s\nPOSTGRES_DB=%s\n' "$DB_USER" "$password" "$DB_NAME" > "$env_file"
  unset password
  # postgres 18+ keeps its data under /var/lib/postgresql, so the volume is mounted there.
  docker run -d --name "$CONTAINER" \
    --restart unless-stopped \
    --env-file "$env_file" \
    -v "${VOLUME}:/var/lib/postgresql" \
    -p "127.0.0.1:${port}:5432" \
    "$IMAGE" >/dev/null
  rm -f "$env_file"
  echo "Created ${CONTAINER} (${IMAGE}) on 127.0.0.1:${port} with volume ${VOLUME}."
fi

echo -n "Waiting for Postgres"
for _ in $(seq 1 60); do
  if docker exec "$CONTAINER" pg_isready -U "$DB_USER" -d "$DB_NAME" -q 2>/dev/null; then break; fi
  echo -n "."; sleep 1
done
echo
docker exec "$CONTAINER" pg_isready -U "$DB_USER" -d "$DB_NAME" -q || { echo "Postgres did not become ready." >&2; exit 1; }

# The URL from the container's own settings, so a reused container needs nothing stored elsewhere.
host_port=$(docker port "$CONTAINER" 5432/tcp | grep -o '127.0.0.1:[0-9]*' | head -1 | cut -d: -f2)
user=$(container_env POSTGRES_USER)
password=$(container_env POSTGRES_PASSWORD)
db=$(container_env POSTGRES_DB)
url="postgresql://${user}:${password}@127.0.0.1:${host_port}/${db}"
unset password

# Replace (or append) only the URL line.
touch "$ENV_FILE"
tmp=$(mktemp)
grep -v "^${URL_VAR}=" "$ENV_FILE" > "$tmp" || true
if [[ -s "$tmp" && -n "$(tail -c1 "$tmp")" ]]; then echo >> "$tmp"; fi
printf '%s=%s\n' "$URL_VAR" "$url" >> "$tmp"
cat "$tmp" > "$ENV_FILE"
rm -f "$tmp"
unset url
echo "Wrote ${URL_VAR} to ${ENV_FILE} (port ${host_port})."

# Verify with the app's credentials over TCP, the way the app connects.
uv run python - <<'PY'
import asyncio, os
from pathlib import Path
from dotenv import load_dotenv
import asyncpg

load_dotenv(Path(".env.local"))

async def main() -> None:
    conn = await asyncpg.connect(os.environ[os.environ["URL_VAR"]])
    row = await conn.fetchrow("SELECT current_user, current_database(), current_setting('server_version') AS v")
    await conn.close()
    print(f"Connection check passed: {row['current_user']} @ {row['current_database']} (Postgres {row['v']})")

asyncio.run(main())
PY
