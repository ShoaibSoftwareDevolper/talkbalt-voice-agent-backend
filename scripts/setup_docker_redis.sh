#!/usr/bin/env bash
# Starts (or reuses) a password-protected Redis container for this service and writes REDIS_URL
# into .env.local (that line only). The password is never printed. Idempotent.
#
# Usage (from the project root): bash scripts/setup_docker_redis.sh <container-name> <host-port>
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?container name}"
PORT="${2:?host port}"
IMAGE="redis:7.4.9-alpine"
ENV_FILE=".env.local"

if ! docker container inspect "$NAME" >/dev/null 2>&1; then
  password="$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')"
  docker run -d --name "$NAME" --restart unless-stopped \
    -p "127.0.0.1:${PORT}:6379" -v "${NAME}-data:/data" \
    "$IMAGE" redis-server --appendonly yes --requirepass "$password" >/dev/null
  echo "Created container $NAME on 127.0.0.1:${PORT}."
else
  docker start "$NAME" >/dev/null
  echo "Container $NAME already exists: reusing it."
fi

# The password lives in the container's own command line, so a reused container needs nothing stored elsewhere.
password="$(docker inspect -f '{{json .Config.Cmd}}' "$NAME" | python3 -c 'import json,sys; c=json.load(sys.stdin); print(c[c.index("--requirepass")+1])')"
host_port="$(docker port "$NAME" 6379/tcp | head -1 | sed 's/.*://')"

touch "$ENV_FILE"
tmp="$(mktemp)"
grep -v '^REDIS_URL=' "$ENV_FILE" > "$tmp" || true
printf 'REDIS_URL=redis://:%s@127.0.0.1:%s/0\n' "$password" "$host_port" >> "$tmp"
mv "$tmp" "$ENV_FILE"
chmod 600 "$ENV_FILE"

for _ in $(seq 1 20); do
  if docker exec "$NAME" redis-cli --no-auth-warning -a "$password" ping 2>/dev/null | grep -q PONG; then
    echo "Wrote REDIS_URL to ${ENV_FILE}. Redis is answering on port ${host_port}."
    exit 0
  fi
  sleep 0.5
done
echo "Redis did not answer in time." >&2
exit 1
