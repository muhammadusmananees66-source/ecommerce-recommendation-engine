#!/bin/bash
# Polls dependent services before starting the app. Actually invoked from
# docker/entrypoint.sh (Dockerfile CMD), not left as dead code.
set -e

TIMEOUT="${WAIT_TIMEOUT:-60}"
INTERVAL="${WAIT_INTERVAL:-2}"

# Services to wait for, as HOST:PORT. Only includes what this specific
# deployment actually depends on -- Postgres is deliberately absent because
# nothing in this codebase currently uses it; add it here if/when it does.
SERVICES=(
    "${REDIS_HOST:-redis}:${REDIS_PORT:-6379}"
)

echo "Waiting for dependent services (timeout ${TIMEOUT}s): ${SERVICES[*]}"

for SERVICE in "${SERVICES[@]}"; do
    HOST="${SERVICE%%:*}"
    PORT="${SERVICE##*:}"
    START=$(date +%s)

    until nc -z "$HOST" "$PORT" 2>/dev/null; do
        NOW=$(date +%s)
        if [ $((NOW - START)) -gt "$TIMEOUT" ]; then
            echo "Timed out waiting for $HOST:$PORT" >&2
            exit 1
        fi
        echo "  ...waiting for $HOST:$PORT"
        sleep "$INTERVAL"
    done
    echo "  $HOST:$PORT is ready"
done

echo "All dependent services are ready."