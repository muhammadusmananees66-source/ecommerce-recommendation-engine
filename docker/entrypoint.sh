#!/bin/bash
# Entrypoint: wait for dependencies, then start the API server.
# This is what actually gets invoked (see Dockerfile CMD) -- in earlier
# drafts of this project, wait_for_services.sh existed but nothing called it.
set -e

if [ "${SKIP_WAIT_FOR_SERVICES:-false}" != "true" ]; then
    bash /app/scripts/wait_for_services.sh
fi

exec uvicorn src.api.app:create_app --factory --host 0.0.0.0 --port 8000