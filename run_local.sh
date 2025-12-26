#!/bin/bash

set -euo pipefail

cleanup() {
  # Find any still-running background jobs; if none, skip.
  local pids
  pids=$(jobs -p)
  if [[ -z "${pids}" ]]; then
    return
  fi

  # Try gentle first, then escalate to ensure long-running work stops.
  kill -INT -- -$$ 2>/dev/null || true
  sleep 0.3
  kill -TERM -- -$$ 2>/dev/null || true
  sleep 0.3
  kill -KILL -- -$$ 2>/dev/null || true
}

# Trap Ctrl+C / SIGTERM and on shell exit; send signals to entire process group.
trap cleanup INT TERM EXIT

# Start frontend in background
(cd ~/objarc/viz && VITE_API_BASE_URL=http://localhost:8010/arc/api npm run dev) &

# Start backend in background (mirror server command; isolate uv cache to avoid permission issues)
(cd ~/objarc/src && UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn api:app --host 127.0.0.1 --port 8010 --proxy-headers --forwarded-allow-ips=*) &

# Wait for all background jobs
wait
