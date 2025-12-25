#!/bin/bash

# Trap SIGINT (Ctrl+C) and forward to process group
trap 'kill 0' SIGINT

# Start frontend in background
(cd ~/objarc/viz && VITE_API_BASE_URL=http://localhost:8010/arc/api npm run dev) &

# Start backend in background (mirror server command; isolate uv cache to avoid permission issues)
(cd ~/objarc/src && UV_CACHE_DIR=/tmp/uv-cache uv run uvicorn api:app --host 127.0.0.1 --port 8010 --proxy-headers --forwarded-allow-ips=*) &

# Wait for all background jobs
wait
