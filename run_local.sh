#!/bin/bash

# Trap SIGINT (Ctrl+C) and forward to process group
trap 'kill 0' SIGINT

# Start frontend in background
(cd ~/objarc/viz && VITE_API_BASE_URL=http://localhost:8010/arc/api npm run dev) &

# Start backend in background
(cd ~/objarc && uv run python -m src.api.app) &

# Wait for all background jobs
wait
