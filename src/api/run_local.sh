#!/bin/bash

# Trap SIGINT (Ctrl+C) and forward to process group
trap 'kill 0' SIGINT

# Start frontend in background
(cd ~/objarc/viz && npm run dev) &

# Start backend in background
(cd ~/objarc/api && uv run app.py) &

# Wait for all background jobs
wait