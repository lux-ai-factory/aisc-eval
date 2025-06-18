#!/bin/bash
# Script to start the development server for A4S Evaluation Service
# This script:
# 1. Loads environment variables from .env.dev file if it exists
# 2. Starts the uvicorn development server with hot-reloading

# Check if .env.dev exists and is not empty
if [[ -f ".env.dev" && -s ".env.dev" ]]; then
  # Check if file contains valid environment variable definitions
  if grep -qE '^[a-zA-Z_][a-zA-Z0-9_]*=.*' .env.dev; then
    # Export all non-comment lines as environment variables
    export $(grep -v '^#' .env.dev | xargs)
  else
    echo "Malformed lines in .env.dev"
    exit 1
  fi
else
  echo ".env.dev is missing or empty"
fi

uv run python -m celery_worker &
# Start uvicorn development server:
# - UV package manager to run uvicorn
# - Host on all interfaces (0.0.0.0)
# - Port 8001
# - Enable auto-reload on code changes
uv run uvicorn a4s_eval.main:app --host 0.0.0.0 --port 8001 --reload
