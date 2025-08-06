#!/bin/bash
# Script to start the development server for A4S Evaluation Service
# This script:
# 1. Validates and loads environment variables (priority: .env.dev-local > .env.dev)
# 2. Starts the uvicorn development server with hot-reloading

# Load and validate environment variables
# Priority: .env.dev-local > .env.dev
if [ -f ".env.dev-local" ] && [ -s ".env.dev-local" ]; then
  if grep -qE '^[a-zA-Z_][a-zA-Z0-9_]*=.*' .env.dev-local; then
    # Export non-comment lines as environment variables
    export $(grep -v '^#' .env.dev-local | xargs)
    echo "Loaded environment from .env.dev-local"
  else
    echo "Malformed lines in .env.dev-local"
    exit 1
  fi
elif [ -f ".env.dev" ] && [ -s ".env.dev" ]; then
  if grep -qE '^[a-zA-Z_][a-zA-Z0-9_]*=.*' .env.dev; then
    # Export non-comment lines as environment variables
    export $(grep -v '^#' .env.dev | xargs)
    echo "Loaded environment from .env.dev"
  else
    echo "Malformed lines in .env.dev"
    exit 1
  fi
else
  echo "Neither .env.dev-local nor .env.dev found or empty"
fi

# uv run python -m a4s_eval.celery_worker

# Start Celery worker in background, redirect logs to /tmp to avoid permission issues
# Use --pool=solo to avoid multiprocessing issues that cause segmentation faults
uv run celery -A a4s_eval.celery_worker worker --loglevel=info --pool=solo --concurrency=1

# Start uvicorn development server:
# - UV package manager to run uvicorn
# - Host on all interfaces (0.0.0.0)
# - Port 8001
# - Enable auto-reload on code changes
# uv run uvicorn a4s_eval.main:app --host 0.0.0.0 --port 8001 --reload
