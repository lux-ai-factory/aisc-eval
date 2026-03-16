# ---- Base ----
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Install system dependencies for production
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    procps \
    netcat-openbsd \
    git \
    gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Builder stage for dependencies and compilation
FROM base AS builder

ENV UV_HTTP_TIMEOUT=300

# Install production dependencies with caching
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy application code
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --upgrade-package a4s-plugin-manager \
        --no-dev

# Final stage for runtime
FROM base AS runtime

ENV UV_NO_SYNC=1

# Copy built application from builder
COPY --from=builder /app /app
WORKDIR /app

# Add virtual environment to PATH and set Python environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Create necessary directories
RUN mkdir -p /app/data /app/logs && \
    chmod 755 /app/data /app/logs


CMD ["uv", "run", "celery", "-A", "a4s_eval.celery_worker:celery_app", "worker", "--loglevel=debug"]

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD uv run celery --app a4s_eval.celery_app inspect ping -d "celery@$$HOSTNAME"
