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

# The code sits at /app/apps/eval, as in the aisc repo, because pyproject.toml takes the
# shared packages from ../../shared: that path must resolve, to /app/shared, which compose
# mounts. They are installed from the mount when the container starts (the compose
# command), so the build leaves them out. The environment stays at /app/.venv.
WORKDIR /app/apps/eval
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Builder stage for dependencies and compilation
FROM base AS builder

# Install production dependencies with caching
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev \
        --no-install-package aisc-plugin-interface --no-install-package aisc-plugin-manager

# Copy application code
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
        --no-install-package aisc-plugin-interface --no-install-package aisc-plugin-manager

# Create necessary directories
RUN mkdir -p /app/data /app/logs && \
    chmod 755 /app/data /app/logs

# Final stage for runtime
FROM base AS runtime

ENV UV_NO_SYNC=1

# Copy built application from builder
COPY --from=builder /app /app

# Add virtual environment to PATH and set Python environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "celery", "-A", "aisc_eval.celery_worker:celery_app", "worker", "--loglevel=debug"]

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD uv run celery --app aisc_eval.celery_app inspect ping -d "celery@$$HOSTNAME"
