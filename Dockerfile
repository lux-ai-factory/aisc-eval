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

# Install production dependencies with caching
COPY pyproject.toml uv.lock ./
# the plugin interface, plugin-manager and aisc_identity wheels the lock names (vendor/, not PyPI or GitHub)
COPY vendor/ ./vendor/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Copy application code
COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --upgrade-package aisc-plugin-manager \
        --no-dev

# Create necessary directories
RUN mkdir -p /app/data /app/logs && \
    chmod 755 /app/data /app/logs

# Final stage for runtime
FROM base AS runtime

ENV UV_NO_SYNC=1

# Plugins run as this user, not as the root worker (aisc_eval/plugin_process.py): the kernel
# then refuses them the worker's /proc/<pid>/environ, and /app, root's only, is closed to them.
RUN groupadd --system --gid 10001 aisc-plugin \
    && useradd --system --uid 10001 --gid aisc-plugin --no-user-group \
        --create-home --home-dir /home/aisc-plugin --shell /usr/sbin/nologin aisc-plugin \
    && chmod 700 /home/aisc-plugin
# Plugins are installed as this one, neither root nor the plugin user: an install runs the package's own
# build code, and the install cache in its home must be out of a running plugin's reach.
RUN groupadd --system --gid 10002 aisc-install \
    && useradd --system --uid 10002 --gid aisc-install --no-user-group \
        --create-home --home-dir /home/aisc-install --shell /usr/sbin/nologin aisc-install \
    && chmod 700 /home/aisc-install

# Copy built application from builder
COPY --from=builder /app /app
RUN chmod 700 /app

# Add virtual environment to PATH and set Python environment variables
ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "celery", "-A", "aisc_eval.celery_worker:celery_app", "worker", "--loglevel=debug"]

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD uv run celery --app aisc_eval.celery_app inspect ping -d "celery@$$HOSTNAME"
