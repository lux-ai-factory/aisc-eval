# Production Dockerfile for A4S Eval


# ----- BASE
# Base stage with common components
FROM python:3.12-slim-bookworm AS base

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /bin/uv


# ----- Builder
FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.4.9 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
COPY uv.lock pyproject.toml /app/
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-install-project --no-dev
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen --no-dev


# ----- Test
FROM base AS test
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
  uv sync --frozen
CMD ["pytest", "--maxfail=1", "--disable-warnings"]


# ----- Production
# Final stage for runtime
FROM builder AS prod

# Copy built application from builder
COPY --from=builder /app /app

WORKDIR /app

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# # Set up entrypoint script
# RUN chmod +x entrypoint.sh
# ENTRYPOINT ["./entrypoint.sh"]

# Expose API port
EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "a4s_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
