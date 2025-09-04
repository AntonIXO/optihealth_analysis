# Use the official uv image with Python 3.13 (includes uv + Python)
FROM ghcr.io/astral-sh/uv:python3.13-bookworm AS base

# Ensure stdout/stderr are unbuffered and no .pyc files are written
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal). Some scientific libs may require build tooling.
# If builds succeed without these, you can comment this out to slim further.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Create and populate a project-local virtualenv with production deps
# --frozen ensures uv.lock is honored exactly
RUN uv sync --frozen --no-dev

# Copy application source
COPY . .

# Default command runs the worker
# "uv run" uses the project virtualenv created by "uv sync"
CMD ["uv", "run", "python", "main_worker.py"]
