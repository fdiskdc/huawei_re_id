# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install curl for downloading uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock .python-version ./

# Set uv environment for non-root venv location (avoid bind mount conflict)
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies (no dev, no compile for CPU-only)
RUN uv sync --locked --no-dev --no-install-project

# Copy application source
COPY . .

# Compile LinearFold
RUN cd LinearFold && make

# Set runtime environment variables
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1
ENV REDIS_HOST=redis

EXPOSE 8000

# Default startup: Gunicorn
CMD ["uv", "run", "gunicorn", "-w", "1", "-b", "0.0.0.0:8000", "--timeout", "120", "wsgi:app"]
