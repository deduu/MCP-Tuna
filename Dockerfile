# ============================================================================
# Transcendence — Multi-stage Dockerfile
#
# Build targets:
#   docker build -t transcendence .                    # GPU (default)
#   docker build --target cpu -t transcendence-cpu .   # CPU only
# ============================================================================

# ---------- GPU image (default) ----------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install Python 3.11 + uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip curl ca-certificates \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra all-servers

COPY . .
RUN uv sync --frozen --no-dev --extra all-servers

EXPOSE 8000
CMD ["uv", "run", "transcendence-gateway", "http", "--port", "8000"]

# ---------- CPU image (data + eval only, no torch) ----------
FROM python:3.11-slim AS cpu

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra data --extra eval

COPY . .
RUN uv sync --frozen --no-dev --extra data --extra eval

EXPOSE 8000
CMD ["uv", "run", "transcendence-gateway", "http", "--port", "8000"]
