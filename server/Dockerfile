# Multi-stage build for MLTriageEnv
# Compatible with: standalone Docker, openenv build, HF Spaces

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app/env

WORKDIR /app/env

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.115.0" \
    "pydantic>=2.0.0" \
    "uvicorn>=0.24.0" \
    "requests>=2.31.0" \
    "openai>=1.0.0" \
    "python-dotenv>=1.0.0"

# Final runtime stage
FROM ${BASE_IMAGE}

WORKDIR /app

# Copy installed packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy environment code
COPY --from=builder /app/env /app/env

# Set PYTHONPATH so imports work
ENV PYTHONPATH="/app/env:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
