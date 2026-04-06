FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    MAX_CONCURRENT_ENVS=50

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD sh -c 'curl -fsS http://localhost:${PORT}/health || exit 1'

EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host ${HOST} --port ${PORT}"]
