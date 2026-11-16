# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY app/ ./app/
COPY models/ ./models/

# Create directory for offline buffer SQLite DB and non-root appuser
RUN useradd -u 1000 -m appuser \
    && mkdir -p /data /app/outputs /app/evidences \
    && chown -R appuser:appuser /data /app

ENV OFFLINE_BUFFER_DB=/data/offline_buffer.db
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

USER appuser

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
