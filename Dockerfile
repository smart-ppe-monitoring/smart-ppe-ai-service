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

# Create directory for offline buffer SQLite DB
RUN mkdir -p /data
ENV OFFLINE_BUFFER_DB=/data/offline_buffer.db

CMD ["python", "-m", "app.main"]
