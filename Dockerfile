# DubbLM Docker Image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    pkg-config \
    python3-dev \
    libsndfile1 \
    libportaudio2 \
    git \
    curl \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Deno (required JS runtime for yt-dlp YouTube challenge solving)
ARG DENO_VERSION=v2.3.0
RUN curl -fsSL https://github.com/denoland/deno/releases/download/${DENO_VERSION}/deno-x86_64-unknown-linux-gnu.zip -o /tmp/deno.zip \
    && unzip -o /tmp/deno.zip -d /usr/local/bin \
    && rm /tmp/deno.zip \
    && chmod +x /usr/local/bin/deno \
    && deno --version

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN printf "setuptools<81\n" > /tmp/pip-constraints.txt \
    && PIP_CONSTRAINT=/tmp/pip-constraints.txt pip install --no-cache-dir --upgrade "pip==25.1.1" "setuptools<81" wheel \
    && PIP_CONSTRAINT=/tmp/pip-constraints.txt pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p projects cache

# Expose port
EXPOSE 8000

# Default command (overridden in docker-compose)
CMD ["python", "run_api.py"]
