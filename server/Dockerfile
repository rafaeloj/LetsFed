# Multi-stage build for optimized server container
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy application code
COPY server/ ./server/
COPY conf/ ./conf/
COPY model/ ./model/
COPY utils/ ./utils/

# Set Python path
ENV PYTHONPATH=/app

# Run server
CMD ["python", "-u", "server/strategies_manager.py"]
