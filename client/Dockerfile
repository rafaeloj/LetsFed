# Multi-stage build for optimized client container
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
COPY requirements-client.txt .
RUN pip install --no-cache-dir -r requirements-client.txt

# Copy application code
COPY client/ ./client/
COPY conf/ ./conf/
COPY dataset_manager/ ./dataset_manager/
COPY model/ ./model/
COPY utils/ ./utils/

# Set Python path
ENV PYTHONPATH=/app

# Run client
CMD ["python", "-u", "client/strategies_manager.py"]
