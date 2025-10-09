# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables for cache directories
ENV HF_HOME=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch
ENV XDG_CACHE_HOME=/tmp/xdg_cache
ENV PYTHONUNBUFFERED=1

# Install system dependencies for audio/video processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary folders and set permissions
RUN mkdir -p /app/logs /app/data/audio/enhanced /tmp/huggingface /tmp/torch /tmp/xdg_cache && \
    # Make directories owned by non-root user
    chown -R 1000:1000 /app/logs /app/data /tmp/huggingface /tmp/torch /tmp/xdg_cache

# Switch to non-root user to avoid permission issues
USER 1000

# Expose port
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
