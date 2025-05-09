# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP=app.py
ENV PORT=5000
ENV HOST=0.0.0.0

# Create a non-root user and group
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# Install system dependencies required by opencv-python and other libraries
# libgl1-mesa-glx is for OpenGL, often needed by OpenCV GUI components.
# If using opencv-python-headless, this might not be strictly necessary.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    # libglib2.0-0 is a common dependency for many libraries
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker build cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container and set ownership
# This ensures that the application files are owned by the non-root user
COPY --chown=appuser:appgroup . .

# Create necessary directories for runtime data and set their ownership to appuser.
# This allows the non-root user (appuser) to write logs, uploads, results,
# and potentially create default model/config files if they are missing and
# not provided by mounted volumes.
RUN mkdir -p /app/uploads /app/results /app/logs /app/models/pretrained /app/config && \
    chown -R appuser:appgroup /app/uploads /app/results /app/logs /app/models /app/config

# Switch to the non-root user
USER appuser

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application using Gunicorn
# Using sh -c allows environment variable expansion for $PORT
# exec ensures Gunicorn replaces the shell, becoming PID 1 for signal handling
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app"]
