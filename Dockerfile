FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    APP_MODULE=api.app:app

# System deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Expose the port (HF/Fly will set $PORT)
EXPOSE 7860

# Start the API
CMD ["bash", "-lc", "uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT}"]
