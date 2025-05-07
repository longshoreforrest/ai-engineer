FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    TORCH_HOME=/opt/torch

# --- OS dependencies --------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# --- Python dependencies ----------------------------------------------------
COPY docker/base_api/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# --- Application code -------------------------------------------------------
WORKDIR /app
COPY app /app

EXPOSE 8000
# Multiprocess Uvicorn provides concurrency
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4" ]