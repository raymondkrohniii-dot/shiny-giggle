FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

# System deps
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY motion_server.py .

# Run server
CMD ["uvicorn", "motion_server:app", "--host", "0.0.0.0", "--port", "8000"]
