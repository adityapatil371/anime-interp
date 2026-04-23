FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY app.py .

COPY checkpoints/flownet.pkl checkpoints/flownet.pkl
COPY checkpoints/unet_best.pth checkpoints/unet_best.pth

ENTRYPOINT ["python", "scripts/interpolate.py"]
