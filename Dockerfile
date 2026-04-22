FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY app.py .

# Copy model checkpoints
COPY checkpoints/flownet.pkl checkpoints/flownet.pkl
COPY checkpoints/unet_best.pth checkpoints/unet_best.pth

EXPOSE 7860

CMD ["python", "app.py", "--unet-checkpoint", "checkpoints/unet_best.pth"]