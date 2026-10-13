#!/bin/bash
set -e

REMOTE_DIR="/content/open-enricher"
TARBALL_PATH="/content/open-enricher.tgz"

echo "[INFO] Force-stopping existing FastAPI workers to free up VRAM and port 8000..."
pkill -9 -f "uvicorn" || true
sleep 2

echo "[INFO] Extracting new codebase..."
mkdir -p "$REMOTE_DIR"
tar -xzf "$TARBALL_PATH" -C "$REMOTE_DIR"

cd "$REMOTE_DIR"

# 1. Conditional Virtual Environment creation
if [ ! -f "venv/bin/pip" ]; then
    rm -rf venv
    echo "[INFO] Creating isolated Virtual Environment..."
    pip3 install virtualenv -q
    virtualenv venv --system-site-packages -q
    
    echo "[INFO] Upgrading core tools and patching setuptools..."
    ./venv/bin/pip install --upgrade pip wheel "setuptools<82" -q
else
    echo "[INFO] Virtual Environment already exists. Skipping creation."
fi

# 2. Dependency check
echo "[INFO] Verifying Python dependencies..."
./venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 > install.log 2>&1 || { tail -n 30 install.log; exit 1; }
./venv/bin/pip install --upgrade "transformers>=4.57.0" ctranslate2 faster-whisper accelerate bitsandbytes >> install.log 2>&1 || { tail -n 30 install.log; exit 1; }
echo "[INFO] Dependencies verified!"

# 3. Environment Variables & Cache Configuration
echo "[INFO] Injecting correct Database Cache paths for Hugging Face..."
export HF_HOME="/content/drive/MyDrive/Open-Enricher-Database/Models-Cache/huggingface"
export TORCH_HOME="/content/drive/MyDrive/Open-Enricher-Database/Models-Cache/torch"
export HF_HUB_DISABLE_SYMLINKS=1
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

# 4. Background server startup
echo "[INFO] Starting new FastAPI instance in the background..."

./venv/bin/python -c "import torch; print(f'\n[DIAGNOSTIC] PyTorch GPU Available: {torch.cuda.is_available()} | Device count: {torch.cuda.device_count()}\n')"

nohup ./venv/bin/python -m uvicorn src.open_enricher.remote.server:app --host 0.0.0.0 --port 8000 >> server.log 2>&1 &

echo "[SUCCESS] Deployment successful! API is now listening."