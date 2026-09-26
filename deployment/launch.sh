#!/bin/bash
set -e

REMOTE_DIR="/content/open-enricher"
TARBALL_PATH="/content/open-enricher.tgz"

echo "[INFO] Stopping existing FastAPI workers to free up VRAM..."
pkill -f "uvicorn" || true
sleep 2

echo "[INFO] Extracting new codebase..."
mkdir -p "$REMOTE_DIR"
tar -xzf "$TARBALL_PATH" -C "$REMOTE_DIR"

cd "$REMOTE_DIR"

# 1. Conditional Virtual Environment creation
if [ ! -d "venv" ]; then
    echo "[INFO] Creating isolated Virtual Environment (Safe Zone)..."
    pip3 install virtualenv -q
    virtualenv venv --system-site-packages -q
    
    echo "[INFO] Upgrading core tools and patching setuptools..."
    ./venv/bin/pip install --upgrade pip wheel "setuptools<82" -q
else
    echo "[INFO] Virtual Environment already exists. Skipping creation."
fi

# 2. Dependency installation
echo "[INFO] Installing Python dependencies..."
./venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121 > install.log 2>&1

export CUDA_VISIBLE_DEVICES=0

# 3. Background server startup
echo "[INFO] Starting new FastAPI instance in the background..."
nohup ./venv/bin/python -m uvicorn src.open_enricher.remote.server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

echo "[SUCCESS] Deployment successful! API is now listening."