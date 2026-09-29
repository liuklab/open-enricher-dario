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

# 2. Dependency check and strict isolation upgrade
echo -n "[INFO] Verifying and isolating Python dependencies... "
> install.log
(
    ./venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
    # FORZIAMO uvicorn e le librerie LLM a installarsi dentro il VENV per evitare conflitti con Colab
    ./venv/bin/pip install --upgrade ctranslate2 faster-whisper uvicorn fastapi "transformers[torch]" accelerate bitsandbytes lm-format-enforcer pydantic
) >> install.log 2>&1 &
PID=$!

spin='-\|/'
i=0
while kill -0 $PID 2>/dev/null; do
  i=$(( (i+1) %4 ))
  printf "\b${spin:$i:1}"
  sleep 0.1
done
printf "\bDone!\n"

# 3. Environment Variables & Cache Configuration
echo "[INFO] Injecting correct Database Cache paths for Hugging Face..."
export HF_HOME="/content/drive/MyDrive/Open-Enricher-Database/Models-Cache/huggingface"
export TORCH_HOME="/content/drive/MyDrive/Open-Enricher-Database/Models-Cache/torch"
export HF_HUB_DISABLE_SYMLINKS=1
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

# 4. Background server startup
echo "[INFO] Starting new FastAPI instance in the background..."

# DIAGNOSTIC CHECK: Print PyTorch GPU status directly to the terminal
./venv/bin/python -c "import torch; print(f'\n[DIAGNOSTIC] PyTorch GPU Available: {torch.cuda.is_available()} | Device count: {torch.cuda.device_count()}\n')"

# Start the server and redirect output to server.log for Colab to stream
nohup ./venv/bin/python -m uvicorn src.open_enricher.remote.server:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &

echo "[SUCCESS] Deployment successful! API is now listening."