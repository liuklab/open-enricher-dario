#!/bin/bash
# Login-node check: expected external checkpoints exist; non-GPU imports/versions.
# No #SBATCH. No CUDA. GPU probes stay in next_run/.
# Re-triggered after watcher reorder (commands before sbatch).

set -euo pipefail

cd /SLURM_shared/st_mdp26/open-enricher
export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:$PATH"

ROOT="dataset/models_cache/external"
missing=0

check_path() {
  local label="$1"
  local path="$2"
  local kind="$3"
  if [[ "$kind" == "dir" ]]; then
    if [[ -d "$path" ]]; then
      echo "OK  dir  $label  $path"
    else
      echo "MISS dir  $label  $path"
      missing=1
    fi
  else
    if [[ -f "$path" ]]; then
      echo "OK  file $label  $path"
    else
      echo "MISS file $label  $path"
      missing=1
    fi
  fi
}

echo "=== checkpoint paths under ${ROOT} ==="
check_path "bassl" "${ROOT}/bassl" "dir"
check_path "bassl/pytorch_model.pt" "${ROOT}/bassl/pytorch_model.pt" "file"
check_path "bassl/configuration.json" "${ROOT}/bassl/configuration.json" "file"
check_path "insightface/buffalo_l" "${ROOT}/insightface/models/buffalo_l" "dir"
check_path "talknet" "${ROOT}/talknet/pretrain_TalkSet.model" "file"
check_path "lavis/eva_vit_g.pth" "${ROOT}/lavis/eva_vit_g.pth" "file"
check_path "lavis/blip2_pretrained_flant5xxl.pth" "${ROOT}/lavis/blip2_pretrained_flant5xxl.pth" "file"

echo
echo "=== non-GPU imports / versions ==="
python - <<'PY'
import importlib
import sys

print(f"python {sys.version.split()[0]}")

def show(name: str, attr: str | None = "__version__") -> None:
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report setup gaps
        print(f"FAIL import {name}: {exc}")
        return
    version = getattr(mod, attr, None) if attr else None
    if version is None and attr:
        version = "(no version attr)"
    print(f"OK   import {name}" + (f"  {version}" if attr else ""))

show("torch")
show("cv2")
show("insightface")
show("modelscope")
show("decord")
show("onnxruntime")
try:
    import torch
    print(f"torch.cuda.is_available (login node, expect False/True): {torch.cuda.is_available()}")
except Exception as exc:  # noqa: BLE001
    print(f"torch.cuda check failed: {exc}")
PY

echo
if [[ "$missing" -ne 0 ]]; then
  echo "checkpoint check FAILED: one or more paths missing"
  exit 1
fi
echo "checkpoint check OK"
exit 0
