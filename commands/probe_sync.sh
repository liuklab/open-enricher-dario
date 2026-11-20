#!/bin/bash
# Login-node probe: verify command → jobs_logs → GitHub sync.
# No #SBATCH. Fast (<30s). No GPU.

set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "=== probe_sync ==="
echo "date:     $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname: $(hostname)"
echo "HEAD:     $(git rev-parse HEAD)"

echo
echo "=== squeue -u \$USER (short) ==="
if command -v squeue >/dev/null 2>&1; then
  squeue -u "${USER:-$(whoami)}" -o "%.18i %.9P %.30j %.8T %.10M" 2>/dev/null || true
else
  echo "(squeue not available)"
fi

echo
echo "=== jobs_logs/command-*.log basenames ==="
shopt -s nullglob
logs=(jobs_logs/command-*.log)
if ((${#logs[@]})); then
  for f in "${logs[@]}"; do
    basename "$f"
  done | sort
else
  echo "(none)"
fi

echo
echo "probe_sync OK"
exit 0
