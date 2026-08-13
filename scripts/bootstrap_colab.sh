#!/usr/bin/env bash
# Provision a Colab Pro GPU session, mount Drive, install the library, run inference.
# Copy this repo to Drive first (default: MyDrive/open_enricher/code) and put videos under
# MyDrive/open_enricher/data. Install the CLI locally: uv tool install google-colab-cli
#
# Usage from the repository root:
#   export HF_TOKEN=hf_...    # optional, for Hugging Face
#   ./scripts/bootstrap_colab.sh
#   ./scripts/bootstrap_colab.sh embeddings
#
# Extra flags after the mode are forwarded to colab exec (for example --env OPEN_ENRICHER_VIDEO=clip.mp4).

set -euo pipefail

SESSION="${COLAB_SESSION:-enricher}"
GPU="${COLAB_GPU:-A100}"
MODE="${1:-metadata}"
if [[ "${#}" -gt 0 ]]; then
  shift
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMEOUT="${COLAB_TIMEOUT:-7200}"
INSTALL_TIMEOUT="${COLAB_INSTALL_TIMEOUT:-600}"

if [[ "${MODE}" != "metadata" && "${MODE}" != "embeddings" ]]; then
  echo "Usage: $0 [metadata|embeddings] [extra colab exec args...]" >&2
  exit 1
fi

echo "Starting session ${SESSION} on GPU ${GPU}"
colab new -s "${SESSION}" --gpu "${GPU}" --high-mem

echo "Mounting Google Drive (browser OAuth may open once)"
colab drivemount -s "${SESSION}"

echo "Installing Python dependencies on the VM"
colab install -s "${SESSION}" -r "${ROOT}/requirements.txt"

echo "Editable-installing open-enricher from Drive"
colab exec -s "${SESSION}" --timeout "${INSTALL_TIMEOUT}" -f "${ROOT}/scripts/install_package.py"

RUNNER="${ROOT}/scripts/run_metadata.py"
if [[ "${MODE}" == "embeddings" ]]; then
  RUNNER="${ROOT}/scripts/run_embeddings.py"
fi

ENV_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_ARGS+=(--env "HF_TOKEN=${HF_TOKEN}")
fi

echo "Running ${RUNNER}"
colab exec -s "${SESSION}" --timeout "${TIMEOUT}" "${ENV_ARGS[@]}" "$@" -f "${RUNNER}"

echo "Done. Session ${SESSION} is still running. Stop it with: colab stop -s ${SESSION}"
