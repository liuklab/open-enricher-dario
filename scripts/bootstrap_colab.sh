#!/usr/bin/env bash
# Provision a Colab Pro GPU session, mount Drive (videos only), sync code via git, run inference.
# Push your commits first. Videos live under MyDrive/open_enricher/data.
# Install the CLI locally: uv tool install google-colab-cli
#
# Usage from the repository root:
#   export HF_TOKEN=hf_...         # optional, Hugging Face
#   export GITHUB_TOKEN=ghp_...    # optional, private git remotes
#   ./scripts/bootstrap_colab.sh
#   ./scripts/bootstrap_colab.sh embeddings
#
# Extra flags after the mode are forwarded to the inference colab exec
# (for example --env OPEN_ENRICHER_VIDEO=clip.mp4).

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

GIT_URL="${OPEN_ENRICHER_GIT_URL:-$(git -C "${ROOT}" remote get-url origin 2>/dev/null || true)}"
GIT_REF="${OPEN_ENRICHER_GIT_REF:-$(git -C "${ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)}"
if [[ "${GIT_REF}" == "HEAD" ]]; then
  GIT_REF="$(git -C "${ROOT}" rev-parse HEAD)"
fi

echo "Starting session ${SESSION} on GPU ${GPU}"
colab new -s "${SESSION}" --gpu "${GPU}" --high-mem

echo "Mounting Google Drive for videos (browser OAuth may open once)"
colab drivemount -s "${SESSION}"

echo "Installing Python dependencies on the VM"
colab install -s "${SESSION}" -r "${ROOT}/requirements.txt"

INSTALL_ENV=()
if [[ -n "${GIT_URL}" ]]; then
  INSTALL_ENV+=(--env "OPEN_ENRICHER_GIT_URL=${GIT_URL}")
fi
if [[ -n "${GIT_REF}" ]]; then
  INSTALL_ENV+=(--env "OPEN_ENRICHER_GIT_REF=${GIT_REF}")
fi
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  INSTALL_ENV+=(--env "GITHUB_TOKEN=${GITHUB_TOKEN}")
elif [[ -n "${GIT_TOKEN:-}" ]]; then
  INSTALL_ENV+=(--env "GIT_TOKEN=${GIT_TOKEN}")
fi

echo "Syncing library from git${GIT_URL:+ (${GIT_URL} @ ${GIT_REF})}. Push local commits first."
colab exec -s "${SESSION}" --timeout "${INSTALL_TIMEOUT}" "${INSTALL_ENV[@]}" -f "${ROOT}/scripts/install_package.py"

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
