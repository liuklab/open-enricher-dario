#!/usr/bin/env bash
# Provision a Colab Pro GPU session, mount Drive (videos only), sync code via git, run inference.
# Secrets: .env. Project settings: config.toml.
# Push your commits first. Install the CLI locally: uv tool install google-colab-cli
#
# Usage from the repository root:
#   ./scripts/bootstrap_colab.sh
#   ./scripts/bootstrap_colab.sh embeddings

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT}/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
else
  echo "No .env found at ${ENV_FILE}. Copy .env.example to .env and fill in tokens." >&2
fi

while IFS= read -r line || [[ -n "${line}" ]]; do
  [[ -z "${line}" || "${line}" == \#* ]] && continue
  key="${line%%=*}"
  val="${line#*=}"
  if [[ -z "${!key:-}" ]]; then
    export "${key}=${val}"
  fi
done < <(python3 "${ROOT}/scripts/print_config_env.py")

SESSION="${COLAB_SESSION:-enricher}"
GPU="${COLAB_GPU:-A100}"
AUTH="${COLAB_AUTH:-adc}"
MODE="${1:-metadata}"
if [[ "${#}" -gt 0 ]]; then
  shift
fi

TIMEOUT="${COLAB_TIMEOUT:-7200}"
INSTALL_TIMEOUT="${COLAB_INSTALL_TIMEOUT:-600}"

if [[ "${MODE}" != "metadata" && "${MODE}" != "embeddings" ]]; then
  echo "Usage: $0 [metadata|embeddings] [extra colab exec args...]" >&2
  exit 1
fi

if [[ -z "${OPEN_ENRICHER_GIT_URL:-}" ]]; then
  OPEN_ENRICHER_GIT_URL="$(git -C "${ROOT}" remote get-url origin 2>/dev/null || true)"
fi
if [[ -z "${OPEN_ENRICHER_GIT_REF:-}" ]]; then
  OPEN_ENRICHER_GIT_REF="$(git -C "${ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ "${OPEN_ENRICHER_GIT_REF}" == "HEAD" ]]; then
    OPEN_ENRICHER_GIT_REF="$(git -C "${ROOT}" rev-parse HEAD)"
  fi
fi

FORWARD_KEYS=(
  HF_TOKEN
  GITHUB_TOKEN
  GIT_TOKEN
  OPEN_ENRICHER_GIT_URL
  OPEN_ENRICHER_GIT_REF
  OPEN_ENRICHER_REPO
  OPEN_ENRICHER_DATA
  OPEN_ENRICHER_OUTPUT
  OPEN_ENRICHER_VIDEO
  OPEN_ENRICHER_MODEL
  OPEN_ENRICHER_NUM_FRAMES
  OPEN_ENRICHER_MAX_NEW_TOKENS
)

append_forwarded_env() {
  local -n dest=$1
  local key
  for key in "${FORWARD_KEYS[@]}"; do
    if [[ -n "${!key:-}" ]]; then
      dest+=(--env "${key}=${!key}")
    fi
  done
}

run_colab() {
  colab --auth="${AUTH}" "$@"
}

echo "Starting session ${SESSION} on GPU ${GPU} (auth=${AUTH})"
run_colab new -s "${SESSION}" --gpu "${GPU}"

echo "Mounting Google Drive for videos (browser OAuth may open once)"
run_colab drivemount -s "${SESSION}"

echo "Installing Python dependencies on the VM"
run_colab install -s "${SESSION}" -r "${ROOT}/requirements.txt"

INSTALL_ENV=()
append_forwarded_env INSTALL_ENV

echo "Syncing library from git${OPEN_ENRICHER_GIT_URL:+ (${OPEN_ENRICHER_GIT_URL} @ ${OPEN_ENRICHER_GIT_REF:-})}. Push local commits first."
run_colab exec -s "${SESSION}" --timeout "${INSTALL_TIMEOUT}" "${INSTALL_ENV[@]}" -f "${ROOT}/scripts/install_package.py"

RUNNER="${ROOT}/scripts/run_metadata.py"
if [[ "${MODE}" == "embeddings" ]]; then
  RUNNER="${ROOT}/scripts/run_embeddings.py"
fi

RUN_ENV=()
append_forwarded_env RUN_ENV

echo "Running ${RUNNER}"
run_colab exec -s "${SESSION}" --timeout "${TIMEOUT}" "${RUN_ENV[@]}" "$@" -f "${RUNNER}"

echo "Done. Session ${SESSION} is still running. Stop it with: colab --auth=${AUTH} stop -s ${SESSION}"
