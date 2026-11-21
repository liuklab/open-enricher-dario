#!/bin/bash
#SBATCH --job-name=smoke-eva-tokens
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=01:00:00
#SBATCH --output=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/smoke-eva-tokens-%j.out
#SBATCH --error=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/smoke-eva-tokens-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1

# EVA-CLIP / Q-Former token smoke for How_I_Met_Your_Mother.S01E02.mkv.
# MAX_FRAMES limits the sampled frames. Unset, the smoke reads the whole episode.
# A key mismatch leaves the visual bank empty.

set -euo pipefail

# Slurm copies this file under /var/spool/slurmd, so BASH_SOURCE is not the repo.
# Submit from the repo root. SLURM_SUBMIT_DIR is that directory.
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/test_scripts/eva_encode.py" ]; then
  ROOT="$SLURM_SUBMIT_DIR"
elif [ -f "$(pwd)/test_scripts/eva_encode.py" ]; then
  ROOT="$(pwd)"
else
  echo "eva smoke: submit from the repo root (test_scripts/eva_encode.py not found)" >&2
  exit 1
fi
cd "$ROOT"
export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:$PATH"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -d "$ROOT/dataset/models_cache" ]; then
  export OPEN_ENRICHER_DATASET_ROOT="/SLURM_shared/st_mdp26/open-enricher/dataset"
fi
DATA="${OPEN_ENRICHER_DATASET_ROOT:-$ROOT/dataset}"

FFMPEG_ROOT=/SLURM_shared/st_mdp26/ffmpeg-libs
SITE=/SLURM_shared/st_mdp26/open-enricher/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$SITE/cudnn/lib:$SITE/cublas/lib:$SITE/cuda_runtime/lib:$SITE/cufft/lib:$SITE/curand/lib:$SITE/cusolver/lib:$SITE/cusparse/lib:$SITE/cuda_nvrtc/lib:$SITE/nvjitlink/lib:$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VIDEO="How_I_Met_Your_Mother.S01E02.mkv"
ARGS=(--video "${DATA}/video_inputs/${VIDEO}" --sample-fps 0.25)
if [ -n "${MAX_FRAMES:-}" ]; then
  ARGS+=(--max-frames "${MAX_FRAMES}")
fi

srun --unbuffered --chdir="$ROOT" \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python \
  "${ROOT}/test_scripts/eva_encode.py" \
  "${ARGS[@]}"
