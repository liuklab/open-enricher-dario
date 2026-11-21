#!/bin/bash
#SBATCH --job-name=open-enricher-12-bassl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/SLURM_shared/st_mdp26/open-enricher-wt-combined-aa84/jobs_logs/open-enricher-12-bassl-%j.out
#SBATCH --error=/SLURM_shared/st_mdp26/open-enricher-wt-combined-aa84/jobs_logs/open-enricher-12-bassl-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1

# Only the BaSSL stage, on the full HIMYM file, with the 12-combined spec.
# A fresh dataset directory so the recorded fallback is not reused and the
# long 122809 run keeps its own outputs. Weights stay on the shared cache.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-/SLURM_shared/st_mdp26/open-enricher-wt-combined-aa84}"
cd "$ROOT"
mkdir -p "$ROOT/jobs_logs"

export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:/SLURM_shared/st_mdp26/ffmpeg-libs/bin:${PATH}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

SHARED=/SLURM_shared/st_mdp26/open-enricher
SHARED_DATA="$SHARED/dataset"
DATA="$ROOT/dataset-bassl-check"
mkdir -p "$DATA"
if [ ! -e "$DATA/models_cache" ]; then
  ln -s "$SHARED_DATA/models_cache" "$DATA/models_cache"
fi
if [ ! -e "$ROOT/synopsis" ] && [ -d "$SHARED/synopsis" ]; then
  ln -s "$SHARED/synopsis" "$ROOT/synopsis"
fi
if [ ! -e "$ROOT/.env" ] && [ -f "$SHARED/.env" ]; then
  ln -s "$SHARED/.env" "$ROOT/.env"
fi

VIDEO="How_I_Met_Your_Mother.S01E02.mkv"
export OPEN_ENRICHER_DATASET_ROOT="$DATA"
export OPEN_ENRICHER_VIDEO_INPUTS="$SHARED_DATA/video_inputs"
test -f "${OPEN_ENRICHER_VIDEO_INPUTS}/${VIDEO}"

FFMPEG_ROOT=/SLURM_shared/st_mdp26/ffmpeg-libs
SITE=/SLURM_shared/st_mdp26/open-enricher/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$SITE/cudnn/lib:$SITE/cublas/lib:$SITE/cuda_runtime/lib:$SITE/cufft/lib:$SITE/curand/lib:$SITE/cusolver/lib:$SITE/cusparse/lib:$SITE/cuda_nvrtc/lib:$SITE/nvjitlink/lib:$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

srun --unbuffered --chdir="$ROOT" --export=ALL \
  /usr/bin/env \
  OPEN_ENRICHER_DATASET_ROOT="$DATA" \
  OPEN_ENRICHER_VIDEO_INPUTS="${OPEN_ENRICHER_VIDEO_INPUTS}" \
  PYTHONPATH="${ROOT}/src" \
  "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python -m open_enricher \
  --only 12-combined-bassl \
  --stages bassl \
  --video "$VIDEO"
