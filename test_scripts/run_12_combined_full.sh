#!/bin/bash
#SBATCH --job-name=open-enricher-12-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --output=jobs_logs/open-enricher-12-full-%j.out
#SBATCH --error=jobs_logs/open-enricher-12-full-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1

# Run 12-combined on all videos by default, or on the filenames passed
# to this script. Stage files go under dataset-full. Models are shared.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root with sbatch}"
cd "$ROOT"
mkdir -p "$ROOT/jobs_logs"

export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:/SLURM_shared/st_mdp26/ffmpeg-libs/bin:${PATH}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

SHARED=/SLURM_shared/st_mdp26/open-enricher
SHARED_DATA="$SHARED/dataset"
DATA="$ROOT/dataset-full"
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

export OPEN_ENRICHER_DATASET_ROOT="$DATA"
export OPEN_ENRICHER_VIDEO_INPUTS="$SHARED_DATA/video_inputs"
VIDEO_ARGS=()
if [ "$#" -gt 0 ]; then
  VIDEO_ARGS=(--video "$@")
fi
for VIDEO in "$@"; do
  test -f "${OPEN_ENRICHER_VIDEO_INPUTS}/${VIDEO}"
done

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
  --only 12-combined \
  "${VIDEO_ARGS[@]}"
