#!/bin/bash
#SBATCH --job-name=open-enricher-graph-free-control
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=jobs_logs/open-enricher-graph-free-control-%j.out
#SBATCH --error=jobs_logs/open-enricher-graph-free-control-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1

# Replay one encoded clip with and without character graph construction.
# Review the profiles against evidence from this span of the source episode.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root with sbatch}"
cd "$ROOT"
mkdir -p "$ROOT/jobs_logs"

export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:/SLURM_shared/st_mdp26/ffmpeg-libs/bin:${PATH}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

SHARED=/SLURM_shared/st_mdp26/open-enricher
SHARED_DATA="$SHARED/dataset"
DATA="$ROOT/var/graph-free-control-${SLURM_JOB_ID:?}"
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

VIDEO="${1:?Pass a video filename}"
START="${2:?Pass a start time in seconds}"
DURATION="${3:?Pass a duration in seconds}"
if [ "$#" -gt 3 ] || [ "$VIDEO" != "${VIDEO##*/}" ]; then
  printf 'Usage: sbatch %s filename start duration\n' "$0" >&2
  exit 2
fi
SRC="$SHARED_DATA/video_inputs/$VIDEO"
CLIP_DIR="$DATA/video_inputs"
mkdir -p "$CLIP_DIR"
# Re-encode so the container duration is the span. Stream copy can leave
# a duration that does not match the pictures BaSSL has to cover.
ffmpeg -y -hide_banner -loglevel error \
  -ss "$START" -t "$DURATION" -i "$SRC" \
  -c:v libx264 -preset veryfast -pix_fmt yuv420p \
  -c:a aac \
  "$CLIP_DIR/$VIDEO"

export OPEN_ENRICHER_DATASET_ROOT="$DATA"
export OPEN_ENRICHER_VIDEO_INPUTS="$CLIP_DIR"

FFMPEG_ROOT=/SLURM_shared/st_mdp26/ffmpeg-libs
SITE=/SLURM_shared/st_mdp26/open-enricher/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$SITE/cudnn/lib:$SITE/cublas/lib:$SITE/cuda_runtime/lib:$SITE/cufft/lib:$SITE/curand/lib:$SITE/cusolver/lib:$SITE/cusparse/lib:$SITE/cuda_nvrtc/lib:$SITE/nvjitlink/lib:$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# --export=ALL keeps the dataset root and the clip directory. A narrower
# --export list would drop them, and Python would look under a missing dataset.
srun --unbuffered --chdir="$ROOT" --export=ALL \
  /usr/bin/env \
  OPEN_ENRICHER_DATASET_ROOT="$DATA" \
  OPEN_ENRICHER_VIDEO_INPUTS="$CLIP_DIR" \
  PYTHONPATH="${ROOT}/src" \
  "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python -m open_enricher \
  --only 12-combined 12-combined-no-graph \
  --video "$VIDEO"
