#!/bin/bash
#SBATCH --job-name=smoke-talknet-forward
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/SLURM_shared/st_mdp26/open-enricher-wt-talknet-0ef6/jobs_logs/smoke-talknet-forward-%j.out
#SBATCH --error=/SLURM_shared/st_mdp26/open-enricher-wt-talknet-0ef6/jobs_logs/smoke-talknet-forward-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes

# Short TalkNet probe. Eight seconds of HIMYM from 00:00:20, at most four
# one-second windows. The checkout is the directory sbatch was run from.

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
export OPEN_ENRICHER_DATASET_ROOT=/SLURM_shared/st_mdp26/open-enricher/dataset
export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:/SLURM_shared/st_mdp26/ffmpeg-libs/bin:${PATH}"

FFMPEG_ROOT=/SLURM_shared/st_mdp26/ffmpeg-libs
SITE=/SLURM_shared/st_mdp26/open-enricher/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$SITE/cudnn/lib:$SITE/cublas/lib:$SITE/cuda_runtime/lib:$SITE/cufft/lib:$SITE/curand/lib:$SITE/cusolver/lib:$SITE/cusparse/lib:$SITE/cuda_nvrtc/lib:$SITE/nvjitlink/lib:$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$ROOT/jobs_logs"

srun --unbuffered \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python \
  test_scripts/talknet_speakers.py \
  --start 20 \
  --seconds 8 \
  --sample-fps 5 \
  --max-frames 40 \
  --max-tracks 4
