#!/bin/bash
#SBATCH --job-name=smoke-insightface-mid30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/smoke-insightface-mid30-%j.out
#SBATCH --error=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/smoke-insightface-mid30-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --export=ALL

# 30s middle segment of How_I_Met_Your_Mother.S01E02.mkv.
# Frames are sampled with the character-graph detect_tracks step, not a denser rate.
# Submit with: sbatch --export=ALL test_scripts/smoke_insightface_mid30.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-/SLURM_shared/st_mdp26/open-enricher-wt-insightface-mid-7023}"
export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:$PATH"
export OPEN_ENRICHER_DATASET_ROOT=/SLURM_shared/st_mdp26/open-enricher/dataset

FFMPEG_ROOT=/SLURM_shared/st_mdp26/ffmpeg-libs
SITE=/SLURM_shared/st_mdp26/open-enricher/.venv/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH="$SITE/cudnn/lib:$SITE/cublas/lib:$SITE/cuda_runtime/lib:$SITE/cufft/lib:$SITE/curand/lib:$SITE/cusolver/lib:$SITE/cusparse/lib:$SITE/cuda_nvrtc/lib:$SITE/nvjitlink/lib:$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

srun --export=ALL --unbuffered \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python \
  test_scripts/insightface_mid30.py
