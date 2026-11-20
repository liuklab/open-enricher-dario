#!/bin/bash
#SBATCH --job-name=smoke-eva-himym
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/smoke-eva-himym-%j.out
#SBATCH --error=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/smoke-eva-himym-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1

# EVA-CLIP / Q-Former encode smoke for How_I_Met_Your_Mother.S01E02.mkv only.
# Unbound forward is reported clearly; empty frames are not treated as success.

set -euo pipefail

cd /SLURM_shared/st_mdp26/open-enricher
export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:$PATH"

FFMPEG_ROOT=/SLURM_shared/st_mdp26/ffmpeg-libs
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VIDEO="How_I_Met_Your_Mother.S01E02.mkv"

srun --unbuffered \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python \
  test_scripts/eva_encode.py \
  --video "dataset/video_inputs/${VIDEO}" \
  --sample-fps 0.25
