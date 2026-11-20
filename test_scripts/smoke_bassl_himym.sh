#!/bin/bash
#SBATCH --job-name=smoke-bassl-himym
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/smoke-bassl-himym-%j.out
#SBATCH --error=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/smoke-bassl-himym-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1

# BaSSL scene-boundary smoke for How_I_Met_Your_Mother.S01E02.mkv only.

set -euo pipefail

cd /SLURM_shared/st_mdp26/open-enricher
export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:$PATH"

FFMPEG_ROOT=/SLURM_shared/st_mdp26/ffmpeg-libs
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VIDEO="How_I_Met_Your_Mother.S01E02.mkv"

srun --unbuffered \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python \
  test_scripts/bassl_scenes.py \
  --video "dataset/video_inputs/${VIDEO}"
