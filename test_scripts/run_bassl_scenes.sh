#!/bin/bash
#SBATCH --job-name=bassl-scenes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/bassl-scenes-%j.out
#SBATCH --error=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/bassl-scenes-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1

# Scene boundaries only, for one video. Override with:
#   sbatch test_scripts/run_bassl_scenes.sh Silicon_Valley.S06E01.mkv

cd /SLURM_shared/st_mdp26/open-enricher
export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:$PATH"

FFMPEG_ROOT=/SLURM_shared/st_mdp26/ffmpeg-libs
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VIDEO="${1:-How_I_Met_Your_Mother.S01E02.mkv}"

srun --unbuffered \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python \
  test_scripts/bassl_scenes.py \
  --video "dataset/video_inputs/${VIDEO}"
