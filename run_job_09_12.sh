#!/bin/bash
#SBATCH --job-name=open-enricher-09-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=20:00:00
#SBATCH --output=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/open-enricher-%j.out
#SBATCH --error=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/open-enricher-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1

cd /SLURM_shared/st_mdp26/open-enricher
export PATH="/SLURM_shared/st_mdp26/open-enricher/.venv/bin:$PATH"

FFMPEG_ROOT=/SLURM_shared/st_mdp26/ffmpeg-libs
export LD_LIBRARY_PATH="$FFMPEG_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

srun --unbuffered --cpu-bind=verbose \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python -m open_enricher \
  --only 09-bassl-adaptive 09-bassl-uniform 10-character-graph 11-visual-memory 12-combined \
  --video How_I_Met_Your_Mother.S01E02.mkv Silicon_Valley.S06E01.mkv The_Kingdom.S01E02.mkv
