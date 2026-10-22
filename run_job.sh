#!/bin/bash
#SBATCH --job-name=open-enricher
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/open-enricher-%j.out
#SBATCH --error=/SLURM_shared/st_mdp26/open-enricher/jobs_logs/open-enricher-%j.err
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --gres=gpu:1
cd /SLURM_shared/st_mdp26/open-enricher
srun --unbuffered --cpu-bind=verbose \
  /SLURM_shared/st_mdp26/open-enricher/.venv/bin/python -m open_enricher \
  --only 01-interval-llm \
  --video Suits.S01E02.mkv