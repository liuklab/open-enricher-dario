#!/bin/bash
#SBATCH --job-name=open-enricher-character-lookup
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=jobs_logs/open-enricher-character-lookup-%j.out
#SBATCH --error=jobs_logs/open-enricher-character-lookup-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
SHARED=/SLURM_shared/st_mdp26/open-enricher
VIDEO="${1:?Pass the source clip}"
TRANSCRIPT="${2:?Pass its diarization archive}"
export OPEN_ENRICHER_DATASET_ROOT="$SHARED/dataset"
export OPEN_ENRICHER_PIPELINE_SPEC="$ROOT/src/open_enricher/experiments/variants/12-combined.json"
export PATH="/SLURM_shared/st_mdp26/ffmpeg-libs/bin:$PATH"
export PYTHONPATH="$ROOT/src" CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
srun --unbuffered --export=ALL "$SHARED/.venv/bin/python" "$ROOT/test_scripts/character_lookup.py" \
    "$VIDEO" "$TRANSCRIPT" "$ROOT/var/character-lookup-$SLURM_JOB_ID.json"
