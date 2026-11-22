#!/bin/bash
#SBATCH --job-name=open-enricher-memory-ablation
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=jobs_logs/open-enricher-memory-ablation-%j.out
#SBATCH --error=jobs_logs/open-enricher-memory-ablation-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
SOURCE="${1:?Pass the source dataset directory}"
VIDEO="${2:?Pass the video filename}"
MODE="${3:?Choose current, recent, retrieved, or grounded}"
SHARED=/SLURM_shared/st_mdp26/open-enricher
DATA="$ROOT/var/memory-ablation-$SLURM_JOB_ID"
mkdir -p "$DATA/outputs"
ln -s "$SHARED/dataset/video_inputs" "$DATA/video_inputs"
ln -s "$SHARED/dataset/models_cache" "$DATA/models_cache"
for stage in audio_results dialogue_results memory_results; do
    ln -s "$SOURCE/outputs/$stage" "$DATA/outputs/$stage"
done
mkdir -p "$DATA/outputs/graph_results"
cp "$SOURCE/outputs/graph_results/${VIDEO}_graph.json" "$DATA/outputs/graph_results/"
export OPEN_ENRICHER_DATASET_ROOT="$DATA" OPEN_ENRICHER_VIDEO_INPUTS="$DATA/video_inputs"
export PYTHONPATH="$ROOT/src" HF_HOME="$DATA/models_cache/huggingface"
export OPEN_ENRICHER_PIPELINE_SPEC="$ROOT/src/open_enricher/experiments/variants/12-combined.json"
FFMPEG=/SLURM_shared/st_mdp26/ffmpeg-libs
export PATH="$FFMPEG/bin:$PATH" LD_LIBRARY_PATH="$FFMPEG/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
srun --unbuffered --export=ALL "$SHARED/.venv/bin/python" "$ROOT/test_scripts/visual_memory_ablation.py" \
    "$VIDEO" "$MODE" "$DATA" "$SHARED"
