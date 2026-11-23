#!/bin/bash
#SBATCH --job-name=open-enricher-bassl-benchmark
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=jobs_logs/open-enricher-bassl-benchmark-%j.out
#SBATCH --error=jobs_logs/open-enricher-bassl-benchmark-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
VIDEO="$(realpath "${1:?Pass the benchmark video}")"
REFERENCE="$(realpath "${2:?Pass its reference boundary timestamps}")"
SHARED=/SLURM_shared/st_mdp26/open-enricher
DATA="$ROOT/var/bassl-benchmark-${SLURM_JOB_ID:?}"
mkdir -p "$DATA/video_inputs"
ln -s "$VIDEO" "$DATA/video_inputs/"
ln -s "$SHARED/dataset/models_cache" "$DATA/models_cache"
export OPEN_ENRICHER_DATASET_ROOT="$DATA" OPEN_ENRICHER_VIDEO_INPUTS="$DATA/video_inputs"
export OPEN_ENRICHER_PIPELINE_SPEC="$ROOT/src/open_enricher/experiments/variants/12-combined.json"
export PYTHONPATH="$ROOT/src"
export PATH="/SLURM_shared/st_mdp26/ffmpeg-libs/bin:$PATH"
FFMPEG=/SLURM_shared/st_mdp26/ffmpeg-libs
export LD_LIBRARY_PATH="$FFMPEG/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

srun --unbuffered "$SHARED/.venv/bin/python" - "${VIDEO##*/}" "$SHARED" <<'PY'
import json
import sys
from pathlib import Path

from open_enricher.runner import bootstrap_runtime
from open_enricher.stages.optional import BasslStage

video, shared = sys.argv[1:]
bootstrap_runtime(env_file=Path(shared) / ".env", tee_log=False)
result = BasslStage().run(video)
print(json.dumps(result), flush=True)
if result["status"] not in {"success", "degraded"}:
    raise SystemExit(1)
PY
"$SHARED/.venv/bin/python" "$ROOT/test_scripts/score_scene_boundaries.py" \
    "$DATA/outputs/bassl_results/${VIDEO##*/}_bassl.json" "$REFERENCE" --tolerance 1 \
    > "$DATA/boundary-score.json"
