#!/bin/bash
#SBATCH --job-name=open-enricher-memory-stage
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=jobs_logs/open-enricher-memory-stage-%j.out
#SBATCH --error=jobs_logs/open-enricher-memory-stage-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
SOURCE="${1:?Pass the source dataset directory}"
VIDEO="${2:?Pass the video filename}"
SHARED=/SLURM_shared/st_mdp26/open-enricher
DATA="$ROOT/var/memory-stage-$SLURM_JOB_ID"
mkdir -p "$DATA/outputs"
ln -s "$SHARED/dataset/video_inputs" "$DATA/video_inputs"
ln -s "$SHARED/dataset/models_cache" "$DATA/models_cache"
for stage in timeline_results bassl_results; do
    ln -s "$SOURCE/outputs/$stage" "$DATA/outputs/$stage"
done
export OPEN_ENRICHER_DATASET_ROOT="$DATA" OPEN_ENRICHER_VIDEO_INPUTS="$DATA/video_inputs"
export PYTHONPATH="$ROOT/src" HF_HOME="$DATA/models_cache/huggingface"
export OPEN_ENRICHER_PIPELINE_SPEC="$ROOT/src/open_enricher/experiments/variants/12-combined.json"
FFMPEG=/SLURM_shared/st_mdp26/ffmpeg-libs
export PATH="$FFMPEG/bin:$PATH" LD_LIBRARY_PATH="$FFMPEG/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
srun --unbuffered --export=ALL "$SHARED/.venv/bin/python" - "$VIDEO" "$SHARED" <<'PY'
import json
import os
import sys
from pathlib import Path

from open_enricher import storage
from open_enricher.runner import bootstrap_runtime
from open_enricher.stages.optional import VisualMemoryStage

video, shared = sys.argv[1:]
bootstrap_runtime(env_file=Path(shared) / ".env", tee_log=False)
timeline = storage.get_output_dir("bassl") / f"{video}_bassl.json"
os.environ["OPEN_ENRICHER_ATTEMPT_ID"] = json.loads(timeline.read_text())["attempt_id"]
result = VisualMemoryStage().run(video)
print(json.dumps(result), flush=True)
if result["status"] != "success":
    raise SystemExit(1)
memory = json.loads((storage.get_output_dir("visual_memory") / result["output_file"]).read_text())
entries = {entry["frame_id"]: entry for entry in memory["entries"]}
assert entries and len(entries) <= 64
for query in memory["retrievals"]:
    for frame in query["frames"]:
        assert entries[frame["frame_id"]]["end"] < query["start"]
print(f"Verified {len(entries)} memory entries across {len(memory['retrievals'])} scenes.", flush=True)
PY
