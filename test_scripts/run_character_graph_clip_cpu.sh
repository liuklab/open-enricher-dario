#!/bin/bash
#SBATCH --job-name=open-enricher-graph-clip-cpu
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=3500M
#SBATCH --time=00:30:00
#SBATCH --output=jobs_logs/open-enricher-graph-cpu-%j.out
#SBATCH --error=jobs_logs/open-enricher-graph-cpu-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
SHARED=/SLURM_shared/st_mdp26/open-enricher
# Replay full-film diarization after translating its timestamps to the clip.
VIDEO="${1:?Pass the dataset video filename}"
START="${2:?Pass the clip start in seconds}"
DURATION="${3:-30}"
if [ "$VIDEO" != "${VIDEO##*/}" ]; then
  printf 'Pass a video filename without directories.\n' >&2
  exit 2
fi
DATA="$ROOT/var/graph-clip-$SLURM_JOB_ID"
mkdir -p "$DATA/video_inputs" "$DATA/outputs/audio_results"
ln -s "$SHARED/dataset/models_cache" "$DATA/models_cache"
export OPEN_ENRICHER_DATASET_ROOT="$DATA"
export OPEN_ENRICHER_VIDEO_INPUTS="$DATA/video_inputs"
export OPEN_ENRICHER_PIPELINE_SPEC="$ROOT/src/open_enricher/experiments/variants/12-combined.json"
export PATH="/SLURM_shared/st_mdp26/ffmpeg-libs/bin:$PATH"
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export GRAPH_SOURCE_DATASET="$SHARED/dataset"
srun --unbuffered --export=ALL "$SHARED/.venv/bin/python" - "$VIDEO" "$START" "$DURATION" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

from open_enricher.stages.optional import CharacterGraphStage

video, start_arg, duration_arg = sys.argv[1:]
start, duration = float(start_arg), float(duration_arg)
if start < 0 or duration <= 0:
    raise ValueError("The clip must have a nonnegative start and a positive duration")
source = Path(os.environ["GRAPH_SOURCE_DATASET"])
dataset = Path(os.environ["OPEN_ENRICHER_DATASET_ROOT"])
subprocess.run(
    [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "4",
        "-ss", start_arg, "-t", duration_arg, "-i", str(source / "video_inputs" / video),
        "-c:v", "libx264", "-threads", "4", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-c:a", "aac", str(dataset / "video_inputs" / video),
    ],
    check=True,
)
audio_filename = f"{video}_audio.json"
lines = json.loads((source / "outputs" / "audio_results" / audio_filename).read_text())
clip_lines = [
    {
        **line,
        "start_timestamp": round(max(0.0, line["start_timestamp"] - start), 3),
        "end_timestamp": round(min(duration, line["end_timestamp"] - start), 3),
    }
    for line in lines
    if line["start_timestamp"] < start + duration and line["end_timestamp"] > start
]
if not clip_lines:
    raise ValueError("The cached transcript has no dialogue in this clip")
(dataset / "outputs" / "audio_results" / audio_filename).write_text(
    json.dumps(clip_lines), encoding="utf-8"
)
result = CharacterGraphStage().run(video)
print(json.dumps(result), flush=True)
if result["status"] != "success":
    raise SystemExit(1)
PY
