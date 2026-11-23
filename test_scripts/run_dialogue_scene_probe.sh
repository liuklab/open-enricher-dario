#!/bin/bash
#SBATCH --job-name=open-enricher-dialogue-scene
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=jobs_logs/open-enricher-dialogue-scene-%j.out
#SBATCH --error=jobs_logs/open-enricher-dialogue-scene-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
SOURCE="${1:?Pass the clip dataset directory}"
VIDEO="${2:?Pass the video filename}"
MODE="${3:-scene}"
SHARED=/SLURM_shared/st_mdp26/open-enricher
DATA="$ROOT/var/dialogue-scene-$SLURM_JOB_ID"
mkdir -p "$DATA/outputs"
ln -s "$SOURCE/video_inputs" "$DATA/video_inputs"
ln -s "$SHARED/dataset/models_cache" "$DATA/models_cache"
for stage in audio_results graph_results memory_results; do
    ln -s "$SOURCE/outputs/$stage" "$DATA/outputs/$stage"
done
export PYTHONPATH="$ROOT/src" HF_HOME="$DATA/models_cache/huggingface"
export OPEN_ENRICHER_DATASET_ROOT="$DATA" OPEN_ENRICHER_VIDEO_INPUTS="$DATA/video_inputs"
export OPEN_ENRICHER_PIPELINE_SPEC="$ROOT/src/open_enricher/experiments/variants/12-combined.json"
srun --unbuffered --export=ALL "$SHARED/.venv/bin/python" - "$SOURCE" "$VIDEO" "$DATA" "$SHARED" "$MODE" <<'PY'
import json
import os
import sys
from pathlib import Path

from open_enricher import storage
from open_enricher.runner import bootstrap_runtime
from open_enricher.stages.dialogue import DialogueStage
from open_enricher.stages.llm import LLMStage
from open_enricher.timeline import Interval, assign_utterances
from open_enricher.timings import TimingRecorder

source_arg, video, data_arg, shared_arg, mode = sys.argv[1:]
source, data, shared = map(Path, (source_arg, data_arg, shared_arg))
bootstrap_runtime(env_file=shared / ".env", tee_log=False)

def read(stage, suffix):
    return json.loads((source / "outputs" / stage / f"{video}_{suffix}.json").read_text())

dialogue = read("dialogue_results", "dialogue")
vlm = read("vlms_results", "vlm")
os.environ["OPEN_ENRICHER_ATTEMPT_ID"] = dialogue["attempt_id"]
if mode == "without-plot":
    dialogue["plot_summary"] = ""
elif mode == "scene":
    audio = read("audio_results", "audio")
    intervals = [
        Interval(row["scene_id"], row["start_seconds"], row["end_seconds"], "scene", "replay")
        for row in dialogue["scenes"]
    ]
    timing = TimingRecorder("dialogue_scene_probe")
    scenes = DialogueStage()._scene_pass(
        audio, [(item.start, item.end) for item in intervals], dialogue["plot_summary"], timing,
        buckets=assign_utterances(audio, intervals), scene_ids=[item.id for item in intervals],
    )
    # Keep the archived visual extraction and replace the dialogue descriptions.
    by_window = {(row["start_time"], row["end_time"]): row["description"] for row in scenes}
    for row in vlm["scene_timeline"]:
        key = row["start_time"], row["end_time"]
        if key not in by_window:
            raise SystemExit(f"Archived VLM window has no matching dialogue: {key}")
        row["dialogue_description"] = by_window[key]
    dialogue["scenes"] = scenes
    timing.save(data / "scene_timings.json")
else:
    raise SystemExit(f"Unknown dialogue probe mode: {mode}")
(storage.get_output_dir("dialogue") / f"{video}_dialogue.json").write_text(json.dumps(dialogue, indent=2))
(storage.get_output_dir("vlm") / f"{video}_vlm.json").write_text(json.dumps(vlm, indent=2))
result = LLMStage().run(video)
print(json.dumps({key: result.get(key) for key in ("status", "output_file", "error")}), flush=True)
if result["status"] != "success":
    raise SystemExit(1)
PY
