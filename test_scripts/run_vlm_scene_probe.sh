#!/bin/bash
#SBATCH --job-name=open-enricher-vlm-scene
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=jobs_logs/open-enricher-vlm-scene-%j.out
#SBATCH --error=jobs_logs/open-enricher-vlm-scene-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
ARCHIVE="${1:?Pass the episode archive path}"
VIDEO="${2:?Pass the source filename}"
SHARED=/SLURM_shared/st_mdp26/open-enricher
export PYTHONPATH="$ROOT/src"
export HF_HOME="$SHARED/dataset/models_cache/huggingface"
srun --unbuffered --export=ALL "$SHARED/.venv/bin/python" - \
    "$ARCHIVE" "$VIDEO" "$ROOT/var/vlm-scene-$SLURM_JOB_ID.json" "$SHARED" <<'PY'
import json
import sys
from pathlib import Path

from open_enricher.runner import bootstrap_runtime

archive, video, destination, shared = sys.argv[1:]
archive = Path(archive)
bootstrap_runtime(env_file=Path(shared) / ".env", tee_log=False)

from open_enricher.stages.vlm_core.config import build_configs
from open_enricher.stages.vlm_core.segments import run_segmented

def artifact(suffix):
    return json.loads((archive / f"{video}_{suffix}.json").read_text(encoding="utf-8"))

group = artifact("frame_plan")["groups"][0]
span = artifact("vlm_frames")["spans"][0]
graph = artifact("graph")
scene = artifact("dialogue")["scenes"][0]
paths = [frame["file"] for frame in span["frames"]]
if len(paths) != group["frames"] or any(not Path(path).is_file() for path in paths):
    raise SystemExit("Archived scene frames are incomplete")
shown = ", ".join(f"{stamp:.1f}s" for stamp in group["timestamps"])
scene.update(
    image_paths=paths,
    max_frames=group["frames"],
    prompt_extra=f"Current frames at source times: {shown}.\nSupported character relations: {graph['account']}",
)
model, config = build_configs()
raw, parsed, valid, _, _ = run_segmented(
    Path(shared) / "dataset/video_inputs" / video,
    video, model, config, [scene], experiment_id="12-combined", vlm_memory_scenes=2,
)
Path(destination).write_text(json.dumps({
    "valid": valid, "raw": raw, "parsed": parsed, "frames": paths,
}, indent=2, ensure_ascii=False), encoding="utf-8")
if not valid:
    raise SystemExit("Scene output did not parse")
PY
