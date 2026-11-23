#!/bin/bash
#SBATCH --job-name=open-enricher-graph-control
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=jobs_logs/open-enricher-graph-control-%j.out
#SBATCH --error=jobs_logs/open-enricher-graph-control-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
cd "$ROOT"
SHARED=/SLURM_shared/st_mdp26/open-enricher
SOURCE="${1:?Pass the integrated clip dataset path}"
VIDEO="${2:?Pass the video filename}"
MODE="${3:-writer-only}"
GRAPH_OVERRIDE="${4:-}"
DATA="$ROOT/var/graph-profile-control-$SLURM_JOB_ID"
mkdir -p "$DATA/outputs/graph_results"
ln -s "${GRAPH_SOURCE_VIDEO_INPUTS:-$SOURCE/video_inputs}" "$DATA/video_inputs"
ln -s "$SHARED/dataset/models_cache" "$DATA/models_cache"
stages=(audio_results dialogue_results memory_results)
if [[ "$MODE" != full-graph ]]; then stages+=(vlms_results); fi
for stage in "${stages[@]}"; do
    ln -s "$SOURCE/outputs/$stage" "$DATA/outputs/$stage"
done
export OPEN_ENRICHER_DATASET_ROOT="$DATA" OPEN_ENRICHER_VIDEO_INPUTS="$DATA/video_inputs"
export PYTHONPATH="$ROOT/src" HF_HOME="$DATA/models_cache/huggingface"
export OPEN_ENRICHER_PIPELINE_SPEC="$ROOT/src/open_enricher/experiments/variants/12-combined.json"
FFMPEG=/SLURM_shared/st_mdp26/ffmpeg-libs
SITE="$SHARED/.venv/lib/python3.12/site-packages/nvidia"
export PATH="$FFMPEG/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/cudnn/lib:$SITE/cublas/lib:$SITE/cuda_runtime/lib:$SITE/cufft/lib:$SITE/curand/lib:$SITE/cusolver/lib:$SITE/cusparse/lib:$SITE/cuda_nvrtc/lib:$SITE/nvjitlink/lib:$FFMPEG/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
srun --unbuffered --export=ALL "$SHARED/.venv/bin/python" - "$VIDEO" "$SOURCE" "$SHARED" "$MODE" "$GRAPH_OVERRIDE" <<'PY'
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from open_enricher import storage
from open_enricher.graph import compact_account, recurring_interaction_edges
from open_enricher.runner import bootstrap_runtime

video, source, shared, mode, override = sys.argv[1:]
load_dotenv(Path(shared) / ".env")
bootstrap_runtime(tee_log=False)
from open_enricher.stages.llm import LLMStage

graph_name = f"{video}_graph.json"
graph_path = Path(override) if override else Path(source) / "outputs/graph_results" / graph_name
graph = json.loads(graph_path.read_text())
if override:
    dialogue = json.loads(
        (Path(source) / "outputs/dialogue_results" / f"{video}_dialogue.json").read_text()
    )
    if graph.get("video") != video or graph.get("status") != "success":
        raise SystemExit("The graph override must be a successful replay of the source video")
    graph["attempt_id"] = dialogue["attempt_id"]
os.environ["OPEN_ENRICHER_ATTEMPT_ID"] = graph["attempt_id"]
if mode == "writer-only":
    control_edges = [edge for edge in graph["edges"] if edge["kind"] != "speaks_in_presence_of"]
    if len(control_edges) == len(graph["edges"]):
        raise SystemExit("The integrated graph contains no interaction evidence to ablate")
    graph["edges"] = control_edges
    graph["account"] = compact_account(graph["nodes"], control_edges)
    destination = storage.get_output_dir("character_graph") / graph_name
    destination.write_text(json.dumps(graph), encoding="utf-8")
elif mode in {"retain", "recurring"}:
    if mode == "recurring":
        evidence = recurring_interaction_edges(graph["edges"])
        graph["account"] = compact_account(graph["nodes"], evidence)
    destination = storage.get_output_dir("character_graph") / graph_name
    destination.write_text(json.dumps(graph), encoding="utf-8")
elif mode == "full-graph":
    from open_enricher.stages.vlm import VLMStage

    vlm = VLMStage().run(video)
    print(json.dumps({key: vlm.get(key) for key in ("status", "output_file", "error")}), flush=True)
    if vlm["status"] != "success":
        raise SystemExit("The graph-free VLM replay did not complete successfully")
else:
    raise SystemExit(f"Unknown control mode: {mode}")
result = LLMStage().run(video)
print(json.dumps({key: result.get(key) for key in ("status", "output_file", "error")}))
if result["status"] != "success":
    raise SystemExit(1)
PY
