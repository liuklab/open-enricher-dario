#!/bin/bash
#SBATCH --job-name=open-enricher-dialogue-plot
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=jobs_logs/open-enricher-dialogue-plot-%j.out
#SBATCH --error=jobs_logs/open-enricher-dialogue-plot-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?Submit from the worktree root}"
TRANSCRIPT="${1:?Pass the archived audio JSON}"
SHARED=/SLURM_shared/st_mdp26/open-enricher
DESTINATION="$ROOT/var/dialogue-plot-$SLURM_JOB_ID.json"
export PYTHONPATH="$ROOT/src"
export HF_HOME="$SHARED/dataset/models_cache/huggingface"
srun --unbuffered --export=ALL "$SHARED/.venv/bin/python" - "$TRANSCRIPT" "$DESTINATION" "$SHARED" <<'PY'
import json
import sys
from pathlib import Path

from open_enricher.runner import bootstrap_runtime

transcript, destination, shared = map(Path, sys.argv[1:])
bootstrap_runtime(env_file=shared / ".env", tee_log=False)

from open_enricher.stages.dialogue import DialogueStage, format_dialogue_lines
from open_enricher.timings import TimingRecorder

lines = format_dialogue_lines(json.loads(transcript.read_text(encoding="utf-8")))
timing = TimingRecorder("dialogue")
summary, sampled = DialogueStage()._plot_pass(lines, timing)
if not summary.strip():
    raise SystemExit("Plot summary is empty")
destination.write_text(json.dumps({
    "line_count": len(lines),
    "sampled": sampled,
    "summary": summary,
    "timings": timing.document(),
}, indent=2, ensure_ascii=False), encoding="utf-8")
PY
