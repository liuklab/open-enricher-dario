#!/bin/bash
#SBATCH --job-name=open-enricher-eval-retry
#SBATCH --account=studenti
#SBATCH --partition=all-nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=jobs_logs/open-enricher-eval-retry-%j.out
#SBATCH --error=jobs_logs/open-enricher-eval-retry-%j.err

set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
SHARED=/SLURM_shared/st_mdp26/open-enricher
ARCHIVE="${1:?Pass the profile archive directory}"
VIDEO="${2:?Pass the video filename}"
REFERENCE="${3:-$SHARED/synopsis}"
DATA="$ROOT/var/eval-$SLURM_JOB_ID"
mkdir -p "$DATA/outputs/llms_results"
cp "$ARCHIVE/${VIDEO}_llm.json" "$DATA/outputs/llms_results/"
export OPEN_ENRICHER_DATASET_ROOT="$DATA" PYTHONPATH="$ROOT/src"
export PATH="$SHARED/.venv/bin:$PATH"
srun --unbuffered --export=ALL "$SHARED/.venv/bin/python" - "$VIDEO" "$SHARED" "$REFERENCE" <<'PY'
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from open_enricher.evaluation.run import evaluate_videos

video, project, reference = sys.argv[1:]
load_dotenv(Path(project) / ".env")
results = evaluate_videos([video], with_synopsis=True, synopsis_dir=Path(reference))
print(json.dumps(results), flush=True)
if any(row["status"] not in {"success", "degraded"} for row in results):
    raise SystemExit(1)
PY
