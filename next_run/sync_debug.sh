#!/bin/bash
# Fast-forward this checkout to origin/debug and submit next_run/run_experiments.sh
# when the incoming commits touch something other than logs, runs, or the dataset.
# Pushes that only publish job output do not start another GPU batch.

set -euo pipefail

repo=/SLURM_shared/st_mdp26/open-enricher
state=/SLURM_shared/st_mdp26/actions-runner/sync-debug.sha
lock=/SLURM_shared/st_mdp26/actions-runner/sync-debug.lock

mkdir -p /SLURM_shared/st_mdp26/actions-runner
exec 9>"$lock"
if ! flock -n 9; then
  exit 0
fi

log() {
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

cd "$repo"
git fetch origin debug
remote=$(git rev-parse origin/debug)
if [[ -f "$state" && "$(cat "$state")" == "$remote" ]]; then
  exit 0
fi

if ! git merge-base --is-ancestor HEAD "$remote"; then
  log "origin/debug is not a fast-forward of HEAD; skipping"
  exit 0
fi

mapfile -t changed < <(git diff --name-only HEAD "$remote")
git merge --ff-only origin/debug
printf '%s\n' "$remote" > "$state"

code=0
deps=0
for path in "${changed[@]}"; do
  case "$path" in
    jobs_logs/*|runs/*|dataset/*) ;;
    requirements.txt|pyproject.toml)
      deps=1
      code=1
      ;;
    *)
      code=1
      ;;
  esac
done

if [[ "$code" -eq 0 ]]; then
  log "fast-forwarded logs or archives only ($remote)"
  exit 0
fi

if [[ "$deps" -eq 1 ]]; then
  "$repo/.venv/bin/pip" install -r "$repo/requirements.txt"
  "$repo/.venv/bin/pip" install -e "$repo"
fi

mkdir -p "$repo/jobs_logs"
job=$(/usr/bin/sbatch --parsable "$repo/next_run/run_experiments.sh")
log "submitted SLURM job $job for $remote"
