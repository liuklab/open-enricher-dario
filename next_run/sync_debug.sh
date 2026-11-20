#!/bin/bash
# Fast-forward this checkout to origin/debug. Each next_run/*.sh that was
# added or changed, except this watcher, is submitted with sbatch.
# Log and archive pushes do not launch jobs. Neither does a code change
# that leaves the planned-run scripts untouched.

set -euo pipefail

repo=/SLURM_shared/st_mdp26/open-enricher
state=/SLURM_shared/st_mdp26/actions-runner/sync-debug.sha
lock=/SLURM_shared/st_mdp26/actions-runner/sync-debug.lock

mkdir -p /SLURM_shared/st_mdp26/actions-runner "$repo/next_run" "$repo/jobs_logs"
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
previous=""
if [[ -f "$state" ]]; then
  previous=$(cat "$state")
fi
if [[ "$previous" == "$remote" ]]; then
  exit 0
fi

if ! git merge-base --is-ancestor HEAD "$remote"; then
  log "origin/debug is not a fast-forward of HEAD; skipping"
  exit 0
fi

if [[ -n "$previous" ]] && ! git cat-file -e "${previous}^{commit}" 2>/dev/null; then
  log "saved tip $previous is gone; marking $remote applied without submitting"
  printf '%s\n' "$remote" > "$state"
  exit 0
fi

git merge --ff-only origin/debug

base=$previous
if [[ -z "$base" ]]; then
  base=$(git rev-parse HEAD^ 2>/dev/null || echo "")
fi

declare -a scripts=()
deps=0
if [[ -n "$base" ]]; then
  while IFS=$'\t' read -r status path extra; do
    [[ -z "${status:-}" ]] && continue
    case "$status" in
      A|M|T) ;;
      R*|C*) path=$extra ;;
      *) continue ;;
    esac
    case "$path" in
      requirements.txt|pyproject.toml) deps=1 ;;
    esac
    if [[ "$path" =~ ^next_run/[^/]+\.sh$ && "$path" != "next_run/sync_debug.sh" ]]; then
      scripts+=("$path")
    fi
  done < <(git diff --name-status --find-renames "$base" HEAD)
fi

if [[ "$deps" -eq 1 ]]; then
  "$repo/.venv/bin/pip" install -r "$repo/requirements.txt"
  "$repo/.venv/bin/pip" install -e "$repo"
fi

if [[ ${#scripts[@]} -eq 0 ]]; then
  printf '%s\n' "$remote" > "$state"
  log "fast-forwarded $remote; no planned run scripts to submit"
  exit 0
fi

failed=0
for script in "${scripts[@]}"; do
  if [[ ! -f "$repo/$script" ]]; then
    log "missing $script"
    failed=1
    continue
  fi
  if job=$(/usr/bin/sbatch --parsable "$repo/$script"); then
    log "submitted SLURM job $job for $script"
  else
    log "sbatch failed for $script"
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  log "leaving saved tip at ${previous:-none} so the next pass retries"
  exit 1
fi

printf '%s\n' "$remote" > "$state"
