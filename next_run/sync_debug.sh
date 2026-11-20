#!/bin/bash
# Fast-forward this checkout to origin/debug.
# Added or changed next_run/*.sh files, except this watcher, are submitted
# with sbatch. Added or changed commands/*.sh files are executed here, and
# their output is written under jobs_logs/ so scripts/push_debug_logs.py
# can publish it. A code change that touches neither directory does nothing.

set -euo pipefail

repo=/SLURM_shared/st_mdp26/open-enricher
state=/SLURM_shared/st_mdp26/actions-runner/sync-debug.sha
lock=/SLURM_shared/st_mdp26/actions-runner/sync-debug.lock
command_timeout=900

mkdir -p /SLURM_shared/st_mdp26/actions-runner "$repo/next_run" "$repo/commands" "$repo/jobs_logs"
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
declare -a commands=()
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
    if [[ "$path" =~ ^commands/[^/]+\.sh$ ]]; then
      commands+=("$path")
    fi
  done < <(git diff --name-status --find-renames "$base" HEAD)
fi

if [[ "$deps" -eq 1 ]]; then
  "$repo/.venv/bin/pip" install -r "$repo/requirements.txt"
  "$repo/.venv/bin/pip" install -e "$repo"
fi

if [[ ${#scripts[@]} -eq 0 && ${#commands[@]} -eq 0 ]]; then
  printf '%s\n' "$remote" > "$state"
  log "fast-forwarded $remote; nothing to run"
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

run_command() {
  local script=$1
  local stem short outfile rc
  stem=$(basename "$script" .sh)
  short=$(git rev-parse --short=12 HEAD)
  outfile="$repo/jobs_logs/command-${stem}-${short}.log"
  {
    printf 'script: %s\n' "$script"
    printf 'commit: %s\n' "$(git rev-parse HEAD)"
    printf 'started: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    printf '%s\n' '----'
  } > "$outfile"
  if [[ ! -f "$repo/$script" ]]; then
    printf 'missing: %s\n' "$script" >> "$outfile"
    rc=127
  else
    rc=0
    PATH="$repo/.venv/bin:$PATH" \
      LD_LIBRARY_PATH="/SLURM_shared/st_mdp26/ffmpeg-libs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
      /usr/bin/timeout --signal=TERM --kill-after=30s "${command_timeout}s" \
      /bin/bash "$repo/$script" >> "$outfile" 2>&1 || rc=$?
  fi
  {
    printf '%s\n' '----'
    printf 'exit: %s\n' "$rc"
    printf 'finished: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  } >> "$outfile"
  log "command $script exit $rc -> ${outfile#"$repo"/}"
}

for script in "${commands[@]}"; do
  run_command "$script"
done

printf '%s\n' "$remote" > "$state"
