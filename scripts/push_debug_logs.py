#!/usr/bin/env python3
"""Publish the working tree to origin/debug when an experiment moves.

A pass is triggered by a change under jobs_logs/ or runs/. The commit then
includes the whole tree: logs, runs, and any other project file that is
not ignored. .vscode, .github, and secret files are left out. The script does not
switch branches.

Cadence:
- while a SLURM job from jobs_logs/ is still queued or running, changes are
  pushed at most once every 10 minutes;
- when that job has left the queue, the next poll pushes immediately.

Run one pass (cron, once a minute is enough):

    python3 scripts/push_debug_logs.py --once

Or leave a process on the login node:

    python3 scripts/push_debug_logs.py --loop --poll-seconds 30

State is kept in .git/debug-log-sync.json so it is not committed.
A failed push is retried on the next pass; the 10 minute window moves
only after origin/debug has accepted the commit.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG_DIR = REPO / "jobs_logs"
STATE_PATH = REPO / ".git" / "debug-log-sync.json"
LOCK_PATH = REPO / ".git" / "debug-log-sync.lock"
BRANCH = "debug"
REMOTE = "origin"
INTERVAL_SECONDS = 10 * 60
AUTHOR_NAME = "Dario Napolitano"
AUTHOR_EMAIL = "darionapo2.0@gmail.com"
JOB_FILE = re.compile(r"-(\d+)\.(?:out|err)$")
ADD_PATHSPEC = [
    ".",
    ":(exclude).vscode",
    ":(exclude).github",
    ":(exclude).env",
    ":(exclude).env.*",
    ":(exclude)config/secrets.toml",
]


class SyncError(RuntimeError):
    """A local git or SLURM check failed before any commit."""


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise SyncError(f"git {' '.join(args)} failed: {detail}")
    return result


def current_branch() -> str:
    return git("branch", "--show-current").stdout.strip()


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"last_push": 0.0, "push_pending": False}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def log_job_ids() -> set[str]:
    if not LOG_DIR.is_dir():
        return set()
    return {
        match.group(1)
        for path in LOG_DIR.iterdir()
        if path.is_file() and (match := JOB_FILE.search(path.name))
    }


def is_excluded(path: str) -> bool:
    return (
        path == ".env"
        or path.startswith((".vscode/", ".github/", ".env"))
        or path == "config/secrets.toml"
    )


def is_trigger(path: str) -> bool:
    return path.startswith(("jobs_logs/", "runs/"))


def changed_paths() -> list[str]:
    result = git("status", "--porcelain", "--untracked-files=all")
    paths: list[str] = []
    for line in result.stdout.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path and not is_excluded(path):
            paths.append(path)
    return paths


def running_job_ids(job_ids: set[str]) -> set[str]:
    """Return which of job_ids are still in this user's queue.

    Old log files are not passed to squeue: an unknown id makes SLURM
    reject the whole query. A controller failure aborts the pass.
    """
    if not job_ids:
        return set()
    user = os.environ.get("USER")
    if not user:
        raise SyncError("USER is unset; refusing to guess which jobs are running")
    try:
        result = subprocess.run(
            ["squeue", "-h", "-u", user, "-o", "%i"],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise SyncError("squeue is not on PATH; refusing to guess which jobs finished") from exc
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        raise SyncError(f"squeue failed: {message}")
    visible = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    return visible & job_ids


def job_id_of(path: str) -> str | None:
    match = JOB_FILE.search(path)
    return match.group(1) if match else None


def snapshot_is_due(triggers: list[str], running: set[str], last_push: float, now: float) -> bool:
    if not triggers:
        return False
    dirty_jobs = {job_id for path in triggers if (job_id := job_id_of(path))}
    finished = dirty_jobs - running
    if finished or not running:
        return True
    return now - last_push >= INTERVAL_SECONDS


def commit_snapshot(paths: list[str], running: set[str]) -> None:
    git("add", "-A", "--", *ADD_PATHSPEC)
    dirty_jobs = {job_id for path in paths if (job_id := job_id_of(path))}
    finished = sorted(dirty_jobs - running)
    still_running = sorted(dirty_jobs & running)
    parts = []
    if finished:
        parts.append("finished " + ", ".join(finished))
    if still_running:
        parts.append("running " + ", ".join(still_running))
    detail = f" ({'; '.join(parts)})" if parts else ""
    message = f"Sync job logs and runs{detail}."
    git(
        "-c",
        f"user.name={AUTHOR_NAME}",
        "-c",
        f"user.email={AUTHOR_EMAIL}",
        "commit",
        "-m",
        message,
        "--",
        *paths,
    )


def origin_has_debug() -> bool:
    result = git("rev-parse", "--verify", "--quiet", f"{REMOTE}/{BRANCH}", check=False)
    return result.returncode == 0


def ahead_of_origin() -> bool:
    if not origin_has_debug():
        return True
    result = git("rev-list", "--count", f"{REMOTE}/{BRANCH}..HEAD")
    return int(result.stdout.strip() or "0") > 0


def push_debug() -> None:
    git("push", REMOTE, f"HEAD:{BRANCH}")


def sync_once(dry_run: bool) -> str:
    branch = current_branch()
    if branch != BRANCH:
        raise SyncError(f"HEAD is {branch or 'detached'}; this watcher only publishes {BRANCH}")

    state = load_state()
    now = time.time()
    changes = changed_paths()
    triggers = [path for path in changes if is_trigger(path)]
    running = running_job_ids(log_job_ids())
    due = snapshot_is_due(triggers, running, float(state.get("last_push") or 0), now)

    if dry_run:
        action = "would commit the working tree" if due else "would hold"
        return (
            f"dry-run: {action}; trigger files={len(triggers)}, "
            f"tree files={len(changes)}, running jobs={sorted(running)}, "
            f"push pending={bool(state.get('push_pending'))}"
        )

    if due:
        commit_snapshot(changes, running)
        state["push_pending"] = True
        save_state(state)

    if not state.get("push_pending"):
        if triggers:
            return f"holding {len(triggers)} changed log and run file(s) until the next 10 minute push"
        return "jobs_logs and runs are unchanged"

    if not ahead_of_origin():
        state["push_pending"] = False
        state["last_push"] = now
        save_state(state)
        return "logs already match origin/debug"

    push_debug()
    state["push_pending"] = False
    state["last_push"] = now
    save_state(state)
    return f"pushed {BRANCH} to {REMOTE}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run a single pass and exit")
    mode.add_argument("--loop", action="store_true", help="poll until interrupted")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("another debug log sync is already running", file=sys.stderr)
            return 0

        while True:
            try:
                print(sync_once(args.dry_run))
            except SyncError as exc:
                print(exc, file=sys.stderr)
                return 1
            if not args.loop:
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
