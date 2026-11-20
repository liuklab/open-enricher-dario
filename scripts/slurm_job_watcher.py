#!/usr/bin/env python3
"""Post one Slack wakeup when tracked Slurm jobs reach a terminal state.

Poll ``squeue`` every 30 seconds. Job ids come from a watch file (append
one id per line) and from this user's queued jobs whose name contains
``open-enricher``. When an id leaves the queue, ``sacct`` supplies State,
ExitCode and Elapsed. Every terminal outcome is included. Jobs that
finish in the same poll share one message. A state file records ids
already sent so a job is never posted twice.

The webhook is read from a file and is never written to the log, the
Slack text, or an exception message.

Run from the repo root::

    mkdir -p var/slurm_watcher
    .venv/bin/python -u scripts/slurm_job_watcher.py

``var/`` is gitignored. Append ids with::

    echo JOBID >> var/slurm_watcher/job_ids.txt
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "var" / "slurm_watcher"
DEFAULT_TRACK = DEFAULT_DIR / "job_ids.txt"
DEFAULT_STATE = DEFAULT_DIR / "state.json"
DEFAULT_LOG = DEFAULT_DIR / "watcher.log"
DEFAULT_WEBHOOK = Path.home() / ".config" / "open-enricher" / "slack_webhook"
POLL_SECONDS = 30
PREFIX = "OPEN_ENRICHER_JOB_DONE"

TERMINAL_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "TIMEOUT",
    }
)

_URL_RE = re.compile(r"https?://\S+")
_OPEN_ENRICHER_NAME = re.compile(r"open-enricher", re.IGNORECASE)

logger = logging.getLogger("slurm_job_watcher")
_LOCK_HANDLE: object | None = None


@dataclass(frozen=True)
class LiveJob:
    job_id: str
    name: str
    state: str


@dataclass(frozen=True)
class Accounting:
    job_id: str
    name: str
    state: str
    exit_code: str
    elapsed: str
    workdir: str


@dataclass(frozen=True)
class FinishedJob:
    job_id: str
    name: str
    state: str
    exit_code: str
    elapsed: str
    logs: tuple[str, ...]


def redact(text: str) -> str:
    """Remove URL-shaped text so a webhook cannot land in a log line."""
    return _URL_RE.sub("[redacted-url]", text)


def is_terminal(state: str) -> bool:
    """True when the Slurm state will not return to the queue."""
    token = state.strip().split()[0].split("+")[0].upper() if state.strip() else ""
    return token in TERMINAL_STATES


def parse_job_ids(text: str) -> list[str]:
    """Return numeric job ids, ignoring blanks and ``#`` comments."""
    found: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or not line.isdigit() or line in seen:
            continue
        seen.add(line)
        found.append(line)
    return found


def matches_open_enricher(name: str) -> bool:
    """True for jobs this repo submits under an open-enricher name."""
    return _OPEN_ENRICHER_NAME.search(name) is not None


def format_message(jobs: list[FinishedJob]) -> str:
    """One structured wakeup. Several jobs stay in this single message."""
    lines = [PREFIX, f"count: {len(jobs)}"]
    for job in jobs:
        logs = ",".join(job.logs) if job.logs else "unknown"
        lines.append(
            " ".join(
                [
                    f"job_id={job.job_id}",
                    f"name={job.name}",
                    f"state={job.state}",
                    f"exit={job.exit_code}",
                    f"elapsed={job.elapsed}",
                    f"logs={logs}",
                ]
            )
        )
    return "\n".join(lines)


def find_log_paths(log_dir: Path, job_id: str) -> tuple[str, ...]:
    """Paths under ``jobs_logs`` whose filename contains the job id."""
    if not log_dir.is_dir():
        return ()
    matches = sorted(
        path for path in log_dir.iterdir() if path.is_file() and job_id in path.name
    )
    return tuple(str(path) for path in matches)


def load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"notified": {}, "watching": []}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("state file is unreadable; starting from an empty record")
        return {"notified": {}, "watching": []}
    if not isinstance(document, dict):
        return {"notified": {}, "watching": []}
    notified = document.get("notified")
    watching = document.get("watching")
    if not isinstance(notified, dict):
        notified = {}
    if not isinstance(watching, list):
        watching = []
    return {
        "notified": {str(key): value for key, value in notified.items()},
        "watching": [str(item) for item in watching if str(item).isdigit()],
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_track_file(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        return parse_job_ids(path.read_text(encoding="utf-8"))
    except OSError as exc:
        logger.warning("could not read track file: %s", redact(str(exc)))
        return []


def ensure_track_file(path: Path) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# One Slurm job id per line. Blank lines and # comments are ignored.\n",
        encoding="utf-8",
    )


def _run(command: list[str]) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("command failed: %s", redact(str(exc)))
        return None


def query_live_jobs(user: str) -> dict[str, LiveJob] | None:
    """Jobs still queued or running for ``user``. ``None`` means squeue failed."""
    completed = _run(["squeue", "-u", user, "-h", "-o", "%A|%j|%T"])
    if completed is None or completed.returncode != 0:
        detail = ""
        if completed is not None:
            detail = redact((completed.stderr or completed.stdout or "").strip())
        logger.warning("squeue failed (%s); this poll will not treat jobs as finished", detail)
        return None
    live: dict[str, LiveJob] = {}
    for line in completed.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        live[parts[0]] = LiveJob(job_id=parts[0], name=parts[1], state=parts[2])
    return live


def query_accounting(job_id: str) -> Accounting | None:
    """Allocation row from sacct, or ``None`` when accounting has no row yet."""
    completed = _run(
        [
            "sacct",
            "-n",
            "-P",
            "-X",
            "-j",
            job_id,
            "--format=JobID,JobName,State,ExitCode,Elapsed,WorkDir",
        ]
    )
    if completed is None or completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        parts = line.split("|")
        if len(parts) < 6 or parts[0] != job_id:
            continue
        return Accounting(
            job_id=parts[0],
            name=parts[1] or "unknown",
            state=parts[2] or "unknown",
            exit_code=parts[3] or "unknown",
            elapsed=parts[4] or "unknown",
            workdir=parts[5],
        )
    return None


def post_slack(webhook_file: Path, text: str) -> None:
    """POST ``text``. Raises RuntimeError whose message has no URL."""
    if PREFIX not in text:
        raise RuntimeError("refusing to post a message without the job-done prefix")
    if _URL_RE.search(text):
        raise RuntimeError("refusing to post a message that contains a URL")
    try:
        webhook = webhook_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"could not read slack webhook file: {exc.strerror or 'io error'}") from None
    if not webhook.startswith("https://"):
        raise RuntimeError("slack webhook file does not contain an https URL")
    if webhook in text:
        raise RuntimeError("refusing to post a message that contains the webhook")
    request = urllib.request.Request(
        webhook,
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status = getattr(response, "status", 0)
            body = response.read().decode("utf-8", errors="replace").strip()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"slack post rejected with HTTP {exc.code}") from None
    except urllib.error.URLError:
        raise RuntimeError("slack post failed: network error") from None
    except Exception:
        raise RuntimeError("slack post failed") from None
    if status != 200 or body != "ok":
        raise RuntimeError(f"slack post failed with HTTP {status}")


def poll_once(
    *,
    track_file: Path,
    state_file: Path,
    webhook_file: Path,
    user: str,
    log_dir: Path,
    live_jobs: Callable[[str], dict[str, LiveJob] | None] | None = None,
    accounting: Callable[[str], Accounting | None] | None = None,
    poster: Callable[[Path, str], None] | None = None,
) -> list[str]:
    """
    Notify for tracked jobs that have left the queue.

    Returns the job ids included in this poll's message (empty when none).
    A failed post leaves those ids unmarked so the next poll retries.
    """
    query_live = live_jobs or query_live_jobs
    query_acct = accounting or query_accounting
    send = poster or post_slack

    state = load_state(state_file)
    notified: dict[str, Any] = state["notified"]
    watching = set(state["watching"])
    watching.update(read_track_file(track_file))

    live = query_live(user)
    if live is None:
        return []
    for job in live.values():
        if matches_open_enricher(job.name):
            watching.add(job.job_id)

    pending = sorted(watching - set(notified), key=int)
    finished: list[FinishedJob] = []
    for job_id in pending:
        if job_id in live:
            continue
        record = query_acct(job_id)
        if record is None or not is_terminal(record.state):
            continue
        finished.append(
            FinishedJob(
                job_id=record.job_id,
                name=record.name,
                state=record.state,
                exit_code=record.exit_code,
                elapsed=record.elapsed,
                logs=find_log_paths(log_dir, record.job_id),
            )
        )

    state["watching"] = sorted(watching - set(notified), key=int)
    if not finished:
        save_state(state_file, state)
        return []

    message = format_message(finished)
    try:
        send(webhook_file, message)
    except Exception as exc:  # noqa: BLE001 - retry next poll; never log a URL
        logger.error("slack post failed: %s", redact(str(exc)))
        save_state(state_file, state)
        return []

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for job in finished:
        notified[job.job_id] = {
            "name": job.name,
            "state": job.state,
            "exit": job.exit_code,
            "elapsed": job.elapsed,
            "logs": list(job.logs),
            "notified_at": stamp,
        }
    state["notified"] = notified
    state["watching"] = sorted(watching - set(notified), key=int)
    save_state(state_file, state)
    logger.info("posted %s for jobs %s", PREFIX, ",".join(job.job_id for job in finished))
    logger.info("message %s", message.replace("\n", " | "))
    return [job.job_id for job in finished]


def _configure_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False


def _hold_lock(path: Path) -> None:
    """Exclusive non-blocking lock held for the life of the process."""
    global _LOCK_HANDLE
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise SystemExit("another slurm job watcher is already running")
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    _LOCK_HANDLE = handle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Notify Slack once when tracked Slurm jobs finish.")
    parser.add_argument("--track-file", type=Path, default=DEFAULT_TRACK)
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--webhook-file", type=Path, default=DEFAULT_WEBHOOK)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "jobs_logs")
    parser.add_argument("--user", default=os.environ.get("USER", ""))
    parser.add_argument("--interval", type=float, default=POLL_SECONDS)
    parser.add_argument("--once", action="store_true", help="Poll one time and exit.")
    args = parser.parse_args(argv)
    if not args.user:
        raise SystemExit("set --user or the USER environment variable")
    if args.interval <= 0:
        raise SystemExit("--interval must be positive")

    _configure_logging(args.log_file)
    ensure_track_file(args.track_file)
    _hold_lock(args.state_file.with_name("watcher.lock"))
    logger.info(
        "watcher started pid=%s user=%s interval=%ss track=%s",
        os.getpid(),
        args.user,
        args.interval,
        args.track_file,
    )
    while True:
        try:
            poll_once(
                track_file=args.track_file,
                state_file=args.state_file,
                webhook_file=args.webhook_file,
                user=args.user,
                log_dir=args.log_dir,
            )
        except Exception as exc:  # noqa: BLE001 - keep the poller alive
            logger.error("poll failed: %s", redact(str(exc)))
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
