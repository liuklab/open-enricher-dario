"""Temporary worktrees and experiment branches forked from exp2."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from src.open_enricher.experiments.config import (
    BASE_BRANCH,
    EXPERIMENT_BRANCH_PREFIX,
    REPO_ROOT,
)


def _run(
    args: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd or REPO_ROOT),
        check=check,
        text=True,
        capture_output=True,
    )


def branch_name(experiment_id: str) -> str:
    return f"{EXPERIMENT_BRANCH_PREFIX}/{experiment_id}"


def worktree_path(experiment_id: str) -> Path:
    return REPO_ROOT.parent / f"open-enricher-run-{experiment_id}"


def ensure_experiment_worktree(experiment_id: str) -> Path:
    """Create branch exp2-run/<id> from BASE_BRANCH tip and a temporary worktree.

    Does not change the checkout of the primary working tree.
    """
    branch = branch_name(experiment_id)
    path = worktree_path(experiment_id)
    git_marker = path / ".git"
    if git_marker.exists():
        return path

    existing = _run(["git", "rev-parse", "--verify", branch], check=False)
    if existing.returncode != 0:
        base_tip = _run(["git", "rev-parse", BASE_BRANCH])
        created = _run(
            ["git", "branch", branch, base_tip.stdout.strip()],
            check=False,
        )
        if created.returncode != 0:
            raise RuntimeError(
                f"Failed to create branch {branch}: {created.stderr or created.stdout}"
            )

    if path.exists():
        shutil.rmtree(path)
    result = _run(["git", "worktree", "add", str(path), branch], check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to add worktree for {branch}: {result.stderr or result.stdout}"
        )
    return path


def commit_all(worktree: Path, message: str) -> None:
    _run(["git", "add", "-A"], cwd=worktree)
    status = _run(["git", "status", "--porcelain"], cwd=worktree)
    if not status.stdout.strip():
        return
    _run(["git", "commit", "-m", message], cwd=worktree)


def remove_worktree(experiment_id: str) -> None:
    path = worktree_path(experiment_id)
    _run(["git", "worktree", "remove", "--force", str(path)], check=False)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def install_pipeline_spec(worktree: Path, variant_path: Path) -> Path:
    """Copy the variant JSON onto the path the worker loads."""
    dest = (
        worktree
        / "src"
        / "open_enricher"
        / "remote"
        / "stages"
        / "pipeline_spec.json"
    )
    dest.write_text(variant_path.read_text(encoding="utf-8"), encoding="utf-8")
    return dest


def copy_experiment_md(worktree: Path, runs_dir: Path) -> None:
    src = worktree / "EXPERIMENT.md"
    if src.is_file():
        dest = runs_dir / "EXPERIMENT.md"
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
