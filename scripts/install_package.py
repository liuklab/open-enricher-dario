"""Clone or update open-enricher on a Colab VM from git, then editable-install it.

colab exec only ships this file. The library is fetched onto the VM disk
(/content/open-enricher), not from Google Drive.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_GIT_URL = "https://github.com/liuklab/open-enricher.git"
DEFAULT_GIT_REF = "main"
DEFAULT_REPO_DIR = "/content/open-enricher"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clone or pull open-enricher, then pip install -e.")
    parser.add_argument(
        "--git-url",
        default=os.environ.get("OPEN_ENRICHER_GIT_URL", DEFAULT_GIT_URL),
        help="Git remote URL (HTTPS).",
    )
    parser.add_argument(
        "--git-ref",
        default=os.environ.get("OPEN_ENRICHER_GIT_REF", DEFAULT_GIT_REF),
        help="Branch, tag, or commit to check out.",
    )
    parser.add_argument(
        "--repo-dir",
        default=os.environ.get("OPEN_ENRICHER_REPO", DEFAULT_REPO_DIR),
        help="Checkout path on the VM.",
    )
    return parser.parse_args()


def to_https_url(url: str) -> str:
    """Normalize SSH GitHub remotes to HTTPS so the VM can clone without SSH keys."""

    url = url.strip()
    if url.startswith("git@github.com:"):
        return "https://github.com/" + url.split(":", 1)[1]
    if url.startswith("ssh://git@github.com/"):
        return "https://github.com/" + url.removeprefix("ssh://git@github.com/")
    return url


def authenticated_url(url: str) -> str:
    """Inject GITHUB_TOKEN / GIT_TOKEN for private HTTPS remotes. Never print this URL."""

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GIT_TOKEN") or ""
    if not token:
        return url
    if not url.startswith("https://"):
        return url
    rest = url.removeprefix("https://")
    if rest.startswith("x-access-token:") or rest.startswith(token):
        return url
    return f"https://x-access-token:{token}@{rest}"


def run_git(args: list[str], cwd: Path | None = None) -> None:
    subprocess.check_call(["git", *args], cwd=str(cwd) if cwd else None)


def sync_repo(repo: Path, url: str, ref: str) -> None:
    clone_url = authenticated_url(url)
    try:
        if (repo / ".git").is_dir():
            run_git(["remote", "set-url", "origin", clone_url], cwd=repo)
            run_git(["fetch", "origin"], cwd=repo)
        else:
            repo.parent.mkdir(parents=True, exist_ok=True)
            run_git(["clone", clone_url, str(repo)])
            run_git(["fetch", "origin"], cwd=repo)

        origin_ref = f"origin/{ref}"
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", origin_ref],
            cwd=str(repo),
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            run_git(["checkout", "-B", ref, origin_ref], cwd=repo)
        else:
            run_git(["checkout", "--detach", ref], cwd=repo)
    finally:
        if (repo / ".git").is_dir():
            run_git(["remote", "set-url", "origin", url], cwd=repo)


def main() -> None:
    args = parse_args()
    public_url = to_https_url(args.git_url)
    repo = Path(args.repo_dir)
    print(f"Syncing {public_url} @ {args.git_ref} into {repo}")
    sync_repo(repo, public_url, args.git_ref)

    if not (repo / "pyproject.toml").is_file():
        raise FileNotFoundError(f"pyproject.toml missing under {repo}")

    cmd = [sys.executable, "-m", "pip", "install", "-e", str(repo)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Installed open-enricher from", repo)


if __name__ == "__main__":
    main()
