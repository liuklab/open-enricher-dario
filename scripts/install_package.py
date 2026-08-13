"""Install open-enricher on a Colab VM from a Drive checkout.

colab exec only ships this file, so the library must already live on Drive.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO_DIR = "/content/drive/MyDrive/open_enricher/code"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Editable-install open-enricher from Drive.")
    parser.add_argument(
        "--repo-dir",
        default=os.environ.get("OPEN_ENRICHER_REPO", DEFAULT_REPO_DIR),
        help="Path to this repository on the VM (Drive checkout).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_dir)
    if not repo.is_dir():
        raise FileNotFoundError(
            f"Repository not found: {repo}. Mount Drive and copy the repo there first."
        )
    if not (repo / "pyproject.toml").is_file():
        raise FileNotFoundError(f"pyproject.toml missing under {repo}")

    cmd = [sys.executable, "-m", "pip", "install", "-e", str(repo)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("Installed open-enricher from", repo)


if __name__ == "__main__":
    main()
