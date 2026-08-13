"""Extract the uploaded project tarball on a Colab VM and editable-install it.

colab exec ships only this file. The archive is already at /content/open-enricher.tgz.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

TARBALL = Path("/content/open-enricher.tgz")
REPO = Path("/content/open-enricher")


def extract_tarball() -> None:
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.mkdir(parents=True)
    with tarfile.open(TARBALL, "r:gz") as archive:
        try:
            archive.extractall(REPO, filter="data")
        except TypeError:
            archive.extractall(REPO)


def main() -> None:
    if not TARBALL.is_file():
        raise FileNotFoundError(f"Missing tarball: {TARBALL}")

    extract_tarball()

    if not (REPO / "pyproject.toml").is_file():
        raise FileNotFoundError(f"pyproject.toml missing under {REPO}")

    cmd = [sys.executable, "-m", "pip", "install", "-e", str(REPO)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    src = REPO / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    importlib.invalidate_caches()
    import open_enricher  # noqa: F401

    print("Installed open-enricher from", REPO)
    print("Import path:", open_enricher.__file__)


if __name__ == "__main__":
    main()
