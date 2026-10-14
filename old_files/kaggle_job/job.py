"""Run open-enricher inference on a Kaggle kernel.

``launch.py`` uploads two private datasets and then this file:

  - ``darionapolitano123/open-enricher-dataset`` mounts the videos at
    ``/kaggle/input/open-enricher-dataset``.
  - ``darionapolitano123/open-enricher-code`` mounts this checkout as
    ``open-enricher.tgz``. Paths inside that archive already point at the
    Kaggle input and at ``/kaggle/working/outputs``.

The archive is extracted under ``/tmp`` so the checkout and ``.env`` are not
part of the kernel output. ``scripts/run.py`` writes textual metadata under
``/kaggle/working``, which ``kaggle kernels output`` downloads.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

CODE_TARBALL = Path("/kaggle/input/open-enricher-code/open-enricher.tgz")
REPO = Path("/tmp/open-enricher")


def extract_checkout() -> None:
    """Unpack the code dataset into a writable directory."""

    if not CODE_TARBALL.is_file():
        raise FileNotFoundError(f"Missing code archive: {CODE_TARBALL}")
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.mkdir(parents=True)
    with tarfile.open(CODE_TARBALL, "r:gz") as archive:
        archive.extractall(REPO, filter="data")
    if not (REPO / "pyproject.toml").is_file():
        raise FileNotFoundError(f"pyproject.toml missing under {REPO}")


def install_checkout() -> None:
    """Install the extracted checkout into the kernel interpreter."""

    cmd = [sys.executable, "-m", "pip", "install", "-e", str(REPO)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    extract_checkout()
    install_checkout()
    runner = REPO / "scripts" / "run.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Missing runner: {runner}")
    print("Running:", sys.executable, runner)
    subprocess.check_call([sys.executable, str(runner)])


if __name__ == "__main__":
    main()
