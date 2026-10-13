"""Paste this file into one Google Colab cell to start the FastAPI worker.

This bootstrap only prepares the runtime: Drive, code, venv, Tailscale, and the
server process. It does not run pipeline stages. Stages run later via HTTP from
the local orchestrator.

Colab userdata secrets required:
  GITHUB         GitHub token for a private clone
  TAILSCALE_KEY  Tailscale auth key

Edit BRANCH and REPO_URL below before pasting.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

# Session constants. Change BRANCH when deploying a different experiment branch.
BRANCH = "dario"
REPO_URL = "https://github.com/liuklab/open-enricher.git"
REPO_DIR = Path("/content/open-enricher")
VENV_DIR = REPO_DIR / "venv"
REQ_STAMP = VENV_DIR / ".requirements.sha256"
CACHE_ROOT = Path("/content/drive/MyDrive/open_enricher/cache")


def _run(cmd: list[str] | str, *, shell: bool = False) -> None:
    print("Running:", cmd if isinstance(cmd, str) else " ".join(cmd))
    subprocess.run(cmd, shell=shell, check=True)


def _requirements_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _install_dependencies() -> None:
    pip = str(VENV_DIR / "bin" / "pip")
    print("Installing dependencies into the virtual environment")
    _run([pip, "install", "--upgrade", "pip", "setuptools", "wheel", "-q"])
    _run([pip, "install", "-r", "requirements.txt", "-q"])
    _run([pip, "install", "-e", ".[remote]", "-q"])
    REQ_STAMP.write_text(_requirements_digest(REPO_DIR / "requirements.txt"), encoding="utf-8")


def main() -> None:
    from google.colab import drive, userdata

    print("Mounting Google Drive")
    drive.mount("/content/drive")

    gh_token = userdata.get("GITHUB")
    if not gh_token:
        raise SystemExit("Colab userdata secret GITHUB is missing")

    auth_url = REPO_URL.replace("https://", f"https://{gh_token}@")
    if not REPO_DIR.is_dir():
        print(f"Cloning branch {BRANCH}")
        _run(["git", "clone", "-b", BRANCH, auth_url, str(REPO_DIR)])
    else:
        print(f"Updating branch {BRANCH}")
        _run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH])
        _run(["git", "-C", str(REPO_DIR), "checkout", BRANCH])
        _run(["git", "-C", str(REPO_DIR), "pull", "origin", BRANCH])

    os.chdir(REPO_DIR)

    req_file = REPO_DIR / "requirements.txt"
    req_digest = _requirements_digest(req_file)
    if not VENV_DIR.is_dir():
        print("Creating virtual environment")
        _run(["pip", "install", "virtualenv", "-q"])
        _run(["virtualenv", str(VENV_DIR), "-q"])
        _install_dependencies()
    elif not REQ_STAMP.is_file() or REQ_STAMP.read_text(encoding="utf-8").strip() != req_digest:
        print("requirements.txt changed; reinstalling dependencies")
        _install_dependencies()
    else:
        print("Reusing existing virtual environment")
        # Editable install so pulled stage code is importable without a full reinstall.
        pip = str(VENV_DIR / "bin" / "pip")
        _run([pip, "install", "-e", ".[remote]", "-q"])

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(CACHE_ROOT / "huggingface")
    os.environ["TORCH_HOME"] = str(CACHE_ROOT / "torch")
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    ts_key = userdata.get("TAILSCALE_KEY")
    if not ts_key:
        raise SystemExit("Colab userdata secret TAILSCALE_KEY is missing")
    os.environ["TAILSCALE_KEY"] = ts_key

    python = str(VENV_DIR / "bin" / "python")
    # Import after the editable install so the venv sees the package.
    connect_code = (
        "from open_enricher.remote.tailscale import connect; "
        "import os; "
        "print(connect(os.environ['TAILSCALE_KEY']))"
    )
    print("Connecting Tailscale")
    _run([python, "-c", connect_code])

    print("Starting FastAPI worker in the foreground (logs appear below)")
    _run([python, "-m", "open_enricher.remote.server"])


if __name__ == "__main__":
    main()
