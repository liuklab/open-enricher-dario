"""Extract the uploaded tarball, install with the remote extra, restart the API.

colab exec ships only this file. The archive is already at /content/open-enricher.tgz.
"""

from __future__ import annotations

import importlib
import os
import signal
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

TARBALL = Path("/content/open-enricher.tgz")
REPO = Path("/content/open-enricher")
PID_FILE = Path("/content/open-enricher-server.pid")
LOG_FILE = Path("/content/open-enricher-server.log")
CACHE_ROOT = Path("/content/drive/MyDrive/open_enricher/cache")


def extract_tarball() -> None:
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.mkdir(parents=True)
    with tarfile.open(TARBALL, "r:gz") as archive:
        if sys.version_info >= (3, 12):
            archive.extractall(REPO, filter="data")
        else:
            archive.extractall(REPO)


def stop_server() -> None:
    if not PID_FILE.is_file():
        return
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        PID_FILE.unlink(missing_ok=True)
        return
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(1)
    except ProcessLookupError:
        pass
    PID_FILE.unlink(missing_ok=True)
    print(f"Stopped previous server pid {pid}")


def start_server() -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HF_HOME"] = str(CACHE_ROOT / "huggingface")
    env["TORCH_HOME"] = str(CACHE_ROOT / "torch")
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(env["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    log_handle = LOG_FILE.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-m", "open_enricher.remote.server"],
        cwd=str(REPO),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(f"{proc.pid}\n", encoding="utf-8")
    print(f"Started server pid {proc.pid}; logs: {LOG_FILE}")


def maybe_connect_tailscale() -> None:
    """Connect when TS_AUTH_KEY or TAILSCALE_KEY is set and Tailscale is not up yet."""

    auth_key = os.environ.get("TS_AUTH_KEY") or os.environ.get("TAILSCALE_KEY")
    if not auth_key:
        env_file = REPO / ".env"
        if env_file.is_file():
            from dotenv import load_dotenv

            load_dotenv(env_file)
            auth_key = os.environ.get("TS_AUTH_KEY") or os.environ.get("TAILSCALE_KEY")
    if not auth_key:
        print("No Tailscale auth key in the environment or .env; skipping connect.")
        return

    try:
        current = subprocess.check_output(["tailscale", "ip", "-4"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        current = ""
    if current:
        print(f"Tailscale already connected: {current}")
        return

    from open_enricher.remote.tailscale import connect

    ip_address = connect(auth_key)
    print(f"Update the local orchestrator host to: {ip_address}")


def main() -> None:
    if not TARBALL.is_file():
        raise FileNotFoundError(f"Missing tarball: {TARBALL}")

    extract_tarball()
    if not (REPO / "pyproject.toml").is_file():
        raise FileNotFoundError(f"pyproject.toml missing under {REPO}")

    cmd = [sys.executable, "-m", "pip", "install", "-e", f"{REPO}[remote]"]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    src = REPO / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    for name in list(sys.modules):
        if name == "open_enricher" or name.startswith("open_enricher."):
            del sys.modules[name]
    importlib.invalidate_caches()
    import open_enricher  # noqa: F401

    print("Installed open-enricher[remote] from", REPO)
    maybe_connect_tailscale()
    stop_server()
    start_server()


if __name__ == "__main__":
    main()
