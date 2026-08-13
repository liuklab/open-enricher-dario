#!/usr/bin/env python3
"""Provision a Colab GPU session, upload this checkout as a tarball, run inference.

Usage from the repository root:
  python scripts/bootstrap_colab.py
  python scripts/bootstrap_colab.py embeddings
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_project_config() -> dict:
    """Load settings.py by path so bootstrap does not import the full package."""

    import importlib.util

    settings_path = ROOT / "src" / "open_enricher" / "settings.py"
    spec = importlib.util.spec_from_file_location("open_enricher_settings", settings_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load {settings_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_project_config(ROOT / "config.toml")


REMOTE_TARBALL = "/content/open-enricher.tgz"
MODE_LINE = 'MODE = "metadata"'
PACK_FILES = ("pyproject.toml", "requirements.txt", "config.toml")


def colab_section() -> dict:
    return load_project_config().get("colab", {})


def run_colab(auth: str, *args: str) -> None:
    cmd = ["colab", f"--auth={auth}", *args]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"colab failed ({result.returncode}): {' '.join(cmd)}")


def session_exists(auth: str, session: str) -> bool:
    """Return True only when the named session is in the local Colab CLI store.

    `colab status -s NAME` exits 0 even when the session is missing (it prints
    "not found"), so exit code alone is not enough.
    """

    result = subprocess.run(
        ["colab", f"--auth={auth}", "status", "-s", session],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    combined = f"{result.stdout}\n{result.stderr}".lower()
    return "not found" not in combined


def build_tarball(dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "open-enricher"
        staging.mkdir()
        shutil.copytree(ROOT / "src", staging / "src")
        for name in PACK_FILES:
            shutil.copy2(ROOT / name, staging / name)
        env_file = ROOT / ".env"
        if env_file.is_file():
            shutil.copy2(env_file, staging / ".env")
        with tarfile.open(dest, "w:gz") as archive:
            archive.add(staging, arcname=".")


def write_runner(mode: str, dest: Path) -> None:
    source = (ROOT / "scripts" / "run.py").read_text(encoding="utf-8")
    if MODE_LINE not in source:
        raise RuntimeError(f"scripts/run.py must contain {MODE_LINE}")
    dest.write_text(source.replace(MODE_LINE, f'MODE = "{mode}"', 1), encoding="utf-8")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "metadata"
    if mode not in {"metadata", "embeddings"}:
        print("Usage: python scripts/bootstrap_colab.py [metadata|embeddings]", file=sys.stderr)
        raise SystemExit(1)

    if shutil.which("colab") is None:
        print("colab CLI not found on PATH. Install with: uv tool install google-colab-cli", file=sys.stderr)
        raise SystemExit(1)

    colab_cfg = colab_section()
    session = str(colab_cfg.get("session") or "enricher")
    gpu = str(colab_cfg.get("gpu") or "A100")
    auth = str(colab_cfg.get("auth") or "adc")
    timeout = str(colab_cfg.get("timeout") or 7200)
    install_timeout = str(colab_cfg.get("install_timeout") or 600)

    if session_exists(auth, session):
        print(f"Reusing session {session}")
    else:
        print(f"Starting session {session} on GPU {gpu} (auth={auth})")
        run_colab(auth, "new", "-s", session, "--gpu", gpu)

    print("Mounting Google Drive for videos (browser OAuth may open once)")
    run_colab(auth, "drivemount", "-s", session)

    print("Installing Python dependencies on the VM")
    run_colab(auth, "install", "-s", session, "-r", str(ROOT / "requirements.txt"))

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "open-enricher.tgz"
        runner = Path(tmp) / "run.py"
        print("Packing local checkout (unpushed edits are included)")
        build_tarball(tarball)
        write_runner(mode, runner)

        print(f"Uploading {tarball.name} to {REMOTE_TARBALL}")
        run_colab(auth, "upload", "-s", session, str(tarball), REMOTE_TARBALL)

        print("Extracting tarball and installing the package")
        run_colab(
            auth,
            "exec",
            "-s",
            session,
            "--timeout",
            install_timeout,
            "-f",
            str(ROOT / "scripts" / "install_local.py"),
        )

        print(f"Running {mode} inference")
        run_colab(auth, "exec", "-s", session, "--timeout", timeout, "-f", str(runner))

    print(f"Done. Session {session} is still running. Stop it with: colab --auth={auth} stop -s {session}")


if __name__ == "__main__":
    main()
