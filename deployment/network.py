"""Networking strategies for transferring files and executing remote commands."""

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

REMOTE_TARBALL_PATH = "/content/open-enricher.tgz"
REMOTE_ENV_PATH = "/content/.env"


def _session_exists(session: str) -> bool:
    """Return True when `colab status` reports the named session."""
    result = subprocess.run(
        ["colab", "status", "-s", session],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0 or "not found" in combined:
        return False
    return True


def ensure_colab_session(session: str, gpu: str) -> None:
    """Start a Colab GPU session and mount Drive when that session is missing.

    Matches the old colab_cli bootstrap: reuse the session if `colab status`
    already sees it, otherwise `colab new` plus `colab drivemount`.
    """
    if shutil.which("colab") is None:
        print("[ERROR] colab CLI not found on PATH.")
        sys.exit(1)
    if _session_exists(session):
        print(f"[INFO] Reusing Colab session '{session}'")
        return
    print(f"[INFO] Session '{session}' not found. Starting it on GPU {gpu}...")
    _run_cmd(["colab", "new", "-s", session, "--gpu", gpu])
    print("[INFO] Mounting Google Drive (a browser OAuth prompt may open once)")
    _run_cmd(["colab", "drivemount", "-s", session])


def _run_cmd(cmd: list[str], input_data: str | None = None) -> None:
    print(f"[RUN] Running: {' '.join(cmd)}")
    try:
        stdin_bytes = input_data.encode("utf-8") if input_data else None
        subprocess.run(cmd, input=stdin_bytes, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}.")
        sys.exit(1)


class DeployStrategy(Protocol):
    """Interface for all deployment network strategies (Strategy Pattern)."""
    def deploy(self, tarball_path: Path, env_path: Path, target: str, remote_script: str) -> None:
        ...


class TailscaleStrategy:
    """Deploys over a direct Tailscale SSH connection."""
    def deploy(self, tarball_path: Path, env_path: Path, target: str, remote_script: str) -> None:
        print(f"[INFO] Deploying via Tailscale SSH to {target}...")
        remote_host = f"root@{target}"

        _run_cmd(["scp", "-o", "StrictHostKeyChecking=no", str(tarball_path), f"{remote_host}:{REMOTE_TARBALL_PATH}"])

        if env_path.exists():
            print("[INFO] Uploading the local .env file to the worker...")
            _run_cmd(["scp", "-o", "StrictHostKeyChecking=no", str(env_path), f"{remote_host}:{REMOTE_ENV_PATH}"])
        else:
            print("[WARNING] No local .env file found. Deploying without extra secrets.")

        # Start the remote worker.
        _run_cmd(["ssh", "-o", "StrictHostKeyChecking=no", remote_host, "bash", "-s"], input_data=remote_script)


class ColabCliStrategy:
    """Deploys utilizing Google Colab CLI tunneling."""

    def __init__(self, gpu: str = "T4") -> None:
        self.gpu = gpu

    def deploy(self, tarball_path: Path, env_path: Path, target: str, remote_script: str) -> None:
        print(f"[INFO] Deploying via Colab CLI to session '{target}'...")
        ensure_colab_session(target, self.gpu)

        _run_cmd(["colab", "upload", "-s", target, str(tarball_path), REMOTE_TARBALL_PATH])

        if env_path.exists():
            print("[INFO] Uploading the local .env file to the worker...")
            _run_cmd(["colab", "upload", "-s", target, str(env_path), REMOTE_ENV_PATH])
        else:
            print("[WARNING] No local .env file found. Deploying without extra secrets.")

        # `colab exec` runs Python in the kernel and does not take a shell command.
        # The default 30 second timeout is too short for the pip install in launch.sh.
        script_path = tarball_path.parent / "launch_tmp.sh"
        runner_path = tarball_path.parent / "launch_runner.py"
        try:
            script_path.write_text(remote_script, encoding="utf-8", newline="\n")
            runner_path.write_text(
                "import subprocess\n"
                "process = subprocess.Popen(\n"
                "    ['bash', '/content/launch.sh'],\n"
                "    stdout=subprocess.PIPE,\n"
                "    stderr=subprocess.STDOUT,\n"
                "    text=True,\n"
                ")\n"
                "assert process.stdout is not None\n"
                "for line in process.stdout:\n"
                "    print(line, end='', flush=True)\n"
                "returncode = process.wait()\n"
                "if returncode != 0:\n"
                "    raise RuntimeError(f'launch.sh failed with exit code {returncode}')\n",
                encoding="utf-8",
            )
            _run_cmd(["colab", "upload", "-s", target, str(script_path), "/content/launch.sh"])
            _run_cmd(
                [
                    "colab",
                    "exec",
                    "-s",
                    target,
                    "--timeout",
                    "3600",
                    "-f",
                    str(runner_path),
                ]
            )
        finally:
            if script_path.exists():
                script_path.unlink()
            if runner_path.exists():
                runner_path.unlink()