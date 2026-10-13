#!/usr/bin/env python3
"""Copy this checkout to a Tailscale host with scp and restart the FastAPI worker.

The host must already be on the tailnet with SSH access. This path does not
install Tailscale and does not use Colab userdata.

Examples from the repository root:

  python remote_host/deploy.py \\
    --host 100.x.y.z \\
    --user ubuntu \\
    --remote-dir /home/ubuntu/open-enricher

  python remote_host/deploy.py \\
    --host worker \\
    --user ubuntu \\
    --remote-dir /home/ubuntu/open-enricher \\
    --python /home/ubuntu/open-enricher/.venv/bin/python
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK_FILES = ("pyproject.toml", "requirements.txt")
PID_NAME = "open-enricher-server.pid"
LOG_NAME = "open-enricher-server.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy the FastAPI worker to a remote host over Tailscale with scp/ssh.",
    )
    parser.add_argument("--host", required=True, help="Tailscale IPv4 or MagicDNS hostname.")
    parser.add_argument("--user", required=True, help="SSH user on the remote host.")
    parser.add_argument(
        "--remote-dir",
        required=True,
        help="Absolute directory on the host where the checkout is extracted.",
    )
    parser.add_argument(
        "--python",
        default="python3",
        help="Python interpreter on the host (default python3).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Worker HTTP port (default 8000).",
    )
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=22,
        help="SSH port (default 22).",
    )
    return parser.parse_args()


def build_tarball(dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "open-enricher"
        staging.mkdir()
        shutil.copytree(
            ROOT / "src",
            staging / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for name in PACK_FILES:
            shutil.copy2(ROOT / name, staging / name)
        env_file = ROOT / ".env"
        if env_file.is_file():
            shutil.copy2(env_file, staging / ".env")
        with tarfile.open(dest, "w:gz") as archive:
            archive.add(staging, arcname=".")


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    if shutil.which("scp") is None or shutil.which("ssh") is None:
        raise SystemExit("scp and ssh must be on PATH.")

    remote = f"{args.user}@{args.host}"
    remote_dir = args.remote_dir.rstrip("/")
    remote_tarball = f"{remote_dir}.tgz"
    pid_file = f"{remote_dir}/{PID_NAME}"
    log_file = f"{remote_dir}/{LOG_NAME}"

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "open-enricher.tgz"
        print("Packing local checkout (unpushed edits are included)")
        build_tarball(tarball)

        run(
            [
                "scp",
                "-P",
                str(args.ssh_port),
                str(tarball),
                f"{remote}:{remote_tarball}",
            ]
        )

    python = shlex.quote(args.python)
    remote_dir_q = shlex.quote(remote_dir)
    remote_tarball_q = shlex.quote(remote_tarball)
    pid_q = shlex.quote(pid_file)
    log_q = shlex.quote(log_file)
    port = int(args.port)

    remote_script = f"""
set -e
mkdir -p {remote_dir_q}
if [ -f {pid_q} ]; then
  kill "$(cat {pid_q})" 2>/dev/null || true
  rm -f {pid_q}
fi
rm -rf {remote_dir_q}
mkdir -p {remote_dir_q}
tar -xzf {remote_tarball_q} -C {remote_dir_q}
{python} -m pip install -e {remote_dir_q}'[remote]'
cd {remote_dir_q}
nohup {python} -m open_enricher.remote.server > {log_q} 2>&1 &
echo $! > {pid_q}
echo "Started server pid $(cat {pid_q}) on port {port}"
"""

    print("Running remote extract, install, and server restart")
    result = subprocess.run(
        ["ssh", "-p", str(args.ssh_port), remote, "bash", "-s"],
        input=remote_script,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Remote deploy failed with exit code {result.returncode}")

    print(
        f"Done. Dispatch stages with: python -m open_enricher.pipeline "
        f"--host {args.host} --port {port} --video <file.mkv>"
    )


if __name__ == "__main__":
    main()
