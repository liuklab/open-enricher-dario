#!/usr/bin/env python3
"""Deploy the FastAPI pipeline worker to a Colab GPU session via Colab CLI.

Packs the local checkout (including unpushed edits), uploads it, installs the
remote extra, connects Tailscale when a key is present, and starts the API in
the background. Later runs with --skip-setup only refresh the code and restart.

Examples from the repository root:

  python colab_cli/deploy_server.py \\
    --session enricher-api \\
    --gpu T4 \\
    --auth adc \\
    --install-timeout 600

  python colab_cli/deploy_server.py \\
    --session enricher-api \\
    --gpu T4 \\
    --auth adc \\
    --install-timeout 600 \\
    --skip-setup
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REMOTE_TARBALL = "/content/open-enricher.tgz"
PACK_FILES = ("pyproject.toml", "requirements.txt")
INSTALL_SERVER = Path(__file__).resolve().parent / "install_server.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload this checkout and start the remote FastAPI worker on Colab.",
    )
    parser.add_argument("--session", required=True, help="Named Colab CLI session.")
    parser.add_argument("--gpu", required=True, help="GPU type, for example T4, L4, or A100.")
    parser.add_argument("--auth", required=True, help="Colab CLI auth method, for example adc.")
    parser.add_argument(
        "--install-timeout",
        required=True,
        type=int,
        help="Seconds for remote extract, pip install, and server start.",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Reuse an existing session: skip new, drivemount, and requirements install.",
    )
    return parser.parse_args()


def run_colab(auth: str, *args: str) -> None:
    cmd = ["colab", f"--auth={auth}", *args]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"colab failed ({result.returncode}): {' '.join(cmd)}")


def session_exists(auth: str, session: str) -> bool:
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


def main() -> None:
    args = parse_args()

    if shutil.which("colab") is None:
        raise SystemExit("colab CLI not found on PATH. Install with: uv tool install google-colab-cli")

    exists = session_exists(args.auth, args.session)
    if args.skip_setup:
        if not exists:
            raise SystemExit(
                f"Session {args.session} does not exist; cannot --skip-setup. "
                "Run without --skip-setup first."
            )
        print(f"Skipping setup; reusing session {args.session}")
    elif exists:
        print(f"Reusing session {args.session}")
    else:
        print(f"Starting session {args.session} on GPU {args.gpu} (auth={args.auth})")
        run_colab(args.auth, "new", "-s", args.session, "--gpu", args.gpu)

    if not args.skip_setup:
        print("Mounting Google Drive for videos (browser OAuth may open once)")
        run_colab(args.auth, "drivemount", "-s", args.session)

        print("Installing base Python dependencies on the VM")
        run_colab(args.auth, "install", "-s", args.session, "-r", str(ROOT / "requirements.txt"))

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "open-enricher.tgz"
        print("Packing local checkout (unpushed edits are included)")
        build_tarball(tarball)

        print(f"Uploading {tarball.name} to {REMOTE_TARBALL}")
        run_colab(args.auth, "upload", "-s", args.session, str(tarball), REMOTE_TARBALL)

        print("Extracting tarball, installing remote extra, starting the API")
        run_colab(
            args.auth,
            "exec",
            "-s", args.session,
            "--timeout", str(args.install_timeout),
            "-f", str(INSTALL_SERVER),
        )

    print(
        f"Done. Session {args.session} is running the FastAPI worker. "
        f"Dispatch stages with: python -m open_enricher.pipeline "
        f"--host <tailscale-ip> --port 8000 --video <file.mkv>"
    )


if __name__ == "__main__":
    main()
