#!/usr/bin/env python3
"""Provision a Colab GPU session, upload this checkout as a tarball, run metadata.

Every runtime flag must be passed on the command line. Missing values are errors.

Examples from the repository root:

  python colab_cli/bootstrap.py \\
    --session enricher \\
    --gpu T4 \\
    --auth adc \\
    --timeout 7200 \\
    --install-timeout 600

  # Later evaluation iterations (session already up):
  python colab_cli/bootstrap.py \\
    --session enricher \\
    --gpu T4 \\
    --auth adc \\
    --timeout 7200 \\
    --install-timeout 600 \\
    --skip-setup

Then stop the session when you are done:

  colab --auth=adc stop -s enricher
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
OUTPUT_REMOTE_TARBALL = "/content/open-enricher/last_run.tgz"
PACK_FILES = ("pyproject.toml", "requirements.txt")
INSTALL_LOCAL = Path(__file__).resolve().parent / "install_local.py"


def parse_args() -> argparse.Namespace:
    """Parse required Colab flags. Fail if any flag is omitted."""

    parser = argparse.ArgumentParser(
        description="Upload this checkout to a Colab GPU session and run metadata inference.",
    )
    parser.add_argument("--session", required=True, help="Named Colab CLI session.")
    parser.add_argument("--gpu", required=True, help="GPU type, for example T4 (Colab free), L4, or A100.")
    parser.add_argument("--auth", required=True, help="Colab CLI auth method, for example adc.")
    parser.add_argument("--timeout", required=True, type=int, help="Seconds for inference exec.")
    parser.add_argument(
        "--install-timeout",
        required=True,
        type=int,
        help="Seconds for remote pip install.",
    )
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Reuse an existing session: skip new, drivemount, and requirements install.",
    )
    return parser.parse_args()


def run_colab(auth: str, *args: str) -> None:
    """Run one `colab` CLI command and raise if it exits non-zero."""

    cmd = ["colab", f"--auth={auth}", *args]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"colab failed ({result.returncode}): {' '.join(cmd)}")


def session_exists(auth: str, session: str) -> bool:
    """Return True only when the named session is in the local Colab CLI store."""

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
    """Pack src, project metadata, and .env (if present) into a gzip tarball."""

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
    """Create or reuse the session, mount Drive, install, upload, and run metadata."""

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

        print("Installing Python dependencies on the VM")
        run_colab(args.auth, "install", "-s", args.session, "-r", str(ROOT / "requirements.txt"))

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "open-enricher.tgz"
        print("Packing local checkout (unpushed edits are included)")
        build_tarball(tarball)

        print(f"Uploading {tarball.name} to {REMOTE_TARBALL}")
        run_colab(args.auth, "upload", "-s", args.session, str(tarball), REMOTE_TARBALL)

        print("Extracting tarball and installing the package")
        run_colab(
            args.auth,
            "exec",
            "-s", args.session,
            "--timeout", str(args.install_timeout),
            "-f", str(INSTALL_LOCAL),
        )

        print("Running metadata inference")
        run_colab(
            args.auth,
            "exec",
            "-s", args.session,
            "--timeout", str(args.timeout),
            "-f", str(ROOT / "scripts" / "run.py"),
        )

        local_tarball = ROOT / "outputs" / "last_run.tgz"
        local_tarball.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading output tarball")
        run_colab(
            args.auth,
            "download",
            "-s", args.session,
            OUTPUT_REMOTE_TARBALL,
            str(local_tarball),
        )

        print("Extracting output tarball")
        with tarfile.open(local_tarball, "r:gz") as archive:
            archive.extractall(ROOT / "outputs", filter="data")

    print(
        f"Done. Session {args.session} is still running. "
        f"Stop it with: colab --auth={args.auth} stop -s {args.session}"
    )


if __name__ == "__main__":
    main()
