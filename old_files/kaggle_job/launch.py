#!/usr/bin/env python3
"""Push this checkout to Kaggle and run textual metadata inference.

The kernel cannot see this PC. Each launch uploads a private code dataset
(``<user>/open-enricher-code``) and attaches the video dataset named in
``dataset/dataset-metadata.json`` (``darionapolitano123/open-enricher-dataset``).
Videos mount at ``/kaggle/input/open-enricher-dataset``. Results land in
``kaggle_job/output/``.

``.env`` is packed into the private code dataset so the kernel can read
``HF_TOKEN``. It is not written under ``/kaggle/working``.

Requires ``kaggle`` on PATH and a prior ``kaggle auth login``.

Run from anywhere, after ``kaggle datasets create -p dataset`` has finished:

  python kaggle_job/launch.py

Defaults are a T4 GPU, internet on (the model is downloaded from Hugging Face),
and a 9 hour wait. A normal account's GPU is ``NvidiaTeslaT4`` (2× T4, 16 GB).
``device_map`` is ``cuda``, so inference uses one of those cards.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
JOB_FILE = HERE / "job.py"
VIDEO_METADATA = ROOT / "dataset" / "dataset-metadata.json"
OUTPUT_DIR = HERE / "output"
CODE_SLUG = "open-enricher-code"
CODE_TITLE = "open-enricher-code"
STATUS_RE = re.compile(r'has status "([^"]+)"')
TERMINAL_OK = {"COMPLETE"}
TERMINAL_FAIL = {"ERROR", "CANCEL_ACKNOWLEDGED"}
DRIVE_DATA_LINE = 'DRIVE_DATA = "/content/drive/MyDrive/open_enricher/data"'
LOCAL_OUTPUT_LINE = 'LOCAL_OUTPUT = "/content/open-enricher/outputs"'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-enricher metadata inference on a private Kaggle kernel.",
    )
    parser.add_argument(
        "--slug",
        default="open-enricher-job",
        help="Kernel slug under your Kaggle username. Reused on the next push.",
    )
    parser.add_argument(
        "--user",
        default=None,
        help="Kaggle username. Defaults to the account saved by kaggle auth login.",
    )
    parser.add_argument(
        "--accelerator",
        default="NvidiaTeslaT4",
        help="Kaggle machine_shape (default NvidiaTeslaT4).",
    )
    parser.add_argument(
        "--internet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let the kernel reach the network (default on, so it can download the model).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=32400,
        help="Maximum kernel runtime in seconds (default 32400, 9 hours).",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=32400,
        help="Seconds to poll locally before leaving the kernel running (default 32400).",
    )
    parser.add_argument(
        "--poll",
        type=int,
        default=20,
        help="Seconds between kernel status checks (default 20).",
    )
    return parser.parse_args()


def kaggle_username(explicit: str | None) -> str:
    """Resolve the account slug without printing tokens."""

    if explicit:
        return explicit
    credentials = Path.home() / ".kaggle" / "credentials.json"
    legacy = Path.home() / ".kaggle" / "kaggle.json"
    for path in (credentials, legacy):
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        username = data.get("username")
        if isinstance(username, str) and username:
            return username
    raise SystemExit(
        "Kaggle username not found. Run `kaggle auth login` or pass --user."
    )


def video_dataset_ref() -> tuple[str, str]:
    """Return ``(owner/slug, slug)`` from dataset/dataset-metadata.json."""

    if not VIDEO_METADATA.is_file():
        raise SystemExit(f"Missing {VIDEO_METADATA}. Create the video dataset first.")
    data = json.loads(VIDEO_METADATA.read_text(encoding="utf-8"))
    ref = data.get("id")
    if not isinstance(ref, str) or ref.count("/") != 1:
        raise SystemExit(f"{VIDEO_METADATA} id must look like owner/slug.")
    _owner, slug = ref.split("/", 1)
    if not slug:
        raise SystemExit(f"{VIDEO_METADATA} has an empty dataset slug.")
    return ref, slug


def accelerator_fields(accelerator: str) -> dict[str, str | bool]:
    """Map a machine_shape id onto the GPU and TPU flags Kaggle expects."""

    if accelerator.lower().startswith("tpu"):
        return {"enable_gpu": False, "enable_tpu": True, "machine_shape": accelerator}
    return {"enable_gpu": True, "enable_tpu": False, "machine_shape": accelerator}


def run_kaggle(*args: str, echo: bool = True) -> str:
    """Run one kaggle CLI command. Raise if it exits non-zero or reports a push error."""

    cmd = ["kaggle", *args]
    if echo:
        print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if echo and result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if echo and result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    combined = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or "Kernel push error" in combined:
        if not echo:
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
        raise RuntimeError(f"kaggle failed ({result.returncode}): {' '.join(cmd)}")
    return combined


def dataset_status(ref: str) -> str | None:
    """Return the dataset status, or None when Kaggle has no such dataset."""

    result = subprocess.run(
        ["kaggle", "datasets", "status", ref],
        check=False,
        text=True,
        capture_output=True,
    )
    combined = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        lowered = combined.lower()
        if "404" in lowered or "403" in lowered or "not found" in lowered or "forbidden" in lowered:
            return None
        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
        raise RuntimeError(f"kaggle datasets status failed ({result.returncode}): {ref}")
    status = result.stdout.strip().splitlines()[-1].strip().lower()
    if not status:
        raise RuntimeError(f"Empty dataset status for {ref}")
    return status


def wait_until_ready(ref: str, wait_seconds: int) -> None:
    """Poll a dataset until Kaggle reports it ready."""

    deadline = time.monotonic() + wait_seconds
    seen = None
    while True:
        status = dataset_status(ref)
        if status is None:
            raise SystemExit(f"Dataset {ref} does not exist.")
        if status != seen:
            print(f"{ref} is {status}")
            seen = status
        if status == "ready":
            return
        if status in {"error", "failed", "deleted"}:
            raise SystemExit(f"Dataset {ref} is {status}.")
        if time.monotonic() >= deadline:
            raise SystemExit(f"Dataset {ref} still {status} after {wait_seconds}s.")
        time.sleep(10)


def build_code_tarball(dest: Path, video_slug: str) -> None:
    """Pack the checkout with Kaggle paths for metadata inference."""

    env_file = ROOT / ".env"
    if not env_file.is_file():
        raise SystemExit("Missing .env. The kernel needs HF_TOKEN from that file.")
    run_source = (ROOT / "scripts" / "run.py").read_text(encoding="utf-8")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "open-enricher"
        shutil.copytree(
            ROOT / "src",
            staging / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copy2(ROOT / "pyproject.toml", staging / "pyproject.toml")
        shutil.copy2(ROOT / "requirements.txt", staging / "requirements.txt")
        shutil.copy2(env_file, staging / ".env")
        script_dir = staging / "scripts"
        script_dir.mkdir()
        (script_dir / "run.py").write_text(run_source, encoding="utf-8")

        config_path = staging / "src" / "open_enricher" / "config.py"
        config_text = config_path.read_text(encoding="utf-8")
        if DRIVE_DATA_LINE not in config_text or LOCAL_OUTPUT_LINE not in config_text:
            raise RuntimeError(
                "config.py path constants changed. Update DRIVE_DATA_LINE and "
                "LOCAL_OUTPUT_LINE in kaggle_job/launch.py."
            )
        config_path.write_text(
            config_text.replace(
                DRIVE_DATA_LINE,
                f'DRIVE_DATA = "/kaggle/input/{video_slug}"',
                1,
            ).replace(
                LOCAL_OUTPUT_LINE,
                'LOCAL_OUTPUT = "/kaggle/working/outputs"',
                1,
            ),
            encoding="utf-8",
        )
        with tarfile.open(dest, "w:gz") as archive:
            archive.add(staging, arcname=".")


def publish_code_dataset(folder: Path, ref: str, wait_seconds: int) -> None:
    """Create the code dataset, or upload a new version when it already exists."""

    if dataset_status(ref) is None:
        print(f"Creating code dataset {ref}")
        run_kaggle("datasets", "create", "-p", str(folder))
    else:
        print(f"Updating code dataset {ref}")
        run_kaggle("datasets", "version", "-p", str(folder), "-m", "inference checkout")
    wait_until_ready(ref, wait_seconds)


def kernel_status(kernel: str) -> str:
    """Return the status name, for example COMPLETE or RUNNING."""

    text = run_kaggle("kernels", "status", kernel, echo=False)
    match = STATUS_RE.search(text)
    if match is None:
        raise RuntimeError(f"Unrecognized status output:\n{text}")
    return match.group(1).rsplit(".", 1)[-1]


def wait_for_kernel(kernel: str, wait_seconds: int, poll_seconds: int) -> None:
    """Poll until the run completes, fails, or the local wait budget runs out."""

    deadline = time.monotonic() + wait_seconds
    seen = None
    while True:
        status = kernel_status(kernel)
        if status != seen:
            print(f"{kernel} is {status}")
            seen = status
        if status in TERMINAL_OK:
            return
        if status in TERMINAL_FAIL:
            print(f"Kernel logs for {kernel}:")
            subprocess.run(["kaggle", "kernels", "logs", kernel], check=False)
            raise SystemExit(f"Kernel {kernel} finished with status {status}")
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"Still {status} after {wait_seconds}s. "
                f"The kernel keeps running. Check it with: kaggle kernels status {kernel}"
            )
        time.sleep(poll_seconds)


def main() -> None:
    args = parse_args()
    if shutil.which("kaggle") is None:
        raise SystemExit("kaggle CLI not found on PATH. Install with: pipx install kaggle")
    if not JOB_FILE.is_file():
        raise SystemExit(f"Missing job file: {JOB_FILE}")
    if args.poll < 1:
        raise SystemExit("--poll must be at least 1 second.")

    username = kaggle_username(args.user)
    video_ref, video_slug = video_dataset_ref()
    code_ref = f"{username}/{CODE_SLUG}"
    kernel = f"{username}/{args.slug}"

    print(f"Waiting for video dataset {video_ref}")
    wait_until_ready(video_ref, args.wait)

    with tempfile.TemporaryDirectory() as tmp:
        code_dir = Path(tmp) / "code"
        code_dir.mkdir()
        (code_dir / "dataset-metadata.json").write_text(
            json.dumps(
                {
                    "title": CODE_TITLE,
                    "id": code_ref,
                    "licenses": [{"name": "CC0-1.0"}],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Packing checkout for /kaggle/input/{video_slug}")
        build_code_tarball(code_dir / "open-enricher.tgz", video_slug)
        publish_code_dataset(code_dir, code_ref, args.wait)

        push_dir = Path(tmp) / "kernel"
        push_dir.mkdir()
        shutil.copy2(JOB_FILE, push_dir / JOB_FILE.name)
        metadata = {
            "id": kernel,
            "title": args.slug,
            "code_file": JOB_FILE.name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_internet": args.internet,
            "dataset_sources": [video_ref, code_ref],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        metadata.update(accelerator_fields(args.accelerator))
        (push_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Pushing {kernel} on {args.accelerator}")
        run_kaggle(
            "kernels",
            "push",
            "-p",
            str(push_dir),
            "--timeout",
            str(args.timeout),
        )

    wait_for_kernel(kernel, args.wait, args.poll)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading output to {OUTPUT_DIR}")
    run_kaggle("kernels", "output", kernel, "-p", str(OUTPUT_DIR), "-o")
    print(f"Wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
