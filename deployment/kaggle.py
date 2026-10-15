"""Push the worker to a private Kaggle kernel and recover its tunnel URL.

The laptop flow matches Colab: ``deploy.py`` uploads the checkout and starts
the GPU worker, then the client calls the same stage API. Kaggle has no SSH
port forward, so the kernel opens an outbound tunnel and the client reads the
URL from the kernel log.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

CODE_SLUG = os.environ.get("OPEN_ENRICHER_CODE_SLUG", "open-enricher-code").strip() or "open-enricher-code"
CODE_TITLE = os.environ.get("OPEN_ENRICHER_CODE_TITLE", CODE_SLUG).strip() or CODE_SLUG
SESSION_DIR = Path(".session")
TUNNEL_RE = re.compile(r"OPEN_ENRICHER_TUNNEL_URL=(https://\S+)")
STATUS_RE = re.compile(r'has status "([^"]+)"')
FILE_ROW_RE = re.compile(
    r"^(?P<name>\S+)\s+\d+\s+(?P<created>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*$"
)
TERMINAL_FAIL = {"ERROR", "CANCEL_ACKNOWLEDGED"}


def kaggle_username(explicit: str | None = None) -> str:
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
    raise SystemExit("Kaggle username not found. Run `kaggle auth login` or pass --user.")


def video_dataset_ref(root: Path) -> str:
    """Return ``owner/slug`` from dataset/dataset-metadata.json."""
    metadata_path = root / "dataset" / "dataset-metadata.json"
    if not metadata_path.is_file():
        raise SystemExit(
            f"Missing {metadata_path}. Create the private video dataset before a Kaggle deploy."
        )
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    ref = data.get("id")
    if not isinstance(ref, str) or ref.count("/") != 1:
        raise SystemExit(f"{metadata_path} id must look like owner/slug.")
    return ref


def accelerator_name(gpu: str) -> str:
    """Map the shared --gpu flag onto a Kaggle machine shape."""
    if gpu.upper() == "T4":
        return "NvidiaTeslaT4"
    return gpu


def session_url_path(slug: str) -> Path:
    return SESSION_DIR / f"{slug}.url"


def _run_kaggle(args: list[str], *, echo: bool = True) -> str:
    if shutil.which("kaggle") is None:
        print("[ERROR] kaggle CLI not found on PATH.")
        sys.exit(1)
    command = ["kaggle", *args]
    if echo:
        print(f"[RUN] Running: {' '.join(command)}")
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if echo and result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if echo and result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    combined = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0 or "Kernel push error" in combined:
        raise RuntimeError(f"kaggle failed ({result.returncode}): {' '.join(command)}")
    return combined


def _dataset_status(ref: str) -> str | None:
    result = subprocess.run(
        ["kaggle", "datasets", "status", ref],
        check=False,
        text=True,
        capture_output=True,
    )
    combined = f"{result.stdout}\n{result.stderr}".lower()
    if result.returncode != 0:
        if any(token in combined for token in ("404", "403", "not found", "forbidden")):
            return None
        raise RuntimeError(f"kaggle datasets status failed ({result.returncode}): {ref}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[-1].lower() if lines else None


def _dataset_file_created(ref: str, filename: str) -> str | None:
    """Return the creation timestamp Kaggle lists for one file, if present."""
    text = _run_kaggle(["datasets", "files", ref], echo=False)
    for line in text.splitlines():
        match = FILE_ROW_RE.match(line.strip())
        if match and match.group("name") == filename:
            return match.group("created")
    return None


def _wait_until_ready(ref: str, wait_seconds: int) -> None:
    deadline = time.monotonic() + wait_seconds
    seen = None
    while True:
        status = _dataset_status(ref)
        if status is None:
            raise SystemExit(f"Dataset {ref} does not exist.")
        if status != seen:
            print(f"[INFO] Dataset {ref} is {status}")
            seen = status
        if status == "ready":
            return
        if status in {"error", "failed", "deleted"}:
            raise SystemExit(f"Dataset {ref} is {status}.")
        if time.monotonic() >= deadline:
            raise SystemExit(f"Dataset {ref} still {status} after {wait_seconds}s.")
        time.sleep(10)


def _kernel_status(kernel: str) -> str:
    text = _run_kaggle(["kernels", "status", kernel], echo=False)
    match = STATUS_RE.search(text)
    if match is None:
        raise RuntimeError(f"Unrecognized kernel status:\n{text}")
    return match.group(1).rsplit(".", 1)[-1]


def _wait_for_new_code_version(
    ref: str,
    previous_created: str | None,
    wait_seconds: int,
) -> None:
    """Wait until Kaggle is serving a dataset version that contains this upload.

    ``datasets status`` stays ``ready`` on the previous version while a new one
    is still processing. Pushing a kernel in that window mounts the old files,
    so ``/kaggle/input/open-enricher-code/launch.sh`` is missing.
    """
    deadline = time.monotonic() + wait_seconds
    seen = None
    while True:
        status = _dataset_status(ref)
        created = _dataset_file_created(ref, "launch.sh")
        marker = (status, created)
        if marker != seen:
            print(f"[INFO] Dataset {ref} is {status}; launch.sh created {created}")
            seen = marker
        fresh = created is not None and created != previous_created
        if status == "ready" and fresh:
            print(f"[INFO] Dataset {ref} is serving launch.sh from {created}")
            return
        if status in {"error", "failed", "deleted"}:
            raise SystemExit(f"Dataset {ref} is {status}.")
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"Dataset {ref} did not publish a new launch.sh "
                f"(status {status}, created {created}) after {wait_seconds}s."
            )
        time.sleep(10)


def publish_code_dataset(folder: Path, ref: str, wait_seconds: int = 600) -> None:
    """Create the private code dataset, or upload a new version."""
    previous_created = None
    if _dataset_status(ref) is None:
        print(f"[INFO] Creating code dataset {ref}")
        _run_kaggle(["datasets", "create", "-p", str(folder)])
    else:
        previous_created = _dataset_file_created(ref, "launch.sh")
        print(f"[INFO] Updating code dataset {ref}")
        _run_kaggle(["datasets", "version", "-p", str(folder), "-m", "worker checkout"])
    _wait_for_new_code_version(ref, previous_created, wait_seconds)


def wait_for_tunnel_url(kernel: str, slug: str, wait_seconds: int = 1200) -> str:
    """Poll kernel logs until the worker prints its tunnel URL."""
    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        status = _kernel_status(kernel)
        if status in TERMINAL_FAIL:
            _run_kaggle(["kernels", "logs", kernel])
            raise SystemExit(f"Kernel {kernel} finished with status {status}")
        logs = _run_kaggle(["kernels", "logs", kernel], echo=False)
        match = TUNNEL_RE.search(logs)
        if match:
            url = match.group(1).rstrip(".,)")
            path = session_url_path(slug)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(url + "\n", encoding="utf-8")
            print(f"[INFO] Kaggle worker is reachable at {url}")
            return url
        print(f"[INFO] Kernel {kernel} is {status}; waiting for the tunnel URL...")
        time.sleep(15)
    raise SystemExit(
        f"No tunnel URL from {kernel} after {wait_seconds}s. "
        f"Inspect it with: kaggle kernels logs {kernel}"
    )


class KaggleStrategy:
    """Upload the checkout and start the same worker on a Kaggle GPU kernel."""

    def __init__(self, gpu: str = "T4") -> None:
        self.gpu = gpu

    def deploy(self, tarball_path: Path, env_path: Path, target: str, remote_script: str) -> None:
        import tempfile

        if shutil.which("kaggle") is None:
            print("[ERROR] kaggle CLI not found on PATH.")
            sys.exit(1)
        if not env_path.is_file():
            print("[ERROR] Missing .env. The Kaggle worker needs HF_TOKEN from that file.")
            sys.exit(1)

        root = tarball_path.parent
        username = kaggle_username()
        video_ref = video_dataset_ref(root)
        code_ref = f"{username}/{CODE_SLUG}"
        kernel = f"{username}/{target}"
        accelerator = accelerator_name(self.gpu)
        worker = Path(__file__).resolve().parent / "kaggle_worker.py"

        print(f"[INFO] Deploying via Kaggle kernel '{kernel}' on {accelerator}...")
        _wait_until_ready(video_ref, 600)

        with tempfile.TemporaryDirectory() as tmp:
            code_dir = Path(tmp) / "code"
            code_dir.mkdir()
            (code_dir / "dataset-metadata.json").write_text(
                json.dumps(
                    {"title": CODE_TITLE, "id": code_ref, "licenses": [{"name": "CC0-1.0"}]},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            shutil.copy2(tarball_path, code_dir / "open-enricher.tgz")
            shutil.copy2(env_path, code_dir / ".env")
            (code_dir / "launch.sh").write_text(remote_script, encoding="utf-8", newline="\n")
            publish_code_dataset(code_dir, code_ref)

            push_dir = Path(tmp) / "kernel"
            push_dir.mkdir()
            video_slug = video_ref.split("/", 1)[1]
            worker_source = worker.read_text(encoding="utf-8").replace(
                'Path("/kaggle/input/open-enricher-dataset")',
                f'Path("/kaggle/input/{video_slug}")',
            )
            (push_dir / "kernel_main.py").write_text(worker_source, encoding="utf-8")
            metadata = {
                "id": kernel,
                "title": target,
                "code_file": "kernel_main.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_internet": True,
                "dataset_sources": [video_ref, code_ref],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
                "enable_gpu": True,
                "enable_tpu": False,
                "machine_shape": accelerator,
            }
            (push_dir / "kernel-metadata.json").write_text(
                json.dumps(metadata, indent=2) + "\n",
                encoding="utf-8",
            )
            print(f"[INFO] Pushing {kernel}. The kernel stays up so stages can be called.")
            _run_kaggle(
                ["kernels", "push", "-p", str(push_dir), "--timeout", "28800", "--accelerator", accelerator]
            )

        wait_for_tunnel_url(kernel, target)


def read_tunnel_url(slug: str, wait_seconds: int = 60) -> str:
    """Return a saved tunnel URL, or poll the kernel log briefly."""
    saved = session_url_path(slug)
    if saved.is_file():
        url = saved.read_text(encoding="utf-8").strip()
        if url.startswith("https://"):
            return url
    username = kaggle_username()
    return wait_for_tunnel_url(f"{username}/{slug}", slug, wait_seconds)
