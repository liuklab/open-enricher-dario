"""In-process pipeline runner for a local GPU host.

Builds the same stage registry the old FastAPI worker used, then calls
``stage.run(video)`` in order without HTTP, tunnels, or remote deploy.
Audio and color outputs can be restored from a shared cache so sequential
experiment variants skip work that does not depend on ``pipeline_spec``.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

from open_enricher.storage import db

# Default stage order for a full local enricher pass.
# Stage classes are imported lazily inside ``build_stages`` so CLI helpers
# and unit tests can import this module without loading torch or whisperx.
PIPELINE_STAGES: tuple[str, ...] = ("audio", "color", "vlm", "llm", "judger")

# Stages whose outputs do not depend on the active pipeline_spec knobs.
CACHEABLE_STAGES: frozenset[str] = frozenset({"audio", "color"})

# Primary JSON artifact written by each stage for a given video filename.
STAGE_PRIMARY_ARTIFACT: dict[str, str] = {
    "audio": "{video}_audio.json",
    "color": "{video}_color.json",
    "vlm": "{video}_vlm.json",
    "llm": "{video}_llm.json",
    "judger": "{video}_judger.json",
}


class _TeeStream:
    """
    Mirror stdout or stderr into the experiment worker log.

    Used so a long local run leaves a durable log next to stage outputs
    without replacing the process-wide streams permanently unless requested.
    """

    def __init__(self, primary, log_path: Path) -> None:
        """
        Bind a primary stream and a log file path.

        Args:
            primary: Original ``sys.stdout`` or ``sys.stderr``.
            log_path: File that receives a copy of every write.
        """
        self._primary = primary
        self._log_path = log_path
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        """
        Write to the primary stream and append the same bytes to the log.

        Args:
            data: Text chunk from print or library logging helpers.

        Returns:
            Number of characters written to the primary stream.
        """
        with self._lock:
            self._primary.write(data)
            self._primary.flush()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as handle:
                handle.write(data)
        return len(data)

    def flush(self) -> None:
        """Flush the primary stream."""
        self._primary.flush()

    def fileno(self):
        """Expose the primary file descriptor for libraries that need it."""
        return self._primary.fileno()


def bootstrap_runtime(*, env_file: Path | None = None, tee_log: bool = True) -> None:
    """
    Load secrets, configure caches, and create the data lake folders.

    Args:
        env_file: Optional ``.env`` path. Defaults to ``<repo>/.env`` when
            present, otherwise relies on the process environment alone.
        tee_log: When True, mirror stdout and stderr into ``db.experiment_log``.

    Side effects:
        Sets Hugging Face and Torch cache environment variables, creates
        directories, and may replace ``sys.stdout`` / ``sys.stderr``.
    """
    if env_file is None:
        candidate = Path(__file__).resolve().parents[2] / ".env"
        if candidate.is_file():
            env_file = candidate
    if env_file is not None and env_file.is_file():
        load_dotenv(env_file)

    db.configure_environment()
    db.setup_database()

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token)
            print("[INFO] Hugging Face login succeeded.")
        except Exception as exc:  # noqa: BLE001 - login is best-effort
            print(f"[WARNING] Hugging Face login failed: {exc}")

    if tee_log:
        log_path = db.experiment_log
        sys.stdout = _TeeStream(sys.__stdout__, log_path)
        sys.stderr = _TeeStream(sys.__stderr__, log_path)
        print(f"[INFO] Worker log mirroring to {log_path}")


def build_stages(hf_token: str | None = None) -> dict[str, Any]:
    """
    Construct the in-process stage registry.

    Imports stage modules lazily so callers that only need path helpers do
    not pay for torch, whisperx, or transformers at import time.

    Args:
        hf_token: Hugging Face token for gated WhisperX / pyannote weights.
            Falls back to ``HF_TOKEN`` in the environment.

    Returns:
        Mapping from stage name to a ready-to-run stage instance.
    """
    from open_enricher.stages.audio import AudioStage
    from open_enricher.stages.color import ColorStage
    from open_enricher.stages.judger import JudgerStage
    from open_enricher.stages.llm import LLMStage
    from open_enricher.stages.vlm import VLMStage

    token = (hf_token if hf_token is not None else os.getenv("HF_TOKEN", "")).strip()
    return {
        "audio": AudioStage(hf_token=token),
        "color": ColorStage(max_long_scenes=120, grid_interval_sec=180),
        "vlm": VLMStage(),
        "llm": LLMStage(),
        "judger": JudgerStage(),
    }


def primary_artifact_path(stage: str, video_filename: str) -> Path:
    """
    Return the path of the primary JSON artifact for a stage and video.

    Args:
        stage: Pipeline stage name.
        video_filename: Exact file name under ``video_inputs``.

    Returns:
        Path under the active experiment output folder.
    """
    pattern = STAGE_PRIMARY_ARTIFACT[stage]
    return db.get_output_dir(stage) / pattern.format(video=video_filename)


def stage_output_exists(stage: str, video_filename: str) -> bool:
    """
    Report whether the primary artifact for a stage already exists on disk.

    Args:
        stage: Pipeline stage name.
        video_filename: Exact file name under ``video_inputs``.

    Returns:
        True when the expected JSON file is present and non-empty.
    """
    path = primary_artifact_path(stage, video_filename)
    return path.is_file() and path.stat().st_size > 0


def _artifact_globs(stage: str, video_filename: str) -> list[str]:
    """
    Return filename globs for artifacts belonging to one stage and video.

    Args:
        stage: Pipeline stage name.
        video_filename: Exact file name under ``video_inputs``.

    Returns:
        Glob patterns relative to the stage output directory.
    """
    if stage == "audio":
        return [
            f"{video_filename}_audio.json",
            f"{video_filename}_audio_timings.json",
        ]
    if stage == "color":
        return [
            f"{video_filename}_color.json",
            f"{video_filename}_color_timings.json",
        ]
    if stage == "vlm":
        return [
            f"{video_filename}_vlm.json",
            f"{video_filename}_vlm_timings.json",
            f"{video_filename}_vlm_raw.txt",
            f"{video_filename}_vlm_sidecar.json",
            f"{video_filename}_vlm_context.txt",
            f"{video_filename}_dialogue_summary.txt",
        ]
    if stage == "llm":
        return [
            f"{video_filename}_llm.json",
            f"{video_filename}_llm_timings.json",
        ]
    if stage == "judger":
        return [f"{video_filename}_judger.json"]
    return []


def restore_cached_stage(
    stage: str,
    video_filename: str,
    cache_root: Path,
) -> bool:
    """
    Copy cached audio or color artifacts into the active experiment folders.

    Args:
        stage: Must be a cacheable stage (audio or color).
        video_filename: Exact file name under ``video_inputs``.
        cache_root: Directory holding per-video cache folders.

    Returns:
        True when the primary artifact was restored, False when the cache
        misses or the stage is not cacheable.
    """
    if stage not in CACHEABLE_STAGES:
        return False
    source_dir = cache_root / video_filename / stage
    if not source_dir.is_dir():
        return False
    dest_dir = db.get_output_dir(stage)
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, dest_dir / path.name)
            copied += 1
    ok = stage_output_exists(stage, video_filename)
    if ok:
        print(
            f"[INFO] Restored {copied} cached file(s) for {stage} / {video_filename}"
        )
    return ok


def save_stage_to_cache(
    stage: str,
    video_filename: str,
    cache_root: Path,
) -> None:
    """
    Persist audio or color artifacts into the shared cache for later runs.

    Args:
        stage: Must be a cacheable stage (audio or color).
        video_filename: Exact file name under ``video_inputs``.
        cache_root: Directory holding per-video cache folders.

    Side effects:
        Creates ``cache_root/<video>/<stage>/`` and copies matching artifacts.
    """
    if stage not in CACHEABLE_STAGES:
        return
    if not stage_output_exists(stage, video_filename):
        return
    dest_dir = cache_root / video_filename / stage
    dest_dir.mkdir(parents=True, exist_ok=True)
    source_dir = db.get_output_dir(stage)
    for pattern in _artifact_globs(stage, video_filename):
        for path in source_dir.glob(pattern):
            if path.is_file():
                shutil.copy2(path, dest_dir / path.name)


def list_videos(video_names: Sequence[str] | None = None) -> list[str]:
    """
    Resolve which videos to process from the data lake.

    Args:
        video_names: Explicit filenames. When None or empty, every regular
            file in ``db.video_inputs`` is used (sorted).

    Returns:
        Sorted list of video filenames that exist under ``video_inputs``.

    Raises:
        FileNotFoundError: When the input directory is missing, no videos
            are found, or a requested name is absent.
    """
    inputs = db.video_inputs
    if not inputs.is_dir():
        raise FileNotFoundError(f"Video input directory does not exist: {inputs}")

    if video_names:
        missing = [name for name in video_names if not (inputs / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Video(s) not found under "
                f"{inputs}: {', '.join(missing)}"
            )
        return list(video_names)

    found = sorted(
        path.name for path in inputs.iterdir() if path.is_file() and not path.name.startswith(".")
    )
    if not found:
        raise FileNotFoundError(f"No videos found under {inputs}")
    return found


def run_pipeline(
    video_filename: str,
    *,
    stages: Sequence[str] = PIPELINE_STAGES,
    stage_registry: dict[str, Any] | None = None,
    cache_root: Path | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Execute the requested stages for one video in process.

    Args:
        video_filename: Exact file name under ``video_inputs``.
        stages: Ordered stage names to run.
        stage_registry: Optional pre-built registry. Built on demand otherwise.
        cache_root: Shared cache for audio and color. When None, caching is
            skipped even if ``use_cache`` is True.
        use_cache: When True, restore or skip cacheable stages that already
            have artifacts, and refresh the cache after a successful run.

    Returns:
        Summary dict with per-stage status, overall status, and video name.
        Overall status is ``failed`` if any stage fails or raises.
    """
    video_path = db.video_inputs / video_filename
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found in data lake: {video_path}")

    registry = stage_registry if stage_registry is not None else build_stages()
    results: list[dict[str, Any]] = []
    overall = "success"

    for name in stages:
        if name not in registry:
            row = {
                "stage": name,
                "status": "failed",
                "error": f"Unknown stage '{name}'",
            }
            results.append(row)
            overall = "failed"
            break

        if use_cache and cache_root is not None and name in CACHEABLE_STAGES:
            if not stage_output_exists(name, video_filename):
                restore_cached_stage(name, video_filename, cache_root)
            if stage_output_exists(name, video_filename):
                print(f"[INFO] Skipping {name} for {video_filename} (cached)")
                results.append(
                    {
                        "stage": name,
                        "status": "skipped_cache",
                        "output_file": primary_artifact_path(name, video_filename).name,
                    }
                )
                continue

        print(f"[INFO] Running stage '{name}' on {video_filename}")
        try:
            outcome = registry[name].run(video_filename)
        except Exception as exc:  # noqa: BLE001 - surface stage failures to the loop
            outcome = {
                "stage": name,
                "status": "failed",
                "error": str(exc),
            }
        results.append(outcome)
        status = str(outcome.get("status") or "failed")
        if status != "success":
            overall = "failed"
            print(f"[ERROR] Stage '{name}' failed: {outcome.get('error')}")
            break

        if use_cache and cache_root is not None and name in CACHEABLE_STAGES:
            save_stage_to_cache(name, video_filename, cache_root)

    return {
        "video": video_filename,
        "status": overall,
        "stages": results,
    }


def collect_artifacts(video_filename: str, stages: Iterable[str] = PIPELINE_STAGES) -> list[Path]:
    """
    List existing artifact paths for a video across the given stages.

    Args:
        video_filename: Exact file name under ``video_inputs``.
        stages: Stage names whose artifacts should be collected.

    Returns:
        Paths to files that currently exist on disk.
    """
    found: list[Path] = []
    for stage in stages:
        folder = db.get_output_dir(stage)
        for pattern in _artifact_globs(stage, video_filename):
            for path in folder.glob(pattern):
                if path.is_file():
                    found.append(path)
        # Color stage also writes a frames directory.
        if stage == "color":
            frames = folder / f"{video_filename}_frames"
            if frames.is_dir():
                found.append(frames)
    return found


def archive_artifacts(
    video_filename: str,
    dest_dir: Path,
    *,
    stages: Iterable[str] = PIPELINE_STAGES,
) -> list[str]:
    """
    Copy stage artifacts for one video into an archive directory.

    Args:
        video_filename: Exact file name under ``video_inputs``.
        dest_dir: Destination folder (created if needed).
        stages: Stage names to archive.

    Returns:
        List of destination file names that were copied.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in collect_artifacts(video_filename, stages):
        if path.is_dir():
            target = dest_dir / path.name
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(path, target)
            copied.append(path.name + "/")
            continue
        shutil.copy2(path, dest_dir / path.name)
        copied.append(path.name)
    log_path = db.experiment_log
    if log_path.is_file():
        shutil.copy2(log_path, dest_dir / "worker.log")
        copied.append("worker.log")
    return copied


def load_judger_score(video_filename: str) -> dict[str, Any] | None:
    """
    Load the judger report for a video when present.

    Args:
        video_filename: Exact file name under ``video_inputs``.

    Returns:
        Parsed JSON document, or None when the report is missing or invalid.
    """
    path = primary_artifact_path("judger", video_filename)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
