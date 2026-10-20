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
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

from open_enricher import storage
from open_enricher.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)

# Default stage order for a full local enricher pass.
# Stage classes are imported lazily inside ``build_stages`` so CLI helpers
# and unit tests can import this module without loading torch or whisperx.
PIPELINE_STAGES: tuple[str, ...] = ("audio", "color", "vlm", "llm")

# Stages whose outputs do not depend on the active pipeline_spec knobs.
CACHEABLE_STAGES: frozenset[str] = frozenset({"audio", "color"})

# Primary JSON artifact written by each stage for a given video filename.
STAGE_PRIMARY_ARTIFACT: dict[str, str] = {
    "audio": "{video}_audio.json",
    "color": "{video}_color.json",
    "vlm": "{video}_vlm.json",
    "llm": "{video}_llm.json",
}


def bootstrap_runtime(*, env_file: Path | None = None, tee_log: bool = True) -> None:
    """
    Load secrets, configure logging and caches, and create dataset folders.

    Args:
        env_file: Optional ``.env`` path. Defaults to ``<repo>/.env`` when
            present, otherwise relies on the process environment alone.
        tee_log: When True, also write CSV log lines to ``storage.experiment_log()``.

    Side effects:
        Configures the ``open_enricher`` logger, sets Hugging Face and Torch
        cache environment variables, and creates dataset directories.
    """
    if env_file is None:
        candidate = Path(__file__).resolve().parents[2] / ".env"
        if candidate.is_file():
            env_file = candidate
    if env_file is not None and env_file.is_file():
        load_dotenv(env_file)

    log_file = storage.experiment_log() if tee_log else None
    configure_logging(log_file=log_file, force=True)

    storage.configure_environment()
    storage.setup_dataset()

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token)
            logger.info("Hugging Face login succeeded.")
        except Exception as exc:  # noqa: BLE001 - login is best-effort
            logger.warning("Hugging Face login failed: %s", exc)

    if log_file is not None:
        logger.info("Worker log file: %s", log_file)


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
    from open_enricher.stages.llm import LLMStage
    from open_enricher.stages.vlm import VLMStage

    token = (hf_token if hf_token is not None else os.getenv("HF_TOKEN", "")).strip()
    return {
        "audio": AudioStage(hf_token=token),
        "color": ColorStage(max_long_scenes=120, grid_interval_sec=180),
        "vlm": VLMStage(),
        "llm": LLMStage(),
    }


def primary_artifact_path(stage: str, video_filename: str) -> Path:
    """
    Return the path of the primary JSON artifact for a stage and video.

    Args:
        stage: Pipeline stage name.
        video_filename: Exact file name under ``video_inputs``.

    Returns:
        Path under the shared dataset output folder.
    """
    pattern = STAGE_PRIMARY_ARTIFACT[stage]
    return storage.get_output_dir(stage) / pattern.format(video=video_filename)


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
    return []


def restore_cached_stage(
    stage: str,
    video_filename: str,
    cache_root: Path,
) -> bool:
    """
    Copy cached audio or color artifacts into the shared dataset folders.

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
    dest_dir = storage.get_output_dir(stage)
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, dest_dir / path.name)
            copied += 1
    ok = stage_output_exists(stage, video_filename)
    if ok:
        logger.info(
            "Restored %s cached file(s) for %s / %s",
            copied,
            stage,
            video_filename,
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
    """
    if stage not in CACHEABLE_STAGES:
        return
    if not stage_output_exists(stage, video_filename):
        return
    dest_dir = cache_root / video_filename / stage
    dest_dir.mkdir(parents=True, exist_ok=True)
    source_dir = storage.get_output_dir(stage)
    for pattern in _artifact_globs(stage, video_filename):
        for path in source_dir.glob(pattern):
            if path.is_file():
                shutil.copy2(path, dest_dir / path.name)


def list_videos(video_names: Sequence[str] | None = None) -> list[str]:
    """
    Resolve which videos to process from the dataset.

    Args:
        video_names: Explicit filenames. When None or empty, every regular
            file in ``storage.video_inputs()`` is used (sorted).

    Returns:
        Sorted list of video filenames that exist under ``video_inputs``.

    Raises:
        FileNotFoundError: When the input directory is missing, no videos
            are found, or a requested name is absent.
    """
    inputs = storage.video_inputs()
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
    video_path = storage.video_inputs() / video_filename
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found in dataset: {video_path}")

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
                logger.info("Skipping %s for %s (cached)", name, video_filename)
                results.append(
                    {
                        "stage": name,
                        "status": "skipped_cache",
                        "output_file": primary_artifact_path(name, video_filename).name,
                    }
                )
                continue

        logger.info("Running stage '%s' on %s", name, video_filename)
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
            logger.error("Stage '%s' failed: %s", name, outcome.get("error"))
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
        folder = storage.get_output_dir(stage)
        for pattern in _artifact_globs(stage, video_filename):
            for path in folder.glob(pattern):
                if path.is_file():
                    found.append(path)
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
    log_path = storage.experiment_log()
    if log_path.is_file():
        shutil.copy2(log_path, dest_dir / "worker.log")
        copied.append("worker.log")
    return copied
