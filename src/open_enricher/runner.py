"""In-process pipeline runner for a local GPU host.

Builds the stage registry and calls ``stage.run(video)`` in the configured order.
Audio output can be restored from a shared cache so a later run skips
transcription that does not depend on the pipeline spec.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv

from open_enricher import storage
from open_enricher.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)

# Default local pipeline order: audio -> dialogue -> vlm -> llm.
# Stage classes are imported lazily inside ``build_stages`` so CLI helpers
# and unit tests can import this module without loading torch or whisperx.
PIPELINE_STAGES: tuple[str, ...] = ("audio", "dialogue", "vlm", "llm")

# Stages whose outputs do not depend on the active pipeline_spec knobs.
CACHEABLE_STAGES: frozenset[str] = frozenset({"audio"})

# Primary JSON artifact written by each stage for a given video filename.
STAGE_PRIMARY_ARTIFACT: dict[str, str] = {
    "audio": "{video}_audio.json",
    "dialogue": "{video}_dialogue.json",
    "vlm": "{video}_vlm.json",
    "llm": "{video}_llm.json",
    "timeline": "{video}_timeline.json",
    "bassl": "{video}_bassl.json",
    "character_graph": "{video}_graph.json",
    "visual_memory": "{video}_memory.json",
    "evidence": "{video}_manifest.json",
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
    from open_enricher.stages.dialogue import DialogueStage
    from open_enricher.stages.llm import LLMStage
    from open_enricher.stages.optional import (
        BasslStage,
        CharacterGraphStage,
        TimelineStage,
        VisualMemoryStage,
    )
    from open_enricher.stages.vlm import VLMStage

    token = (hf_token if hf_token is not None else os.getenv("HF_TOKEN", "")).strip()
    return {
        "audio": AudioStage(hf_token=token),
        "timeline": TimelineStage(),
        "bassl": BasslStage(),
        "dialogue": DialogueStage(),
        "character_graph": CharacterGraphStage(),
        "visual_memory": VisualMemoryStage(),
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
            f"{video_filename}_audio_language.json",
            f"{video_filename}_audio_timings.json",
        ]
    if stage == "dialogue":
        return [
            f"{video_filename}_dialogue.json",
            f"{video_filename}_dialogue_timings.json",
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
            f"{video_filename}_vlm_frames.json",
            f"{video_filename}_vlm_raw.txt",
            f"{video_filename}_vlm_sidecar.json",
            f"{video_filename}_vlm_context.txt",
            f"{video_filename}_dialogue_summary.txt",
            f"{video_filename}_frame_plan.json",
        ]
    if stage == "llm":
        return [
            f"{video_filename}_llm.json",
            f"{video_filename}_llm_timings.json",
            f"{video_filename}_llm_raw.txt",
        ]
    if stage == "timeline":
        return [f"{video_filename}_timeline.json"]
    if stage == "bassl":
        return [f"{video_filename}_bassl.json"]
    if stage == "character_graph":
        return [f"{video_filename}_graph.json"]
    if stage == "visual_memory":
        return [f"{video_filename}_memory.json"]
    if stage == "evidence":
        return [
            f"{video_filename}_manifest.json",
            f"{video_filename}_evidence.json",
        ]
    return []


def _cache_path(stage: str, video_filename: str, cache_root: Path, options: dict | None) -> Path:
    from open_enricher.fingerprint import audio_model_identity, source_fingerprint

    source = source_fingerprint(storage.video_inputs() / video_filename)
    settings = {
        "code": source_fingerprint(Path(__file__).parent / "stages" / f"{stage}.py"),
        "options": options, "models": audio_model_identity(),
    }
    producer = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    return cache_root / video_filename / stage / f"{source}-{producer}.json"


def restore_cached_stage(stage: str, video_filename: str, cache_root: Path, options: dict | None = None) -> bool:
    """Restore a complete stage entry for this source, ASR models and producer settings."""
    if stage not in CACHEABLE_STAGES:
        return False
    path = _cache_path(stage, video_filename, cache_root, options)
    try:
        files = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    primary = primary_artifact_path(stage, video_filename)
    names = _artifact_globs(stage, video_filename)
    if not isinstance(files, dict) or not files.get(primary.name):
        return False
    if any(name not in names or not isinstance(body, str) for name, body in files.items()):
        return False
    for name, body in files.items():
        storage.atomic_write_text(primary.parent / name, body)
    return True


def save_stage_to_cache(stage: str, video_filename: str, cache_root: Path, options: dict | None = None) -> None:
    """Publish one atomic cache entry containing the stage output and timings."""
    if stage not in CACHEABLE_STAGES or not stage_output_exists(stage, video_filename):
        return
    directory = storage.get_output_dir(stage)
    files = {
        name: (directory / name).read_text(encoding="utf-8")
        for name in _artifact_globs(stage, video_filename) if (directory / name).is_file()
    }
    storage.atomic_write_text(_cache_path(stage, video_filename, cache_root, options), json.dumps(files))


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
        cache_root: Shared cache for audio. When None, caching is
            skipped even if ``use_cache`` is True.
        use_cache: Restore matching cache entries and publish successful stages.

    Returns:
        Summary dict with per-stage status, overall status, and video name.
        Overall status is ``failed`` if any stage fails or raises.
    """
    video_path = storage.video_inputs() / video_filename
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found in dataset: {video_path}")

    registry = stage_registry if stage_registry is not None else build_stages()
    spec = _load_spec()
    experimental = bool(spec is not None and spec.features.any_enabled)
    from open_enricher.experiments.stages import stages_for_spec

    stage_names = stages_for_spec(stages, spec)
    results: list[dict[str, Any]] = []
    overall = "success"
    previous_env: dict[str, str | None] = {}
    if experimental and spec is not None:
        previous_env = _begin_attempt(video_path, spec, tuple(stages))

    try:
        for name in stage_names:
            if name not in registry:
                row = {
                    "stage": name,
                    "status": "failed",
                    "error": f"Unknown stage '{name}'",
                }
                results.append(row)
                overall = "failed"
                break

            # A live sidecar may belong to an earlier clip or failed attempt.
            # Clear the requested stage only, preserving inputs for explicit subsets.
            for filename in _artifact_globs(name, video_filename):
                (storage.get_output_dir(name) / filename).unlink(missing_ok=True)
            caching = use_cache and cache_root is not None and name in CACHEABLE_STAGES
            options = None
            if caching and hasattr(registry[name], "cache_settings"):
                options = registry[name].cache_settings()
            if caching and restore_cached_stage(name, video_filename, cache_root, options):
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
            if not isinstance(outcome, dict):
                outcome = {"stage": name, "status": "failed", "error": "stage returned no document"}
            outcome.setdefault("stage", name)
            results.append(outcome)
            status = str(outcome.get("status") or "failed")
            if status == "success" and caching and outcome.get("diarized", True):
                save_stage_to_cache(name, video_filename, cache_root, options)
            if status in {"error", "failed"} and name in STAGE_PRIMARY_ARTIFACT:
                primary_artifact_path(name, video_filename).unlink(missing_ok=True)
            if experimental:
                if status not in {"success", "degraded", "fallback", "skipped_cache"}:
                    logger.error("Stage '%s' failed: %s", name, outcome.get("error"))
                if name == "timeline" and status == "failed":
                    overall = "failed"
                    break
                if name == "llm" and status not in {"success", "degraded"}:
                    overall = "failed"
                    break
                continue
            if name == "audio" and outcome.get("diarized") is False:
                overall = "degraded"
            if status != "success":
                overall = "failed"
                logger.error("Stage '%s' failed: %s", name, outcome.get("error"))
                break

    finally:
        try:
            if experimental and spec is not None:
                manifest = _publish(video_filename, spec, results)
                published = str(manifest.get("profile_kind") or "none")
                if published == "full":
                    overall = "success"
                elif published == "degraded":
                    overall = "degraded"
                else:
                    overall = "failed"
        finally:
            _end_attempt(previous_env)

    return {
        "video": video_filename,
        "status": overall,
        "stages": results,
        "stages_ran": [row["stage"] for row in results if row["stage"] in STAGE_PRIMARY_ARTIFACT]
        + (["evidence"] if experimental else []),
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
        for filename in _artifact_globs(stage, video_filename):
            path = folder / filename
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


def _load_spec() -> Any:
    """Load the active spec. A missing file keeps the legacy stage list."""
    try:
        from open_enricher.stages.pipeline_spec import load_pipeline_spec

        return load_pipeline_spec()
    except Exception as exc:  # noqa: BLE001 - the control path must still start
        logger.warning("Pipeline spec unavailable: %s", exc)
        return None


def _begin_attempt(video_path: Path, spec: Any, stages: tuple[str, ...]) -> dict[str, str | None]:
    """Publish the attempt id for optional stages and return the previous env."""
    from open_enricher.fingerprint import (
        ATTEMPT_ID,
        SETTINGS_HASH,
        SOURCE_FINGERPRINT,
        settings_hash,
        source_fingerprint,
    )

    keys = (ATTEMPT_ID, SOURCE_FINGERPRINT, SETTINGS_HASH)
    previous = {key: os.environ.get(key) for key in keys}
    os.environ[ATTEMPT_ID] = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    os.environ[SOURCE_FINGERPRINT] = source_fingerprint(video_path)
    os.environ[SETTINGS_HASH] = settings_hash(spec, stages=stages)
    return previous


def _end_attempt(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _publish(video_filename: str, spec: Any, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Write the evidence note and the manifest for this attempt."""
    from open_enricher.fingerprint import current_stamp
    from open_enricher.manifest import publish_manifest

    kind, reason = _profile_kind(spec, results)
    evidence_path = storage.get_output_dir("evidence") / f"{video_filename}_evidence.json"
    note = {
        "video": video_filename,
        "profile_kind": kind,
        "terminal_reason": reason,
        "outcomes": results,
    }
    note.update(current_stamp())
    storage.atomic_write_text(
        evidence_path,
        json.dumps(note, indent=2, ensure_ascii=False) + "\n",
    )
    paths: list[Path] = [evidence_path]
    for row in results:
        if row.get("status") in {"success", "degraded", "fallback", "skipped_cache"}:
            paths.extend(collect_artifacts(video_filename, [row["stage"]]))
    stamp = current_stamp()
    return publish_manifest(
        video_filename,
        attempt_id=stamp["attempt_id"],
        source_fingerprint=stamp["source_fingerprint"],
        settings_hash=stamp["settings_hash"],
        profile_kind=kind,
        outcomes=results,
        artifact_paths=paths,
        terminal_reason=reason,
        features={
            "bassl": spec.features.bassl,
            "adaptive_frames": spec.features.adaptive_frames,
            "character_graph": spec.features.character_graph,
            "visual_memory": spec.features.visual_memory,
        },
    )


def _profile_kind(spec: Any, results: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Map stage outcomes to full, degraded, or none."""
    by_name = {str(row.get("stage")): row for row in results}
    degraded = any(
        (by_name.get(name) or {}).get("status") not in {"success", "skipped_cache"}
        for name in ("audio", "dialogue", "vlm")
    )
    if (by_name.get("audio") or {}).get("diarized") is False:
        degraded = True
    for name, enabled in (
        ("bassl", spec.features.bassl),
        ("character_graph", spec.features.character_graph),
        ("visual_memory", spec.features.visual_memory),
    ):
        if not enabled:
            continue
        status = str((by_name.get(name) or {}).get("status") or "failed")
        if status in {"fallback", "failed", "degraded"}:
            degraded = True
    llm = by_name.get("llm") or {}
    if llm.get("status") == "success" and not degraded:
        return "full", None
    if llm.get("status") == "success":
        return "degraded", None
    return "none", str(llm.get("error") or "no validated profile")
