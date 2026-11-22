"""Score VideoProfile JSON files with an external evaluator provider."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from open_enricher.evaluation.prompt import system_prompt, user_prompt
from open_enricher.evaluation.providers import DEFAULT_PROVIDER, get_provider
from open_enricher.evaluation.providers.base import EvaluatorProvider
from open_enricher.evaluation.score import (
    build_report,
    failed_report,
    title_from_filename,
)
from open_enricher.evaluation.synopsis import load_synopsis
from open_enricher.experiments.archive import experiment_dir, video_archive_dir
from open_enricher.logging_setup import get_logger
from open_enricher.manifest import profile_decision
from open_enricher import storage

logger = get_logger(__name__)


def evaluation_output_dir() -> Path:
    """
    Return the shared live evaluation output directory.

    Returns:
        ``dataset/outputs/evaluation``.
    """
    path = storage.outputs_dir() / "evaluation"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _clean_json_candidate(raw_text: str) -> str:
    """Isolate the JSON object between the first '{' and last '}'."""
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return cleaned[start : end + 1]
    return cleaned


def _load_json(path: Path) -> Any:
    """Load a JSON file; return None when missing or unreadable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        logger.warning("Failed to parse %s: %s", path.name, err)
        return None


def resolve_profile_path(
    video_filename: str,
    *,
    experiment_id: str | None = None,
) -> Path:
    """
    Locate the LLM profile JSON for a video.

    Args:
        video_filename: Exact video file name.
        experiment_id: When set, look under the experiment archive first.

    Returns:
        Path to ``*_llm.json``.

    Raises:
        FileNotFoundError: When no profile is found.
    """
    if experiment_id:
        archived = video_archive_dir(experiment_id, video_filename) / (
            f"{video_filename}_llm.json"
        )
        if archived.is_file():
            return archived
    live = storage.get_output_dir("llm") / f"{video_filename}_llm.json"
    if live.is_file():
        return live
    raise FileNotFoundError(
        f"Missing LLM profile for {video_filename}"
        + (f" (experiment {experiment_id})" if experiment_id else "")
    )


def resolve_output_paths(
    video_filename: str,
    *,
    experiment_id: str | None = None,
) -> tuple[Path, Path]:
    """
    Resolve report and raw-text output paths for one video.

    Args:
        video_filename: Exact video file name.
        experiment_id: When set, write ``score.json`` under the archive.

    Returns:
        ``(report_path, raw_path)``.
    """
    if experiment_id:
        dest = video_archive_dir(experiment_id, video_filename)
        dest.mkdir(parents=True, exist_ok=True)
        return dest / "score.json", dest / f"{video_filename}_evaluation_raw.txt"
    out = evaluation_output_dir()
    return (
        out / f"{video_filename}_evaluation.json",
        out / f"{video_filename}_evaluation_raw.txt",
    )


def evaluate_video(
    video_filename: str,
    *,
    provider: EvaluatorProvider,
    with_synopsis: bool = False,
    synopsis_dir: Path | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """
    Score one VideoProfile with a fresh provider conversation.

    Args:
        video_filename: Exact video file name under ``video_inputs``.
        provider: Provider instance (new conversation per call).
        with_synopsis: When True, load and inject the synopsis file.
        synopsis_dir: Optional synopsis directory override.
        experiment_id: Optional experiment archive id for I/O paths.

    Returns:
        Status dict with overall score, usability flag, and output file name.
    """
    title = title_from_filename(video_filename)
    provider_name = getattr(provider, "name", DEFAULT_PROVIDER)
    model = getattr(provider, "model", None)
    report_path, raw_path = resolve_output_paths(
        video_filename, experiment_id=experiment_id
    )

    try:
        profile_path = resolve_profile_path(
            video_filename, experiment_id=experiment_id
        )
    except FileNotFoundError as exc:
        report = failed_report(
            video_filename=video_filename,
            title=title,
            profile_file=None,
            error=str(exc),
            uses_synopsis=with_synopsis,
            provider=provider_name,
            model=model,
        )
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {
            "video": video_filename,
            "status": "failed",
            "error": report["error"],
            "output_file": str(report_path),
        }

    manifest = None
    if experiment_id:
        loaded = _load_json(
            video_archive_dir(experiment_id, video_filename)
            / f"{video_filename}_manifest.json"
        )
        if isinstance(loaded, dict):
            manifest = loaded
    decision = profile_decision(manifest, profile_path.name)
    if decision == "degraded":
        note = {
            "video": video_filename,
            "title": title,
            "status": "degraded",
            "profile_kind": "degraded",
            "scored": False,
            "profile_file": profile_path.name,
        }
        report_path.write_text(
            json.dumps(note, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "video": video_filename,
            "status": "degraded",
            "output_file": str(report_path),
        }
    if decision == "rejected":
        report = failed_report(
            video_filename=video_filename,
            title=title,
            profile_file=profile_path.name,
            error="Profile is not an accepted artifact of this attempt.",
            uses_synopsis=with_synopsis,
            provider=provider_name,
            model=model,
        )
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "video": video_filename,
            "status": "failed",
            "error": report["error"],
            "output_file": str(report_path),
        }

    profile_data = _load_json(profile_path)
    if profile_data is None or not isinstance(profile_data, dict):
        report = failed_report(
            video_filename=video_filename,
            title=title,
            profile_file=profile_path.name,
            error=f"Unreadable LLM profile: {profile_path.name}",
            uses_synopsis=with_synopsis,
            provider=provider_name,
            model=model,
        )
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {
            "video": video_filename,
            "status": "failed",
            "error": report["error"],
            "output_file": str(report_path),
        }

    synopsis_text: str | None = None
    if with_synopsis:
        try:
            synopsis_text = load_synopsis(
                video_filename, synopsis_dir=synopsis_dir
            )
        except (FileNotFoundError, ValueError) as exc:
            report = failed_report(
                video_filename=video_filename,
                title=title,
                profile_file=profile_path.name,
                error=str(exc),
                uses_synopsis=True,
                provider=provider_name,
                model=model,
            )
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return {
                "video": video_filename,
                "status": "failed",
                "error": report["error"],
                "output_file": str(report_path),
            }

    system = system_prompt()
    user = user_prompt(
        title=title, profile=profile_data, synopsis=synopsis_text
    )

    last_raw = ""
    try:
        logger.info(
            "Evaluating %s with provider=%s model=%s synopsis=%s",
            video_filename,
            provider_name,
            model,
            with_synopsis,
        )
        last_raw = provider.complete(system, user)
    except Exception as exc:  # noqa: BLE001 - surface provider failures as reports
        report = failed_report(
            video_filename=video_filename,
            title=title,
            profile_file=profile_path.name,
            error=f"Provider failed: {exc}",
            uses_synopsis=with_synopsis,
            provider=provider_name,
            model=model,
        )
        if last_raw:
            raw_path.write_text(last_raw, encoding="utf-8")
            report["raw_output_file"] = raw_path.name
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {
            "video": video_filename,
            "status": "failed",
            "error": report["error"],
            "output_file": str(report_path),
        }

    candidate = _clean_json_candidate(last_raw)
    try:
        payload = json.loads(candidate)
        if not isinstance(payload, dict):
            raise ValueError("Evaluator response is not a JSON object")
        report = build_report(
            video_filename=video_filename,
            title=title,
            profile_file=profile_path.name,
            llm_payload=payload,
            uses_synopsis=with_synopsis,
            provider=provider_name,
            model=model,
        )
        status = "success"
        error = None
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raw_path.write_text(last_raw, encoding="utf-8")
        report = failed_report(
            video_filename=video_filename,
            title=title,
            profile_file=profile_path.name,
            error=f"Evaluator JSON parse failed: {exc}",
            uses_synopsis=with_synopsis,
            provider=provider_name,
            model=model,
        )
        report["raw_output_file"] = raw_path.name
        status = "failed"
        error = report["error"]

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    dimension_scores = {
        name: payload["score"] for name, payload in report["dimensions"].items()
    }
    result: dict[str, Any] = {
        "video": video_filename,
        "status": status,
        "title": title,
        "overall": report["overall"],
        "overall_normalized": report["overall_normalized"],
        "usable": report["usable"],
        "dimensions": dimension_scores,
        "output_file": str(report_path),
        "uses_synopsis": with_synopsis,
        "provider": provider_name,
        "model": model,
    }
    if error:
        result["error"] = error
    return result


def evaluate_videos(
    videos: list[str],
    *,
    provider_name: str = DEFAULT_PROVIDER,
    with_synopsis: bool = False,
    synopsis_dir: Path | None = None,
    experiment_id: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """
    Evaluate one or more videos with a fresh provider conversation each time.

    Args:
        videos: Video filenames to score.
        provider_name: Registry key (default ``cursor``).
        with_synopsis: Inject synopsis text when True.
        synopsis_dir: Optional synopsis directory override.
        experiment_id: Optional experiment archive id.
        api_key: Optional API key for the provider.
        model: Optional model id override.

    Returns:
        List of per-video status dicts. When ``experiment_id`` is set, also
        refreshes ``scores.json`` under the experiment folder.
    """
    provider = get_provider(provider_name, api_key=api_key, model=model)
    results: list[dict[str, Any]] = []
    for video in videos:
        results.append(
            evaluate_video(
                video,
                provider=provider,
                with_synopsis=with_synopsis,
                synopsis_dir=synopsis_dir,
                experiment_id=experiment_id,
            )
        )

    if experiment_id:
        exp = experiment_dir(experiment_id)
        scores: dict[str, Any] = {}
        for video in videos:
            path = video_archive_dir(experiment_id, video) / "score.json"
            if path.is_file():
                try:
                    scores[video] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
        scores_path = exp / "scores.json"
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores_path.write_text(
            json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return results
