"""A short-file archive must not count as the full film."""

from __future__ import annotations

import json
from pathlib import Path

from open_enricher.experiments.archive import (
    append_progress,
    archive_matches_sources,
    retain_different_source,
)
from open_enricher.fingerprint import source_fingerprint
from open_enricher.runner import restore_cached_stage


def _video(directory: Path, name: str, payload: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    return path


def _manifest(runs: Path, video: str, fingerprint: str) -> None:
    folder = runs / "12-combined" / video
    folder.mkdir(parents=True)
    (folder / f"{video}_manifest.json").write_text(
        json.dumps({"source_fingerprint": fingerprint, "profile_kind": "degraded"}),
        encoding="utf-8",
    )
    (folder / f"{video}_llm.json").write_text('{"kept": true}\n', encoding="utf-8")


def test_short_archive_does_not_cover_the_full_films(
    tmp_path: Path, monkeypatch
) -> None:
    """One 30-second HIMYM row does not settle the three-film request."""
    runs = tmp_path / "runs"
    monkeypatch.setattr("open_enricher.experiments.archive.RUNS_ROOT", runs)
    monkeypatch.setattr(
        "open_enricher.experiments.archive.PROGRESS_PATH",
        runs / "progress.jsonl",
    )
    videos = tmp_path / "video_inputs"
    short = _video(videos, "How_I_Met_Your_Mother.S01E02.mkv", b"short-clip")
    _manifest(runs, short.name, source_fingerprint(short))
    append_progress(
        {
            "experiment_id": "12-combined",
            "step": "archived",
            "status": "ok",
            "results": [{"video": short.name, "status": "degraded"}],
        }
    )
    requested = [
        short.name,
        "Silicon_Valley.S06E01.mkv",
        "The_Kingdom.S01E02.mkv",
    ]
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(videos))
    assert archive_matches_sources("12-combined", requested, "archived") is False


def test_same_name_with_a_different_file_is_not_done(
    tmp_path: Path, monkeypatch
) -> None:
    """A manifest from the clip does not match the full file of the same name."""
    runs = tmp_path / "runs"
    monkeypatch.setattr("open_enricher.experiments.archive.RUNS_ROOT", runs)
    monkeypatch.setattr(
        "open_enricher.experiments.archive.PROGRESS_PATH",
        runs / "progress.jsonl",
    )
    videos = tmp_path / "video_inputs"
    short = _video(videos, "How_I_Met_Your_Mother.S01E02.mkv", b"short-clip")
    fingerprint = source_fingerprint(short)
    _manifest(runs, short.name, fingerprint)
    append_progress(
        {
            "experiment_id": "12-combined",
            "step": "archived",
            "status": "ok",
            "results": [{"video": short.name, "status": "degraded"}],
        }
    )
    full = _video(videos, short.name, b"full-film-bytes")
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(videos))
    assert archive_matches_sources("12-combined", [short.name], "archived") is False

    kept = retain_different_source("12-combined", short.name, full)
    assert kept is not None
    assert kept.is_dir()
    assert json.loads((kept / f"{short.name}_llm.json").read_text(encoding="utf-8")) == {
        "kept": True
    }
    assert not (runs / "12-combined" / short.name).exists()


def test_audio_cache_from_another_file_is_not_restored(
    tmp_path: Path, monkeypatch
) -> None:
    """A cache without this file's fingerprint stays on disk and is not copied."""
    videos = tmp_path / "video_inputs"
    video = _video(videos, "How_I_Met_Your_Mother.S01E02.mkv", b"full-film")
    cache = tmp_path / "cache" / video.name / "audio"
    cache.mkdir(parents=True)
    (cache / f"{video.name}_audio.json").write_text("[]\n", encoding="utf-8")
    monkeypatch.setenv("OPEN_ENRICHER_DATASET_ROOT", str(tmp_path / "dataset"))
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(videos))
    (tmp_path / "dataset" / "outputs" / "audio_results").mkdir(parents=True)

    assert restore_cached_stage("audio", video.name, tmp_path / "cache") is False
    assert not (
        tmp_path / "dataset" / "outputs" / "audio_results" / f"{video.name}_audio.json"
    ).exists()
    assert (cache / f"{video.name}_audio.json").is_file()
