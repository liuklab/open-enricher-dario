"""GPU-free tests for the local runner and experiment CLI helpers.

These tests never open sockets and never load torch models. They only check
path validation, CLI listing, and helpers against temporary directories.
Heavy stage imports are deferred until ``build_stages`` runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from open_enricher.experiments.loop import build_parser, main
from open_enricher.runner import list_videos, primary_artifact_path, run_pipeline
from open_enricher.storage import db


def test_cli_list_variants(capsys: pytest.CaptureFixture[str]) -> None:
    """``--list`` prints packaged variant ids without starting the pipeline."""
    main(["--list"])
    out = capsys.readouterr().out
    assert "01-interval-llm" in out
    assert "10-claim-check" in out


def test_build_parser_has_local_flags() -> None:
    """CLI exposes local-only flags and no Colab or Kaggle options."""
    parser = build_parser()
    help_text = parser.format_help()
    assert "--video" in help_text
    assert "--from" in help_text
    assert "colab" not in help_text.lower()
    assert "kaggle" not in help_text.lower()


def test_list_videos_missing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing video names raise FileNotFoundError without network access."""
    video_dir = tmp_path / "video_inputs"
    video_dir.mkdir()
    (video_dir / "present.mkv").write_bytes(b"fake")
    monkeypatch.setenv("OPEN_ENRICHER_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(video_dir))
    monkeypatch.delenv("OPEN_ENRICHER_EXPERIMENT_ID", raising=False)
    db.refresh()

    with pytest.raises(FileNotFoundError) as exc:
        list_videos(["absent.mkv"])
    assert "absent.mkv" in str(exc.value)
    assert list_videos(["present.mkv"]) == ["present.mkv"]


def test_run_pipeline_rejects_missing_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_pipeline`` fails fast when the video file is not on disk."""
    video_dir = tmp_path / "video_inputs"
    video_dir.mkdir()
    monkeypatch.setenv("OPEN_ENRICHER_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(video_dir))
    monkeypatch.delenv("OPEN_ENRICHER_EXPERIMENT_ID", raising=False)
    db.refresh()
    db.setup_database()

    with pytest.raises(FileNotFoundError) as exc:
        run_pipeline("missing.mkv", use_cache=False)
    assert "missing.mkv" in str(exc.value)


def test_primary_artifact_path_uses_experiment_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact paths land under outputs/<prefix>/<experiment_id>/."""
    monkeypatch.setenv("OPEN_ENRICHER_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("OPEN_ENRICHER_EXPERIMENT_PREFIX", "local")
    monkeypatch.setenv("OPEN_ENRICHER_EXPERIMENT_ID", "01-interval-llm")
    db.refresh()
    path = primary_artifact_path("llm", "episode.mkv")
    assert "outputs" in path.parts
    assert "local" in path.parts
    assert "01-interval-llm" in path.parts
    assert path.name == "episode.mkv_llm.json"
