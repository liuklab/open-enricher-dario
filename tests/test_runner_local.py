"""GPU-free tests for the local runner and experiment CLI helpers.

These tests never open sockets and never load torch models. They only check
path validation, CLI listing, and helpers against temporary directories.
Heavy stage imports are deferred until ``build_stages`` runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from open_enricher.experiments.loop import build_parser, main, run_one_experiment
from open_enricher.runner import list_videos, primary_artifact_path, run_pipeline
from open_enricher import storage


def test_cli_list_variants(capsys: pytest.CaptureFixture[str]) -> None:
    """``--list`` prints packaged variant ids without starting the pipeline."""
    main(["--list"])
    out = capsys.readouterr().out
    assert "01-scene-context" in out
    assert "10-claim-check" not in out


def test_run_one_experiment_evaluates_with_cursor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each experiment scores archived profiles with Cursor and the synopsis."""
    variant = tmp_path / "01-scene-context.json"
    variant.write_text(
        '{"experiment_id": "01-scene-context", "title": "scene-context"}',
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    monkeypatch.setattr("open_enricher.experiments.archive.RUNS_ROOT", runs)
    monkeypatch.setattr(
        "open_enricher.experiments.archive.PROGRESS_PATH",
        runs / "progress.jsonl",
    )

    class _Runner:
        def build_stages(self) -> dict:
            return {}

        def run_pipeline(self, video: str, **kwargs: object) -> dict:
            return {"video": video, "status": "success", "stages": []}

        def archive_artifacts(self, video: str, dest: Path, stages: object) -> list:
            return []

    monkeypatch.setattr(
        "open_enricher.experiments.loop._runner", lambda: _Runner()
    )
    seen: dict = {}

    def _evaluate(videos: list[str], **kwargs: object) -> list[dict]:
        seen["videos"] = videos
        seen["kwargs"] = kwargs
        return [{"video": videos[0], "status": "success", "overall": 4.1}]

    monkeypatch.setattr(
        "open_enricher.evaluation.run.evaluate_videos", _evaluate
    )

    summary = run_one_experiment(variant, videos=["Suits.S01E02.mkv"])

    assert summary["status"] == "ok"
    assert seen["videos"] == ["Suits.S01E02.mkv"]
    assert seen["kwargs"]["provider_name"] == "cursor"
    assert seen["kwargs"]["experiment_id"] == "01-scene-context"
    assert seen["kwargs"]["with_synopsis"] is True
    text = (runs / "01-scene-context" / "EXPERIMENT.md").read_text(encoding="utf-8")
    assert "overall `4.1`" in text


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
    monkeypatch.setenv("OPEN_ENRICHER_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(video_dir))

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
    monkeypatch.setenv("OPEN_ENRICHER_DATASET_ROOT", str(tmp_path))
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(video_dir))
    storage.setup_dataset()

    with pytest.raises(FileNotFoundError) as exc:
        run_pipeline("missing.mkv", use_cache=False)
    assert "missing.mkv" in str(exc.value)


def test_primary_artifact_path_uses_shared_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Artifact paths land under the shared dataset outputs/ tree."""
    monkeypatch.setenv("OPEN_ENRICHER_DATASET_ROOT", str(tmp_path))
    path = primary_artifact_path("llm", "episode.mkv")
    assert path == tmp_path / "outputs" / "llms_results" / "episode.mkv_llm.json"
