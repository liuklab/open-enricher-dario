"""InsightFace sampling cap stays out of the full-video pipeline call."""

from __future__ import annotations

from pathlib import Path

from open_enricher.adapters.face_speech import _sample_tracks, detect_tracks
from open_enricher.stages.pipeline_spec import Checkpoints, PipelineSpec


def _spec(root: Path) -> PipelineSpec:
    return PipelineSpec(
        experiment_id="insightface-limit",
        title="insightface-limit",
        interval_seconds=300,
        vlm_memory_scenes=None,
        visual_only=False,
        narrate_from_visual=False,
        checkpoints=Checkpoints(insightface=str(root)),
    )


def test_max_frames_is_sent_only_when_set(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "models" / "buffalo_l").mkdir(parents=True)
    captured: list[dict] = []

    def fake(module: str, function: str, payload: dict, **kwargs: object) -> dict:
        del module, function, kwargs
        captured.append(payload)
        return {"status": "success", "tracks": []}

    monkeypatch.setattr("open_enricher.adapters.face_speech.call_in_child", fake)
    spec = _spec(tmp_path)
    detect_tracks(tmp_path / "video.mkv", spec, sample_fps=1.0, max_frames=20)
    detect_tracks(tmp_path / "video.mkv", spec, sample_fps=1.0)
    assert captured[0]["max_frames"] == 20
    assert "max_frames" not in captured[1]


def test_sample_loop_stops_at_max_frames() -> None:
    class _Capture:
        def __init__(self) -> None:
            self.left = 10

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            del prop
            return 1.0

        def read(self) -> tuple[bool, object]:
            if self.left <= 0:
                return False, None
            self.left -= 1
            return True, object()

        def release(self) -> None:
            return None

    class _CV:
        CAP_PROP_FPS = 5

        def VideoCapture(self, video: str) -> _Capture:
            del video
            return _Capture()

    class _App:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, frame: object) -> list[object]:
            del frame
            self.calls += 1
            return []

    app = _App()
    tracks = _sample_tracks(app, "video.mkv", sample_fps=1.0, cv2=_CV(), max_frames=3)
    assert tracks == []
    assert app.calls == 3
