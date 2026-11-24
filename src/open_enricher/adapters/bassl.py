"""Scene boundaries through the full-video ModelScope demo linked by BaSSL's authors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from open_enricher import storage
from open_enricher.adapters.isolate import call_in_child
from open_enricher.stages.pipeline_spec import PipelineSpec


def checkpoint_dir(spec: PipelineSpec) -> Path:
    """Directory that holds ``configuration.json`` and ``pytorch_model.pt``."""
    if spec.checkpoints.bassl:
        return Path(spec.checkpoints.bassl)
    return storage.models_cache() / "external" / "bassl"


def segment_video(
    video: Path,
    duration: float,
    spec: PipelineSpec,
) -> dict[str, Any]:
    """
    Return shot and scene metadata, or a fallback document.

    The parent does not import ModelScope.
    """
    model_dir = checkpoint_dir(spec)
    weights = model_dir / "pytorch_model.pt"
    config = model_dir / "configuration.json"
    if not weights.is_file() or not config.is_file():
        return {
            "status": "fallback",
            "reason": f"BaSSL checkpoint missing under {model_dir}",
            "scenes": [],
            "shots": [],
        }
    return call_in_child(
        "open_enricher.adapters.bassl",
        "run_modelscope_payload",
        {
            "video": str(video),
            "duration": duration,
            "model_dir": str(model_dir),
        },
    )


def run_modelscope_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the published detector with indexed decoding for overlapping shot windows."""
    from modelscope.outputs import OutputKeys
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    video = str(payload["video"])
    pipe = pipeline(Tasks.movie_scene_segmentation, model=str(payload["model_dir"]))
    # The packaged shot detector scans the full source when both starts are zero.
    shot = pipe.model.cfg.preprocessor.shot_detect
    shot.begin_time = 0.0
    shot.begin_frame = 0
    shot.end_time = float(payload["duration"])
    with IndexedKeyframes(video) as keyframes:
        pipe.model.shot_detector.get_frame_img = keyframes.read_batch
        result = pipe(video)
    return {
        "status": "success",
        "scenes": result[OutputKeys.SCENE_META_LIST],
        "shots": result[OutputKeys.SHOT_META_LIST],
        "scene_num": result[OutputKeys.SCENE_NUM],
        "shot_num": result[OutputKeys.SHOT_NUM],
    }


class IndexedKeyframes:
    """Read requested frames, including repeated and overlapping shot windows."""

    def __init__(self, video: str) -> None:
        import cv2

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(video)
        self.last_index: int | None = None
        self.last_image: Any = None

    def __enter__(self) -> IndexedKeyframes:
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError("BaSSL could not open the video for keyframe decoding")
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.capture.release()

    def read_batch(
        self, timecodes: dict[int, list[Any]], start_shot_idx: int, total_shot_num: int
    ) -> list[list[Any]]:
        from PIL import Image

        # ModelScope supplies the keyframe indices for every shot in this batch.
        shots = []
        for shot, frames in timecodes.items():
            images = []
            for timecode in frames:
                index = timecode.get_frames()
                if self.last_index != index:
                    position = int(self.capture.get(self.cv2.CAP_PROP_POS_FRAMES))
                    if index < position or index - position > 48:
                        if not self.capture.set(self.cv2.CAP_PROP_POS_FRAMES, index):
                            raise ValueError(f"BaSSL could not seek to frame {index} in shot {shot}")
                        position = int(self.capture.get(self.cv2.CAP_PROP_POS_FRAMES))
                    if position > index:
                        raise ValueError(f"BaSSL seek skipped frame {index} in shot {shot}")
                    for _ in range(position, index):
                        if not self.capture.grab():
                            raise ValueError(f"BaSSL could not advance to frame {index} in shot {shot}")
                    ok, frame = self.capture.read()
                    if not ok:
                        raise ValueError(f"BaSSL could not decode frame {index} in shot {shot}")
                    self.last_image = Image.fromarray(
                        self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                    )
                    self.last_index = index
                images.append(self.last_image)
            if not images:
                raise ValueError(f"BaSSL requested no keyframes for shot {shot}")
            shots.append(images)
        return shots
