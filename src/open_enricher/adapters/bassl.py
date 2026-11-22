"""BaSSL scene boundaries through the ModelScope pipeline.

The published configuration and checkpoint are loaded together. The
preprocessor's 120-second ``end_time`` is replaced with the real
duration. Import and constructor failures stay inside the child process.
"""

from __future__ import annotations

import json
import tempfile
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
    """Child entry. Patch ``end_time`` and run the published pipeline."""
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
    except Exception as exc:  # noqa: BLE001 - optional dependency
        return {"status": "fallback", "reason": f"modelscope import failed: {exc}"}

    model_dir = Path(payload["model_dir"])
    duration = float(payload["duration"])
    video = str(payload["video"])
    config = json.loads((model_dir / "configuration.json").read_text(encoding="utf-8"))
    shot = config.setdefault("preprocessor", {}).setdefault("shot_detect", {})
    shot["begin_time"] = 0.0
    shot["end_time"] = duration
    shot["begin_frame"] = 0
    shot["end_frame"] = max(1, int(duration * 30) + 8)
    with tempfile.TemporaryDirectory(prefix="bassl-") as tmp:
        root = Path(tmp)
        (root / "configuration.json").write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        link = root / "pytorch_model.pt"
        link.symlink_to(model_dir / "pytorch_model.pt")
        try:
            pipe = pipeline(
                Tasks.movie_scene_segmentation,
                model=str(root),
                device="gpu",
            )
            result = pipe(video)
        except Exception as exc:  # noqa: BLE001 - recorded fallback
            return {"status": "fallback", "reason": str(exc), "scenes": [], "shots": []}
    return {
        "status": "success",
        "scenes": _rows(result, "SCENE_META_LIST", "scene_meta_list", "scenes"),
        "shots": _rows(result, "SHOT_META_LIST", "shot_meta_list", "shots"),
        "scene_num": _scalar(result, "SCENE_NUM", "scene_num"),
        "shot_num": _scalar(result, "SHOT_NUM", "shot_num"),
    }


def _rows(result: Any, *names: str) -> list[Any]:
    if not isinstance(result, dict):
        return []
    for name in names:
        if name in result:
            value = result[name]
            return list(value) if isinstance(value, list) else []
    return []


def _scalar(result: Any, *names: str) -> int | None:
    if not isinstance(result, dict):
        return None
    for name in names:
        if name in result:
            try:
                return int(result[name])
            except (TypeError, ValueError):
                return None
    return None
