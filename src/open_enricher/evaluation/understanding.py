"""Compare a generated scene timeline with an independent reference synopsis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from open_enricher import storage
from open_enricher.evaluation.prompt import clean_json_candidate, load_prompt
from open_enricher.evaluation.providers.base import EvaluatorProvider
from open_enricher.evaluation.synopsis import load_synopsis, synopsis_path
from open_enricher.experiments.archive import video_archive_dir
from open_enricher.manifest import load_json, manifest_path

REPORT_NAME = "video_understanding_score.json"
DIMENSIONS = ("event_coverage", "factual_fidelity", "chronology_and_causality", "scene_grounding")
SCENE_FIELDS = (
    "description", "visual_description", "dialogue_description", "subjects_and_objects",
    "actions", "on_screen_text", "camera_and_shot", "lighting_and_color",
)


def _scene_texts(scene: dict) -> list[str]:
    texts = []
    for field in SCENE_FIELDS:
        value = scene.get(field)
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str) and item.strip():
                texts.append(item)
    return texts


def _scenes(raw: bytes) -> list[dict]:
    document = json.loads(raw)
    rows = document.get("scene_timeline") if isinstance(document, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("A nonempty scene_timeline is required.")
    scenes = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("start_time", "end_time")
        ):
            raise ValueError(f"Missing or invalid time window in scene {index}.")
        scene = {"scene_id": index, "start_time": row["start_time"], "end_time": row["end_time"]}
        scene.update({key: row[key] for key in SCENE_FIELDS if row.get(key)})
        scenes.append(scene)
    if not any(_scene_texts(scene) for scene in scenes):
        raise ValueError("The timeline contains no scene descriptions or observations.")
    return scenes


def _required_text(obj: dict, key: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected nonempty text in {key}.")
    return value.strip()


def _contains_quote(source: str, quote: str) -> bool:
    return " ".join(quote.split()) in " ".join(source.split())


def _citation(citation: dict, scenes: list[dict]) -> dict:
    scene_id = citation.get("scene_id") if isinstance(citation, dict) else None
    if type(scene_id) is not int or not 1 <= scene_id <= len(scenes):
        raise ValueError(f"Unknown scene ID: {scene_id!r}.")
    scene = scenes[scene_id - 1]
    quote = _required_text(citation, "quote")
    if not any(_contains_quote(text, quote) for text in _scene_texts(scene)):
        raise ValueError(f"Evidence quote was not found in scene {scene_id}.")
    return {
        "scene_id": scene_id,
        "start_time": scene["start_time"],
        "end_time": scene["end_time"],
        "quote": quote,
    }


def _finding(finding: dict, scenes: list[dict], synopsis: str) -> dict:
    if not isinstance(finding, dict):
        raise ValueError("Expected a finding object.")
    kind = finding.get("kind")
    if kind not in ("supported", "partial", "contradicted", "missing", "unverifiable"):
        raise ValueError(f"Invalid finding kind: {kind!r}.")
    reference = finding.get("synopsis_quote")
    if kind == "unverifiable":
        if reference != "":
            raise ValueError("An unverifiable claim requires an empty synopsis quote.")
    elif not _contains_quote(synopsis, _required_text(finding, "synopsis_quote")):
        raise ValueError("A finding quotes text absent from the synopsis.")

    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("Scene evidence must be a list.")
    if kind == "missing" and evidence:
        raise ValueError("A missing event cannot cite a scene.")
    if kind != "missing" and not evidence:
        raise ValueError(f"A {kind} finding needs a scene citation.")
    return {
        "kind": kind,
        "claim": _required_text(finding, "claim"),
        "synopsis_quote": reference,
        "evidence": [_citation(item, scenes) for item in evidence],
    }


def _grade(raw: str, scenes: list[dict], synopsis: str) -> dict:
    payload = json.loads(clean_json_candidate(raw))
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    dims = payload.get("dimensions")
    if not isinstance(dims, dict) or set(dims) != set(DIMENSIONS):
        raise ValueError("Expected all four video understanding dimensions.")
    dimensions = {}
    for name in DIMENSIONS:
        entry = dims[name]
        score = entry.get("score") if isinstance(entry, dict) else None
        if type(score) is not int or not 1 <= score <= 5:
            raise ValueError(f"Invalid score for {name}. Expected an integer from 1 to 5.")
        dimensions[name] = {"score": score, "note": _required_text(entry, "note")}

    findings = payload.get("findings")
    if not isinstance(findings, list) or not findings:
        raise ValueError("Expected at least one finding.")
    checked = [_finding(finding, scenes, synopsis) for finding in findings]
    if all(finding["kind"] == "unverifiable" for finding in checked):
        raise ValueError("At least one finding must assess a reference event.")
    return {
        "overall": sum(dim["score"] for dim in dimensions.values()) / len(dimensions),
        "dimensions": dimensions,
        "summary": _required_text(payload, "summary"),
        "findings": checked,
    }


def evaluate_understanding(
    video: str,
    *,
    provider: EvaluatorProvider,
    synopsis_dir: Path | None = None,
    experiment_id: str | None = None,
) -> dict[str, Any]:
    """Score all windows in one conversation, independently of the RecSys profile."""
    if experiment_id:
        folder = video_archive_dir(experiment_id, video)
        timeline = folder / f"{video}_vlm.json"
        manifest = folder / f"{video}_manifest.json"
        output = folder / REPORT_NAME
    else:
        timeline = storage.get_output_dir("vlm") / f"{video}_vlm.json"
        manifest = manifest_path(video)
        output = storage.outputs_dir() / "evaluation" / f"{video}_video_understanding.json"

    report: dict[str, Any] = {
        "task": "video_understanding",
        "video": video,
        "status": "failed",
        "scored": False,
        "method": {
            "provider": provider.name,
            "model": provider.model,
            "uses_synopsis": True,
            "uses_pipeline_evidence": True,
            "rubric": "video_understanding_v2",
            "aggregation": "unweighted_mean",
        },
        "timeline_file": timeline.name,
        "synopsis_file": synopsis_path(video, synopsis_dir).name,
    }
    raw = ""
    try:
        synopsis = load_synopsis(video, synopsis_dir=synopsis_dir)
        report["synopsis_sha256"] = hashlib.sha256(synopsis.encode("utf-8")).hexdigest()
        data = timeline.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        report["timeline_sha256"] = digest
        if manifest.exists():
            attempt = load_json(manifest) or {}
            if not any(
                item.get("name") == timeline.name and item.get("sha256") == digest
                for item in attempt.get("artifacts") or [] if isinstance(item, dict)
            ):
                raise ValueError("Timeline does not match an artifact in the attempt manifest.")
        scenes = _scenes(data)
        report["scene_count"] = len(scenes)
        user = json.dumps({"synopsis": synopsis, "scene_timeline": scenes}, ensure_ascii=False)
        system = load_prompt("video_understanding.md")
        raw = provider.complete(system, user)
        report.update(_grade(raw, scenes, synopsis), status="success", scored=True)
    except Exception as exc:  # A failed input or provider response must not become a quality score.
        report["error"] = str(exc)
        if raw:
            raw_path = output.with_suffix(".raw.txt")
            storage.atomic_write_text(raw_path, raw)
            report["raw_output_file"] = raw_path.name

    storage.atomic_write_text(output, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    result = {key: report[key] for key in ("video", "status", "overall", "error") if key in report}
    result["output_file"] = str(output)
    return result
