"""Optional stages for timeline, BaSSL, the character graph and memory.

Each ``run`` returns a status dict and does not raise. Heavy imports
happen inside the adapters' child processes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from open_enricher import storage
from open_enricher.adapters.bassl import segment_video
from open_enricher.adapters.face_speech import active_speakers, detect_tracks
from open_enricher.adapters.visual_encode import encode_video
from open_enricher.fingerprint import current_stamp, reusable
from open_enricher.graph import (
    CharacterNode,
    Edge,
    TrackObservation,
    apply_alias,
    assign_character_ids,
    compact_account,
    dedupe_edges,
    interpret_mention,
    link_observations,
    speaker_edges,
    visible_characters,
    visible_speech_edges,
)
from open_enricher.logging_setup import get_logger
from open_enricher.manifest import load_json, matching_artifact
from open_enricher.stages.pipeline_spec import load_pipeline_spec
from open_enricher.stages.vlm_core.video import probe_duration_seconds
from open_enricher.timeline import Interval, fixed_windows, validate_scene_bounds
from open_enricher.visual_memory import (
    MemoryEntry,
    stream_memory,
    temporal_history,
)

logger = get_logger(__name__)


def _write(path: Path, document: dict[str, Any]) -> None:
    storage.atomic_write_text(
        path,
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
    )


def _base(video_filename: str) -> dict[str, Any]:
    document = {"video": video_filename}
    document.update(current_stamp())
    return document


def _reuse(path: Path) -> dict[str, Any] | None:
    document = load_json(path)
    if document is None or not reusable(document):
        return None
    stamp = current_stamp()
    if stamp["attempt_id"] and document.get("attempt_id") != stamp["attempt_id"]:
        document["attempt_id"] = stamp["attempt_id"]
        _write(path, document)
    return {
        "stage": document.get("stage"),
        "status": document.get("status") or "success",
        "reused": True,
        "output_file": path.name,
    }


def _intervals_from(document: dict[str, Any] | None, duration: float, interval: int) -> list[Interval]:
    rows = [] if document is None else document.get("intervals") or []
    intervals: list[Interval] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            intervals.append(
                Interval(
                    id=str(row["id"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                    kind=str(row.get("kind") or "window"),
                    source=str(row.get("source") or "fixed"),
                    parent_id=row.get("parent_id"),
                    scene_ids=tuple(row.get("scene_ids") or []),
                    repaired=bool(row.get("repaired")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if intervals:
        return intervals
    return fixed_windows(duration, interval)


class TimelineStage:
    """Probe the source once and write fixed windows before any model runs."""

    name = "timeline"

    def run(self, video_filename: str) -> dict[str, Any]:
        path = storage.get_output_dir(self.name) / f"{video_filename}_timeline.json"
        cached = _reuse(path)
        if cached is not None:
            return cached
        video = storage.video_inputs() / video_filename
        spec = load_pipeline_spec()
        try:
            duration = probe_duration_seconds(video)
            intervals = fixed_windows(duration, spec.interval_seconds)
        except Exception as exc:  # noqa: BLE001 - terminal for this video
            logger.warning("Timeline probe failed for %s: %s", video_filename, exc)
            return {"stage": self.name, "status": "failed", "error": str(exc)}
        document = _base(video_filename)
        document.update(
            {
                "stage": self.name,
                "status": "success",
                "duration": duration,
                "interval_seconds": spec.interval_seconds,
                "intervals": [item.to_dict() for item in intervals],
                "notes": [],
            }
        )
        _write(path, document)
        return {"stage": self.name, "status": "success", "output_file": path.name}


class BasslStage:
    """Replace fixed windows when BaSSL returns a timeline that covers the file."""

    name = "bassl"

    def run(self, video_filename: str) -> dict[str, Any]:
        path = storage.get_output_dir(self.name) / f"{video_filename}_bassl.json"
        cached = _reuse(path)
        if cached is not None:
            return cached
        spec = load_pipeline_spec()
        video = storage.video_inputs() / video_filename
        timeline_path = storage.get_output_dir("timeline") / f"{video_filename}_timeline.json"
        try:
            duration = probe_duration_seconds(video)
        except Exception as exc:  # noqa: BLE001
            return {"stage": self.name, "status": "fallback", "error": str(exc)}
        raw = segment_video(video, duration, spec)
        scenes = raw.get("scenes") if raw.get("status") == "success" else []
        intervals, notes, replaced = validate_scene_bounds(
            scenes,
            duration,
            interval_seconds=spec.interval_seconds,
        )
        if raw.get("status") != "success":
            replaced = True
            notes.append(str(raw.get("reason") or "BaSSL unavailable"))
            intervals = fixed_windows(duration, spec.interval_seconds)
        status = "fallback" if replaced else "success"
        if not replaced and any(item.repaired for item in intervals):
            status = "degraded"
        document = _base(video_filename)
        document.update(
            {
                "stage": self.name,
                "status": status,
                "duration": duration,
                "notes": notes,
                "replaced_all": replaced,
                "intervals": [item.to_dict() for item in intervals],
                "shots": raw.get("shots") or [],
                "reason": raw.get("reason"),
            }
        )
        _write(path, document)
        timeline = _base(video_filename)
        timeline.update(
            {
                "stage": "timeline",
                "status": "success",
                "duration": duration,
                "interval_seconds": spec.interval_seconds,
                "intervals": document["intervals"],
                "notes": notes,
                "source_stage": self.name,
            }
        )
        _write(timeline_path, timeline)
        return {
            "stage": self.name,
            "status": status,
            "output_file": path.name,
            "reason": raw.get("reason"),
        }


class CharacterGraphStage:
    """Build a per-video graph from faces, speech and name mentions."""

    name = "character_graph"

    def run(self, video_filename: str) -> dict[str, Any]:
        path = storage.get_output_dir(self.name) / f"{video_filename}_graph.json"
        cached = _reuse(path)
        if cached is not None:
            return cached
        spec = load_pipeline_spec()
        video = storage.video_inputs() / video_filename
        reasons: list[str] = []
        sample_fps = min(float(spec.budgets.face_sample_fps), 1.0)
        logger.info("Detecting faces for %s at %.2f fps", video_filename, sample_fps)
        face = detect_tracks(video, spec, sample_fps=sample_fps)
        if face.get("status") != "success":
            reasons.append(str(face.get("reason") or "faces unavailable"))
        observations = _observations(face.get("tracks") or [])
        pooled = link_observations(observations)
        logger.info(
            "Face detection returned %s samples, pooled into %s tracks (%s)",
            len(observations),
            len(pooled),
            face.get("status"),
        )
        nodes, decisions = assign_character_ids(
            pooled,
            threshold=spec.budgets.match_threshold,
            margin=spec.budgets.match_margin,
        )
        lines = _audio_lines(video_filename)
        dialogue_spans = [
            (float(line["start_timestamp"]), float(line["end_timestamp"]))
            for line in lines
            if isinstance(line, dict)
            and line.get("speaker") not in {None, "", "UNKNOWN"}
            and isinstance(line.get("start_timestamp"), (int, float))
            and isinstance(line.get("end_timestamp"), (int, float))
            and line["end_timestamp"] > line["start_timestamp"]
        ]
        speech = active_speakers(
            video,
            spec,
            face.get("tracks") or [],
            max_tracks=96,
            dialogue_spans=dialogue_spans or None,
        )
        scores = speech.get("scores") or []
        if speech.get("status") != "success":
            reasons.append(str(speech.get("reason") or "TalkNet unavailable"))
        edges = _mention_edges(video_filename, nodes)
        speech_links = speaker_edges(scores, decisions, lines)
        kept = visible_characters(nodes)
        kept_ids = {node.character_id for node in kept}
        edges.extend(edge for edge in speech_links if edge.source in kept_ids)
        edges.extend(visible_speech_edges(observations, decisions, speech_links, kept_ids))
        edges = dedupe_edges(edges)
        node_rows = [
            {
                "character_id": node.character_id,
                "alias": node.alias,
                "provisional": node.provisional,
                "tracks": list(node.tracks),
                "weight": node.weight,
                "samples": node.samples,
            }
            for node in kept
        ]
        edge_rows = [edge.to_dict() for edge in edges]
        if kept and not reasons:
            status = "success"
        elif kept or edge_rows:
            status = "degraded"
        else:
            status = "fallback"
        kept_decisions = [
            row for row in decisions if row.get("character_id") in kept_ids
        ]
        document = _base(video_filename)
        document.update(
            {
                "stage": self.name,
                "status": status,
                "reasons": reasons,
                "face_sample_fps": sample_fps,
                "face_provider": face.get("provider"),
                "face_intra_op_threads": face.get("intra_op_threads"),
                "face_samples": len(observations),
                "pooled_tracks": len(pooled),
                "dropped_nodes": len(nodes) - len(kept),
                "speaker_window_scores": [
                    {key: value for key, value in score.items() if key != "frame_scores"}
                    for score in scores
                ],
                "nodes": node_rows,
                "edges": edge_rows,
                "decisions": kept_decisions,
                "account": compact_account(node_rows, edge_rows),
            }
        )
        _write(path, document)
        return {"stage": self.name, "status": status, "output_file": path.name, "reasons": reasons}


class VisualMemoryStage:
    """Encode candidates and consolidate them. An empty bank is a fallback."""

    name = "visual_memory"

    def run(self, video_filename: str) -> dict[str, Any]:
        path = storage.get_output_dir(self.name) / f"{video_filename}_memory.json"
        cached = _reuse(path)
        if cached is not None:
            return cached
        spec = load_pipeline_spec()
        video = storage.video_inputs() / video_filename
        encoded = encode_video(
            video,
            spec,
            sample_fps=spec.budgets.memory_candidate_fps,
        )
        frames = _memory_frames(encoded.get("frames") or [])
        if not frames or encoded.get("status") != "success":
            document = _base(video_filename)
            document.update(
                {
                    "stage": self.name,
                    "status": "fallback",
                    "reason": encoded.get("reason") or "visual memory bank is empty",
                    "entries": [],
                    "retrievals": [],
                    "checkpoint_present": bool(encoded.get("checkpoint_present")),
                }
            )
            _write(path, document)
            return {
                "stage": self.name,
                "status": "fallback",
                "output_file": path.name,
                "reason": document["reason"],
            }
        bank = stream_memory(
            frames,
            short_capacity=spec.budgets.short_term_capacity,
            short_target=spec.budgets.short_term_target,
            long_capacity=spec.budgets.long_term_capacity,
        )
        starts = _scene_starts(video_filename)
        retrievals = []
        for start in starts:
            query = next((frame for frame in frames if frame.timestamp >= start), None)
            if query is None:
                picked: list[dict[str, Any]] = []
            else:
                from open_enricher.visual_memory import retrieve

                picked = retrieve(query, bank["long_term"], limit=4, scene_start=start)
            if not picked:
                picked = temporal_history(
                    [entry.to_dict() for entry in bank["long_term"]],
                    start,
                    limit=4,
                )
            retrievals.append({"start": start, "frames": picked})
        document = _base(video_filename)
        document.update(
            {
                "stage": self.name,
                "status": "success",
                "entries": [entry.to_dict() for entry in bank["long_term"]],
                "spans": bank["spans"],
                "retrievals": retrievals,
                "long_term_compacted": bank["long_term_compacted"],
            }
        )
        _write(path, document)
        return {"stage": self.name, "status": "success", "output_file": path.name}


def _observations(rows: list[dict[str, Any]]) -> list[TrackObservation]:
    observations: list[TrackObservation] = []
    for row in rows:
        embedding = row.get("embedding")
        if not isinstance(embedding, list) or not embedding:
            continue
        observations.append(
            TrackObservation(
                track_id=str(row.get("track_id")),
                embedding=[float(value) for value in embedding],
                start=float(row.get("start") or 0),
                end=float(row.get("end") or 0),
                shot_id=str(row.get("shot_id") or ""),
                co_visible=tuple(str(item) for item in row.get("co_visible") or []),
            )
        )
    return observations


def _audio_lines(video_filename: str) -> list[Any]:
    path = storage.get_output_dir("audio") / f"{video_filename}_audio.json"
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def _mention_edges(video_filename: str, nodes: list[CharacterNode]) -> list[Edge]:
    lines = _audio_lines(video_filename)
    edges: list[Edge] = []
    speaker_node = nodes[0] if len(nodes) == 1 else None
    for index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        text = str(line.get("text") or "")
        role, name = interpret_mention(text)
        if name is None:
            continue
        start = float(line.get("start_timestamp") or 0)
        end = float(line.get("end_timestamp") or start)
        evidence = str(line.get("id") or index)
        if role == "self_introduction" and speaker_node is not None:
            apply_alias(speaker_node, role, name)
            edges.append(
                Edge("speaks", speaker_node.character_id, name, start, end, [evidence], None, "supported")
            )
        elif role == "addressee":
            edges.append(Edge("addresses", "speaker", name, start, end, [evidence], None, "unresolved"))
        else:
            edges.append(Edge("mentions", "utterance", name, start, end, [evidence], None, "unresolved"))
    return edges


def _memory_frames(rows: list[dict[str, Any]]) -> list[MemoryEntry]:
    frames: list[MemoryEntry] = []
    for row in rows:
        tokens = row.get("tokens")
        if not isinstance(tokens, list) or not tokens:
            continue
        frames.append(
            MemoryEntry(
                frame_id=str(row.get("frame_id")),
                timestamp=float(row.get("timestamp") or 0),
                tokens=tokens,
            )
        )
    return frames


def _scene_starts(video_filename: str) -> list[float]:
    dialogue = matching_artifact(
        storage.get_output_dir("dialogue") / f"{video_filename}_dialogue.json"
    )
    if dialogue is None:
        return []
    starts: list[float] = []
    for scene in dialogue.get("scenes") or []:
        if isinstance(scene, dict):
            try:
                starts.append(float(scene.get("start_seconds")))
            except (TypeError, ValueError):
                continue
    return starts
