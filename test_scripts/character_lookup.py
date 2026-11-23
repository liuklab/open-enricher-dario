"""Build a clip-local face and speaker lookup from archived diarization."""

import json
import sys
from collections import defaultdict
from pathlib import Path

from open_enricher.adapters.face_speech import active_speakers, detect_tracks
from open_enricher.graph import (
    align_speaker_scores,
    assign_character_ids,
    interpret_mention,
    link_observations,
    link_speaker,
    visible_characters,
)
from open_enricher.stages.optional import _observations
from open_enricher.stages.pipeline_spec import load_pipeline_spec


def main() -> None:
    video, transcript, output = map(Path, sys.argv[1:])
    spec = load_pipeline_spec()
    lines = json.loads(transcript.read_text(encoding="utf-8"))
    face = detect_tracks(video, spec, sample_fps=1.0)
    if face.get("status") != "success":
        raise SystemExit(face.get("reason") or "Face detection failed")
    tracks = face["tracks"]
    observations = _observations(tracks)
    nodes, decisions = assign_character_ids(
        link_observations(observations),
        threshold=spec.budgets.match_threshold,
        margin=spec.budgets.match_margin,
    )
    spans = [
        (line["start_timestamp"], line["end_timestamp"])
        for line in lines
        if line.get("speaker") not in {None, "", "UNKNOWN"}
        and line["start_timestamp"] < line["end_timestamp"]
    ]
    speech = active_speakers(video, spec, tracks, max_tracks=96, dialogue_spans=spans)
    if speech.get("status") != "success":
        raise SystemExit(speech.get("reason") or "Active speaker detection failed")
    alignments = align_speaker_scores(speech["scores"], decisions, lines)
    links = link_speaker(alignments)
    speakers = defaultdict(list)
    for speaker, character in links.items():
        speakers[character].append(speaker)

    times = {row.track_id: row.start for row in observations}
    appearances = defaultdict(set)
    for decision in decisions:
        appearances[decision["character_id"]].update(
            times[track] for track in decision.get("source_track_ids", ()) if track in times
        )

    names = defaultdict(set)
    for line in lines:
        role, name = interpret_mention(line["text"])
        if role == "self_introduction" and name:
            names[line.get("speaker")].add(name)
    rows = []
    for node in visible_characters(nodes):
        character = node.character_id
        labels = sorted(speakers[character])
        candidates = names[labels[0]] if len(labels) == 1 else set()
        rows.append({
            "character_id": character,
            "provisional": node.provisional,
            "appearance_times": sorted(appearances[character]),
            "speakers": labels,
            "name_candidate": next(iter(candidates)) if len(candidates) == 1 else None,
        })
    output.write_text(json.dumps({"video": video.name, "characters": rows}, indent=2), encoding="utf-8")
    print(f"Published {len(rows)} lookup rows.")


if __name__ == "__main__":
    main()
