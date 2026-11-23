"""Export anonymous face clusters and their clip-local appearance times."""

import json
import sys
from collections import defaultdict
from pathlib import Path

from open_enricher.adapters.face_speech import detect_tracks
from open_enricher.graph import (
    assign_character_ids,
    link_observations,
    visible_characters,
)
from open_enricher.stages.optional import _observations
from open_enricher.stages.pipeline_spec import load_pipeline_spec


def main() -> None:
    video, output = map(Path, sys.argv[1:])
    spec = load_pipeline_spec()
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
    times = {row.track_id: row.start for row in observations}
    appearances = defaultdict(set)
    for decision in decisions:
        appearances[decision["character_id"]].update(
            times[track] for track in decision.get("source_track_ids", ()) if track in times
        )

    rows = []
    for node in visible_characters(nodes):
        character = node.character_id
        rows.append({
            "character_id": character,
            "provisional": node.provisional,
            "appearance_times": sorted(appearances[character]),
        })
    output.write_text(json.dumps({"video": video.name, "characters": rows}, indent=2), encoding="utf-8")
    print(f"Published {len(rows)} lookup rows.")


if __name__ == "__main__":
    main()
