"""Compare short-term consolidation using one set of encoded frames."""

import json
import os
import sys
from pathlib import Path
from time import perf_counter

from open_enricher import storage
from open_enricher.adapters.visual_encode import encode_video
from open_enricher.runner import bootstrap_runtime
from open_enricher.stages.optional import _memory_frames, _scene_starts
from open_enricher.stages.pipeline_spec import load_pipeline_spec
from open_enricher.visual_memory import _compact_long_term, retrieve_scene_history, stream_memory


def main() -> None:
    video, shared, output = sys.argv[1:]
    bootstrap_runtime(env_file=Path(shared) / ".env", tee_log=False)
    timeline_file = storage.get_output_dir("bassl") / f"{video}_bassl.json"
    timeline = json.loads(timeline_file.read_text())
    os.environ["OPEN_ENRICHER_ATTEMPT_ID"] = timeline["attempt_id"]
    spec = load_pipeline_spec()
    budget = spec.budgets
    encoded = encode_video(
        storage.video_inputs() / video, spec, sample_fps=budget.memory_candidate_fps,
    )
    frames = _memory_frames(encoded.get("frames") or [])
    if encoded.get("status") != "success" or len(frames) <= budget.long_term_capacity:
        raise SystemExit(encoded.get("reason") or "Insufficient frames to compare both capacities")
    del encoded

    starts = _scene_starts(video)
    if len(starts) < 2:
        raise SystemExit("The comparison requires at least two scene starts")
    before = perf_counter()
    production = stream_memory(
        frames, short_capacity=budget.short_term_capacity, short_target=budget.short_term_target,
        long_capacity=budget.long_term_capacity,
    )["long_term"]
    production_seconds = perf_counter() - before
    before = perf_counter()
    direct = _compact_long_term(frames, budget.long_term_capacity)
    direct_seconds = perf_counter() - before

    for name, bank, seconds in (
        ("short-then-long", production, production_seconds),
        ("direct-long", direct, direct_seconds),
    ):
        entries = {entry.frame_id: entry.to_dict() for entry in bank}
        retrievals = retrieve_scene_history(frames, bank, starts)
        if not retrievals or not any(row["frames"] for row in retrievals):
            raise SystemExit(f"{name} produced no scene history")
        if len(entries) != budget.long_term_capacity or any(
            entries[frame["frame_id"]]["end"] >= row["start"]
            for row in retrievals for frame in row["frames"]
        ):
            raise SystemExit(f"{name} failed capacity or chronology checks")
        document = {
            "video": video, "status": "success", "attempt_id": timeline["attempt_id"],
            "mode": name, "candidate_count": len(frames), "compaction_seconds": seconds,
            "entries": list(entries.values()), "retrievals": retrievals,
        }
        (Path(output) / f"{name}.json").write_text(json.dumps(document, indent=2))
        print(f"{name}: {len(entries)} entries, {seconds:.2f}s compaction", flush=True)


if __name__ == "__main__":
    main()
