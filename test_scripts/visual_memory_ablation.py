"""Replay VLM and profile generation with matched frame counts per scene."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from open_enricher.runner import bootstrap_runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("mode", choices=("current", "recent", "retrieved", "grounded"))
    parser.add_argument("dataset", type=Path)
    parser.add_argument("shared", type=Path)
    args = parser.parse_args()
    bootstrap_runtime(env_file=args.shared / ".env", tee_log=False)

    from transformers import set_seed

    from open_enricher import storage
    from open_enricher.graph import relations_for_interval
    from open_enricher.stages.llm import LLMStage
    from open_enricher.stages.pipeline_spec import load_pipeline_spec
    from open_enricher.stages.vlm_core.config import build_configs
    from open_enricher.stages.vlm_core import planned
    from open_enricher.stages.vlm_core.segments import run_segmented

    def read_stage(stage: str, suffix: str) -> dict:
        return json.loads(
            (storage.get_output_dir(stage) / f"{args.video}_{suffix}.json").read_text()
        )

    video = storage.video_inputs() / args.video
    dialogue = read_stage("dialogue", "dialogue")
    graph = read_stage("character_graph", "graph")
    memory = read_stage("visual_memory", "memory")
    os.environ["OPEN_ENRICHER_ATTEMPT_ID"] = dialogue["attempt_id"]
    if memory["status"] != "success" or any(
        artifact["attempt_id"] != dialogue["attempt_id"] for artifact in (graph, memory)
    ):
        raise SystemExit("The replay requires matched graph, dialogue, and memory artifacts")

    spec = load_pipeline_spec()
    scenes, plan = planned.prepare_scenes(
        video, args.video, dialogue["scenes"], spec, sample_fps=2, extract_frames=False,
    )
    memory_end = {entry["frame_id"]: entry["end"] for entry in memory["entries"]}
    audit = []
    for scene, group in zip(scenes, plan["groups"], strict=True):
        start, end = group["start"], group["end"]
        current = list(group["timestamps"])
        room = max(0, spec.budgets.max_qwen_frames - len(current))
        room -= room % 2
        eligible = {
            frame["timestamp"]
            for query in memory["retrievals"]
            if abs(query["start"] - start) <= 0.05
            for frame in query["frames"]
            if memory_end[frame["frame_id"]] < start
        }
        history = [
            stamp for stamp in planned._history_stamps(args.video, start, room)
            if stamp in eligible and stamp not in current
        ]
        if any(stamp >= start for stamp in history):
            raise SystemExit("The memory bank returned a frame from the current or future scene")
        if args.mode == "current":
            # Keep all original current frames and add one distinct sample per freed slot.
            extra = []
            for index in range(1, len(history) + len(current) + 2):
                stamp = round(start + (end - start) * index / (len(history) + len(current) + 2), 3)
                if start <= stamp < end and stamp not in current and stamp not in extra:
                    extra.append(stamp)
                if len(extra) == len(history):
                    break
            if history and len(extra) != len(history):
                raise SystemExit(f"Could not match frame counts for {group['call_id']}")
            stamps = sorted([*current, *(extra if history else [])])
            earlier = []
        elif args.mode == "recent":
            earlier = [round(max(0.0, start - index), 3) for index in range(len(history), 0, -1)]
            if any(stamp >= start for stamp in earlier) or len(set(earlier)) != len(earlier):
                raise SystemExit(f"Could not sample distinct earlier frames for {group['call_id']}")
            stamps = earlier + current
        else:
            earlier = history
            stamps = history + current
        if len(stamps) != len(current) + len(history):
            raise SystemExit("Mismatched frame budget")
        if stamps:
            paths = planned._grab(video, stamps, args.dataset / "frames" / group["call_id"])
            if len(paths) != len(stamps):
                raise SystemExit(f"Frame extraction failed for {group['call_id']}")
            scene["image_paths"] = paths
        current_stamps = stamps[len(earlier):]
        context = []
        if earlier:
            context.append(f"Earlier frames from this title, in order: {', '.join(f'{t:.1f}s' for t in earlier)}.")
        if current_stamps:
            context.append(f"Current frames at source times: {', '.join(f'{t:.1f}s' for t in current_stamps)}.")
        if args.mode == "grounded":
            relations = relations_for_interval(graph["edges"], start, end)
            context.append(
                "Describe events from the current frames. Earlier frames may help identify recurring settings "
                "or objects and establish continuity. Use only scene-local graph observations, and treat "
                "speaker labels as local references. Co-presence provides evidence of visibility during "
                "speech. Leave recipient identity and relationship unresolved when the evidence is ambiguous."
            )
            if relations:
                context.append("Scene-local observations: " + json.dumps(relations[:8]))
        else:
            context.append(f"Supported character relations: {graph['account']}")
        scene["prompt_extra"] = "\n".join(context)
        audit.append({"scene": group["call_id"], "earlier": earlier, "current": current_stamps})
    if not any(row["earlier"] for row in audit) and args.mode != "current":
        raise SystemExit("No scene can exercise visual history")
    (args.dataset / "frames.json").write_text(json.dumps(audit, indent=2))
    set_seed(49)
    model, config = build_configs()
    raw, parsed, valid, _, _ = run_segmented(
        video, args.video, model, config, scenes,
        experiment_id=spec.experiment_id, vlm_memory_scenes=spec.vlm_memory_scenes,
        continue_on_error=True,
    )
    vlm = storage.get_output_dir("vlm")
    (vlm / f"{args.video}_vlm.json").write_text(json.dumps(parsed, indent=2))
    (vlm / f"{args.video}_vlm_raw.txt").write_text(raw)
    print(f"VLM JSON coverage: {parsed['segments_parsed']}/{parsed['segments_total']}", flush=True)
    if args.mode == "grounded":
        graph["account"] = (
            "Use scene metadata and timestamped dialogue to ground distinctive tags and the hook. "
            "Earlier frames support continuity only. Use character edges for the observed behavior "
            "they describe. Leave names and intended addressees unresolved unless explicitly supported.\n"
            + graph["account"]
        )
        graph_path = storage.get_output_dir("character_graph") / f"{args.video}_graph.json"
        graph_path.write_text(json.dumps(graph), encoding="utf-8")
    result = LLMStage().run(args.video)
    print(json.dumps({key: result.get(key) for key in ("status", "output_file", "error")}), flush=True)
    if result["status"] != "success":
        raise SystemExit(1)
    if not valid:
        print("VLM returned partial evidence. Review parsing coverage before comparing profiles.", flush=True)


if __name__ == "__main__":
    main()
