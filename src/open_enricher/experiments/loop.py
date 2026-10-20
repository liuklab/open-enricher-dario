"""Sequential local experiment loop.

For each pipeline variant JSON, point the process at that spec, run the
in-process enricher on the configured videos, archive artifacts under
``runs/local/<experiment_id>/``, and continue to the next variant. No
deploy, HTTP client, Colab, Kaggle, or Google Drive is involved.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from open_enricher.experiments.archive import (
    append_progress,
    experiment_dir,
    is_done,
    list_variants,
    merge_timings,
    video_archive_dir,
    write_json,
)
from open_enricher.experiments.config import (
    CACHE_ROOT,
    EXPERIMENT_PREFIX,
    PIPELINE_STAGES,
    REPO_ROOT,
    VARIANTS_DIR,
)
from open_enricher.storage import db


def _runner():
    """
    Lazy-import runner helpers that pull in heavy stage dependencies.

    Returns:
        The ``open_enricher.runner`` module.
    """
    from open_enricher import runner as runner_module

    return runner_module


def _variant_id(path: Path) -> str:
    """
    Read the experiment id from a variant file, falling back to the stem.

    Args:
        path: Path to a variant JSON document.

    Returns:
        Non-empty experiment id string.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("experiment_id") or path.stem)


def _load_variant(path: Path) -> dict[str, Any]:
    """
    Load a variant JSON object and drop private underscore keys.

    Args:
        path: Path to a variant JSON document.

    Returns:
        Dict suitable for ``load_pipeline_spec`` and for archiving.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Variant must be a JSON object: {path}")
    return {key: value for key, value in data.items() if not str(key).startswith("_")}


def _write_stub_experiment_md(
    exp_dir: Path,
    experiment_id: str,
    variant: dict[str, Any],
    video_results: list[dict[str, Any]],
) -> Path:
    """
    Write a short EXPERIMENT.md stub summarizing the local run.

    Args:
        exp_dir: Archive directory for this experiment.
        experiment_id: Stable experiment id.
        variant: Cleaned variant dict.
        video_results: Per-video pipeline summaries.

    Returns:
        Path to the written markdown file.
    """
    lines = [
        f"# Experiment {experiment_id}",
        "",
        "## Spec",
        "",
        f"- Title: `{variant.get('title')}`",
        f"- Segmentation: `{variant.get('segmentation')}`",
        f"- Dialogue/color routing: `{variant.get('dialogue_color_routing')}`",
        f"- VLM memory: `{variant.get('vlm_memory')}`",
        f"- Dialogue summary to VLM: `{variant.get('dialogue_summary_to_vlm')}`",
        f"- Rolling plot: `{variant.get('rolling_plot')}`",
        f"- Claim check: `{variant.get('claim_check')}`",
        "",
        "## Execution",
        "",
    ]
    for result in video_results:
        status = result.get("status")
        err = result.get("error")
        lines.append(
            f"- `{result.get('video')}`: **{status}**"
            + (f" - {err}" if err else "")
        )
    lines.extend(
        [
            "",
            "## Quality",
            "",
            "See each video folder's `score.json` (deterministic judger report).",
            "",
        ]
    )
    path = exp_dir / "EXPERIMENT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _collect_scores(exp_dir: Path, videos: list[str]) -> dict[str, Any]:
    """
    Load archived judger scores for each video under an experiment folder.

    Args:
        exp_dir: Archive directory for this experiment.
        videos: Video filenames that were processed.

    Returns:
        Mapping of video filename to score document when present.
    """
    scores: dict[str, Any] = {}
    for video in videos:
        path = exp_dir / video / "score.json"
        if path.is_file():
            scores[video] = json.loads(path.read_text(encoding="utf-8"))
    return scores


def _activate_variant(variant_path: Path, experiment_id: str) -> None:
    """
    Point the process at one variant and experiment output namespace.

    Args:
        variant_path: Absolute path to the variant JSON file.
        experiment_id: Id used under ``outputs/<prefix>/<id>/``.

    Side effects:
        Sets ``OPEN_ENRICHER_PIPELINE_SPEC``, ``OPEN_ENRICHER_EXPERIMENT_ID``,
        and ``OPEN_ENRICHER_EXPERIMENT_PREFIX``, then refreshes ``db``.
    """
    os.environ["OPEN_ENRICHER_PIPELINE_SPEC"] = str(variant_path.resolve())
    os.environ["OPEN_ENRICHER_EXPERIMENT_ID"] = experiment_id
    os.environ["OPEN_ENRICHER_EXPERIMENT_PREFIX"] = EXPERIMENT_PREFIX
    db.refresh()
    db.setup_database()


def run_one_experiment(
    variant_path: Path,
    *,
    videos: list[str],
    stages: tuple[str, ...] = PIPELINE_STAGES,
    use_cache: bool = True,
    skip_done: bool = True,
) -> dict[str, Any]:
    """
    Run the full local pipeline for one variant across the given videos.

    Args:
        variant_path: Path to the variant JSON file.
        videos: Video filenames under the data lake ``video_inputs``.
        stages: Ordered stage names to execute.
        use_cache: Reuse shared audio/color cache when available.
        skip_done: Skip the experiment when progress already marks it archived.

    Returns:
        Summary dict with experiment_id, per-video results, and archive path.
    """
    experiment_id = _variant_id(variant_path)
    variant = _load_variant(variant_path)
    print(f"\n======== Experiment {experiment_id} ========\n", flush=True)

    if skip_done and is_done(experiment_id, "archived"):
        print(f"[INFO] Skipping {experiment_id}: already archived", flush=True)
        return {
            "experiment_id": experiment_id,
            "status": "skipped_done",
            "results": [],
            "runs_dir": str(experiment_dir(experiment_id)),
        }

    _activate_variant(variant_path, experiment_id)
    runner = _runner()
    registry = runner.build_stages()
    video_results: list[dict[str, Any]] = []

    for video in videos:
        print(f"[INFO] Pipeline for video {video}", flush=True)
        try:
            summary = runner.run_pipeline(
                video,
                stages=stages,
                stage_registry=registry,
                cache_root=CACHE_ROOT if use_cache else None,
                use_cache=use_cache,
            )
        except Exception as exc:  # noqa: BLE001 - keep the loop moving
            summary = {
                "video": video,
                "status": "error",
                "error": str(exc),
                "stages": [],
            }

        dest = video_archive_dir(experiment_id, video)
        archived = runner.archive_artifacts(video, dest, stages=stages)
        summary["archived_files"] = archived

        score = runner.load_judger_score(video)
        if score is not None:
            write_json(dest / "score.json", score)
            summary["score"] = {
                "overall": score.get("overall"),
                "usable": score.get("usable"),
            }

        timing_paths = list(dest.glob(f"{video}_*_timings.json"))
        write_json(dest / "timings.json", merge_timings(timing_paths))
        write_json(dest / "result.json", summary)
        video_results.append(summary)

        append_progress(
            {
                "experiment_id": experiment_id,
                "video": video,
                "step": "pipeline",
                "status": "ok" if summary.get("status") == "success" else "error",
                "detail": summary.get("status"),
            }
        )

    exp_dir = experiment_dir(experiment_id)
    write_json(exp_dir / "video_results.json", video_results)
    write_json(exp_dir / "scores.json", _collect_scores(exp_dir, videos))
    write_json(exp_dir / "variant.json", variant)
    _write_stub_experiment_md(exp_dir, experiment_id, variant, video_results)

    overall = "ok"
    if any(row.get("status") != "success" for row in video_results):
        overall = "error"

    append_progress(
        {
            "experiment_id": experiment_id,
            "step": "archived",
            "status": overall,
            "results": [
                {"video": row.get("video"), "status": row.get("status")}
                for row in video_results
            ],
        }
    )

    return {
        "experiment_id": experiment_id,
        "status": overall,
        "results": video_results,
        "runs_dir": str(exp_dir),
    }


def run_loop(
    *,
    only: list[str] | None = None,
    from_id: str | None = None,
    videos: list[str] | None = None,
    stages: tuple[str, ...] = PIPELINE_STAGES,
    use_cache: bool = True,
    skip_done: bool = True,
) -> list[dict[str, Any]]:
    """
    Run selected variants in sequence on the local GPU host.

    Args:
        only: Optional list of experiment ids or variant stems to run.
        from_id: When set without ``only``, start at this id and continue.
        videos: Optional explicit video filenames. Defaults to every file in
            ``video_inputs``.
        stages: Ordered stage names for each run.
        use_cache: Reuse shared audio/color cache when available.
        skip_done: Skip experiments already marked archived in progress.jsonl.

    Returns:
        List of per-experiment summary dicts. Also writes
        ``runs/local/loop_summary.json``.
    """
    runner = _runner()
    runner.bootstrap_runtime(tee_log=False)
    selected_videos = runner.list_videos(videos)

    variants = list_variants(VARIANTS_DIR)
    if only:
        wanted = set(only)
        variants = [
            path
            for path in variants
            if path.stem in wanted or _variant_id(path) in wanted
        ]
    elif from_id:
        start = None
        for index, path in enumerate(variants):
            if path.stem == from_id or _variant_id(path) == from_id:
                start = index
                break
        if start is None:
            raise ValueError(f"Unknown --from experiment id: {from_id}")
        variants = variants[start:]

    if not variants:
        raise ValueError("No variants selected to run")

    print(
        "[INFO] Videos: "
        + ", ".join(selected_videos)
        + f" | variants: {len(variants)}",
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    for path in variants:
        summaries.append(
            run_one_experiment(
                path,
                videos=selected_videos,
                stages=stages,
                use_cache=use_cache,
                skip_done=skip_done,
            )
        )

    write_json(REPO_ROOT / "runs" / "local" / "loop_summary.json", summaries)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the local experiment loop.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run Open-Enricher experiment variants locally in sequence. "
            "Each variant sets OPEN_ENRICHER_PIPELINE_SPEC and runs the "
            "in-process pipeline on the videos under the local data lake."
        )
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Run only these experiment ids or variant stems (e.g. 01-interval-llm).",
    )
    parser.add_argument(
        "--from",
        dest="from_id",
        help="Start at this experiment id and continue through later variants.",
    )
    parser.add_argument(
        "--video",
        nargs="+",
        dest="videos",
        help="Video filenames under video_inputs. Default: every file there.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(PIPELINE_STAGES),
        help="Subset of stages to run. Default: all stages including judger.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not restore or update the shared audio/color cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even when progress.jsonl already marks them done.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known variants and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    CLI entrypoint for ``python -m open_enricher`` and the experiments module.

    Args:
        argv: Optional argument list for tests. Defaults to ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        for path in list_variants(VARIANTS_DIR):
            print(_variant_id(path), path.name)
        return

    stages = tuple(args.stages) if args.stages else PIPELINE_STAGES
    run_loop(
        only=args.only,
        from_id=args.from_id,
        videos=args.videos,
        stages=stages,
        use_cache=not args.no_cache,
        skip_done=not args.force,
    )


if __name__ == "__main__":
    main()
