"""Unattended experiment loop: variant → dual terminals → scores → notes."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from src.open_enricher.experiments.archive import (
    append_progress,
    experiment_dir,
    list_variants,
    write_json,
)
from src.open_enricher.experiments.config import (
    REPO_ROOT,
    VARIANTS_DIR,
    MachineBinding,
    select_machines,
)
from src.open_enricher.experiments.git_ops import (
    commit_all,
    copy_experiment_md,
    ensure_experiment_worktree,
    install_pipeline_spec,
    remove_worktree,
)
from src.open_enricher.experiments.machine_run import run_machine_pipeline


def _machine_label(machine: MachineBinding) -> str:
    labels = {
        "colab": "Colab (Stranger Things)",
        "kaggle": "Kaggle (Suits)",
    }
    return labels.get(machine.name, machine.name)


def _orchestrator_notes(machines: tuple[MachineBinding, ...]) -> str:
    described = " and ".join(_machine_label(machine) for machine in machines)
    if len(machines) == 1:
        finished = f"the {machines[0].name} run finishes"
    else:
        finished = "BOTH machines finish"
    return f"""# EXPERIMENT.md template (orchestrator)

Write three short sections after {finished}:

1. **Code change** — what the pipeline_spec / stage code changed for this experiment
2. **Execution** — outcome of {described}: errors, OOM, truncations
3. **Quality and next step** — one conclusion; what the next experiment should change

Scores stay in each machine's `score.json`.
"""


def _variant_id(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("experiment_id") or path.stem)


def _write_stub_experiment_md(
    worktree: Path,
    experiment_id: str,
    variant: dict[str, Any],
    machine_results: list[dict[str, Any]],
    machines: tuple[MachineBinding, ...],
) -> Path:
    lines = [
        f"# Experiment {experiment_id}",
        "",
        "## Code change",
        "",
        f"- Spec title: `{variant.get('title')}`",
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
    for result in machine_results:
        status = result.get("status")
        err = result.get("error")
        lines.append(
            f"- `{result.get('machine')}` / `{result.get('video')}`: **{status}**"
            + (f" — {err}" if err else "")
        )
    lines.extend(
        [
            "",
            "## Quality and next step",
            "",
            "_Orchestrator: replace this section with a conclusion from "
            + ", ".join(f"`runs/exp2/<id>/{machine.name}/score.json`" for machine in machines)
            + ", metadata JSON, timings, and worker logs. "
            "That conclusion is the only input for the next experiment's code changes._",
            "",
            _orchestrator_notes(machines),
            "",
        ]
    )
    path = worktree / "EXPERIMENT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _collect_scores(exp_dir: Path, machines: tuple[MachineBinding, ...]) -> dict[str, Any]:
    scores = {}
    for machine in machines:
        path = exp_dir / machine.name / "score.json"
        if path.is_file():
            scores[machine.name] = json.loads(path.read_text(encoding="utf-8"))
    return scores


def run_one_experiment(
    variant_path: Path,
    *,
    machines: tuple[MachineBinding, ...],
    timeout: int,
    gpu: str,
    keep_worktree: bool,
    skip_deploy: bool,
) -> dict[str, Any]:
    experiment_id = _variant_id(variant_path)
    variant = json.loads(variant_path.read_text(encoding="utf-8"))
    print(f"\n======== Experiment {experiment_id} ========\n", flush=True)

    worktree = ensure_experiment_worktree(experiment_id)
    # Prefer variant from worktree package if the harness was committed; else from source.
    local_variant = (
        worktree
        / "src"
        / "open_enricher"
        / "experiments"
        / "variants"
        / variant_path.name
    )
    source = local_variant if local_variant.is_file() else variant_path
    install_pipeline_spec(worktree, source)
    commit_all(worktree, f"Prepare pipeline spec for {experiment_id}")

    results: list[dict[str, Any]] = []
    print(
        "[INFO] Machines: " + ", ".join(machine.name for machine in machines),
        flush=True,
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(machines))) as pool:
        futures = {
            pool.submit(
                run_machine_pipeline,
                worktree,
                machine,
                experiment_id,
                timeout=timeout,
                gpu=gpu,
                skip_deploy=skip_deploy,
            ): machine
            for machine in machines
        }
        for future in concurrent.futures.as_completed(futures):
            machine = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "machine": machine.name,
                        "video": machine.video,
                        "experiment_id": experiment_id,
                        "status": "error",
                        "error": str(exc),
                    }
                )

    exp_dir = experiment_dir(experiment_id)
    write_json(exp_dir / "machine_results.json", results)
    write_json(exp_dir / "scores.json", _collect_scores(exp_dir, machines))
    write_json(exp_dir / "variant.json", variant)

    _write_stub_experiment_md(worktree, experiment_id, variant, results, machines)
    commit_all(worktree, f"Record EXPERIMENT.md for {experiment_id}")
    copy_experiment_md(worktree, exp_dir)

    # Copy worktree EXPERIMENT.md and mirror key notes onto base branch runs/.
    # The harness commits runs/ from the main checkout (BASE_BRANCH).
    append_progress(
        {
            "experiment_id": experiment_id,
            "machine": ",".join(machine.name for machine in machines),
            "step": "archived",
            "status": "ok",
            "results": [
                {"machine": r.get("machine"), "status": r.get("status")} for r in results
            ],
        }
    )

    if not keep_worktree:
        # Keep the branch; drop only the temporary worktree directory.
        remove_worktree(experiment_id)

    return {
        "experiment_id": experiment_id,
        "results": results,
        "runs_dir": str(exp_dir),
    }


def run_loop(
    *,
    only: list[str] | None = None,
    machines: list[str] | None = None,
    colab_target: str = "enricher",
    kaggle_target: str = "open-enricher",
    timeout: int = 10800,
    gpu: str = "T4",
    keep_worktree: bool = False,
    skip_deploy: bool = False,
) -> list[dict[str, Any]]:
    selected = select_machines(
        machines,
        colab_target=colab_target,
        kaggle_target=kaggle_target,
    )
    variants = list_variants(VARIANTS_DIR)
    if only:
        wanted = set(only)
        variants = [
            path
            for path in variants
            if path.stem in wanted or _variant_id(path) in wanted
        ]
    # Fixed first seven; 08-10 are hypotheses the orchestrator may rewrite.
    if not only:
        variants = [p for p in variants if p.name[:2] <= "07"]

    summaries: list[dict[str, Any]] = []
    for path in variants:
        summaries.append(
            run_one_experiment(
                path,
                machines=selected,
                timeout=timeout,
                gpu=gpu,
                keep_worktree=keep_worktree,
                skip_deploy=skip_deploy,
            )
        )
    write_json(REPO_ROOT / "runs" / "exp2" / "loop_summary.json", summaries)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the exp2 experiment loop. Each variant deploys the same commit to "
            "the selected machines (Colab and Kaggle by default)."
        )
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Run only these experiment ids or variant stems (e.g. 01-interval-llm).",
    )
    parser.add_argument(
        "--machines",
        nargs="+",
        choices=["colab", "kaggle"],
        help="Machines to run this round. Default is Colab and Kaggle.",
    )
    parser.add_argument("--colab-target", default="enricher")
    parser.add_argument("--kaggle-target", default="open-enricher")
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--gpu", default="T4")
    parser.add_argument(
        "--keep-worktree",
        action="store_true",
        help="Leave the temporary worktree on disk after the experiment.",
    )
    parser.add_argument(
        "--skip-deploy",
        action="store_true",
        help="Reuse an already-running worker (debug).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known variants and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        for path in list_variants(VARIANTS_DIR):
            print(_variant_id(path), path.name)
        return
    run_loop(
        only=args.only,
        machines=args.machines,
        colab_target=args.colab_target,
        kaggle_target=args.kaggle_target,
        timeout=args.timeout,
        gpu=args.gpu,
        keep_worktree=args.keep_worktree,
        skip_deploy=args.skip_deploy,
    )


if __name__ == "__main__":
    main()
