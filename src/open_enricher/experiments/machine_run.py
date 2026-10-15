"""Deploy and run one machine terminal (Colab or Kaggle) for an experiment."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import requests

# Packager writes a fixed tarball name; serialize deploys from a shared worktree.
_DEPLOY_LOCK = threading.Lock()

from src.open_enricher.client.network import PipelineClient, setup_server_connection
from src.open_enricher.experiments.archive import (
    append_progress,
    cache_dir,
    is_done,
    machine_dir,
    merge_timings,
    write_json,
)
from src.open_enricher.experiments.benchmark import save_score, score_profile
from src.open_enricher.experiments.config import PIPELINE_STAGES, REPO_ROOT, MachineBinding


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def _cli_log_snapshot(machine: MachineBinding, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if machine.mode == "colab-cli":
            result = subprocess.run(
                ["colab", "status", "-s", machine.target],
                check=False,
                text=True,
                capture_output=True,
            )
            dest.write_text(
                (result.stdout or "") + "\n" + (result.stderr or ""),
                encoding="utf-8",
            )
        elif machine.mode == "kaggle":
            from deployment.kaggle import kaggle_username

            kernel = f"{kaggle_username()}/{machine.target}"
            result = subprocess.run(
                ["kaggle", "kernels", "logs", kernel],
                check=False,
                text=True,
                capture_output=True,
            )
            dest.write_text(
                (result.stdout or "") + "\n" + (result.stderr or ""),
                encoding="utf-8",
            )
    except Exception as exc:
        dest.write_text(f"cli log snapshot failed: {exc}\n", encoding="utf-8")


def _download_artifact(
    base_url: str,
    stage: str,
    file_name: str,
    dest: Path,
    timeout: int = 120,
) -> bool:
    url = f"{base_url}/artifact/{stage}/{file_name}"
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(response.content)
        return True
    except requests.RequestException:
        return False


def _fetch_worker_log(base_url: str, dest: Path) -> None:
    try:
        response = requests.get(f"{base_url}/log", params={"tail": 200000}, timeout=60)
        response.raise_for_status()
        dest.write_text(response.text, encoding="utf-8")
    except requests.RequestException as exc:
        dest.write_text(f"failed to fetch worker log: {exc}\n", encoding="utf-8")


def _restore_cached_stage(
    base_url: str,
    stage: str,
    video: str,
    cache_path: Path,
) -> bool:
    if not cache_path.is_file():
        return False
    document = json.loads(cache_path.read_text(encoding="utf-8"))
    response = requests.post(
        f"{base_url}/restore/{stage}",
        json={"video_name": video, "document": document},
        timeout=120,
    )
    response.raise_for_status()
    return True


def _cache_stage_result(machine: str, video: str, stage: str, document: Any) -> Path:
    path = cache_dir(machine) / f"{video}_{stage}.json"
    write_json(path, document)
    return path


def deploy_worker(
    worktree: Path,
    machine: MachineBinding,
    experiment_id: str,
    *,
    gpu: str = "T4",
) -> None:
    with _DEPLOY_LOCK:
        env = os.environ.copy()
        env["OPEN_ENRICHER_EXPERIMENT_ID"] = experiment_id
        env["OPEN_ENRICHER_EXPERIMENT_PREFIX"] = "exp2"
        # Keep code dataset slug configurable for a parallel Codex worktree.
        env.setdefault("OPEN_ENRICHER_CODE_SLUG", "open-enricher-code")

        # Worktrees do not include the gitignored .env; copy it from the primary repo.
        env_path = worktree / ".env"
        source_env = REPO_ROOT / ".env"
        if source_env.is_file():
            text = source_env.read_text(encoding="utf-8")
        elif env_path.is_file():
            text = env_path.read_text(encoding="utf-8")
        else:
            text = ""
        lines = [
            line
            for line in text.splitlines()
            if not line.startswith("OPEN_ENRICHER_EXPERIMENT_")
        ]
        lines.append(f"OPEN_ENRICHER_EXPERIMENT_ID={experiment_id}")
        lines.append("OPEN_ENRICHER_EXPERIMENT_PREFIX=exp2")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # dataset/ is gitignored, so a fresh worktree does not contain it.
        # Kaggle only needs the metadata id; the videos already live in that dataset.
        metadata_src = REPO_ROOT / "dataset" / "dataset-metadata.json"
        if metadata_src.is_file():
            metadata_dest = worktree / "dataset" / "dataset-metadata.json"
            metadata_dest.parent.mkdir(parents=True, exist_ok=True)
            metadata_dest.write_text(metadata_src.read_text(encoding="utf-8"), encoding="utf-8")

        _run(
            [
                sys.executable,
                "deploy.py",
                "--mode",
                machine.mode,
                "--target",
                machine.target,
                "--gpu",
                gpu,
            ],
            cwd=worktree,
            env=env,
        )


def run_machine_pipeline(
    worktree: Path,
    machine: MachineBinding,
    experiment_id: str,
    *,
    timeout: int = 10800,
    gpu: str = "T4",
    skip_deploy: bool = False,
) -> dict[str, Any]:
    """Full terminal: deploy, stages, archive, benchmark for one machine."""
    out = machine_dir(experiment_id, machine.name)
    status: dict[str, Any] = {
        "machine": machine.name,
        "video": machine.video,
        "experiment_id": experiment_id,
        "status": "running",
    }

    if is_done(experiment_id, machine.name, "complete"):
        print(f"[INFO] Skipping {machine.name}/{experiment_id}: already complete")
        status["status"] = "skipped"
        return status

    try:
        if not skip_deploy and not is_done(experiment_id, machine.name, "deploy"):
            try:
                deploy_worker(worktree, machine, experiment_id, gpu=gpu)
                append_progress(
                    {
                        "experiment_id": experiment_id,
                        "machine": machine.name,
                        "step": "deploy",
                        "status": "ok",
                    }
                )
            except Exception as exc:
                _cli_log_snapshot(machine, out / "cli_deploy.log")
                append_progress(
                    {
                        "experiment_id": experiment_id,
                        "machine": machine.name,
                        "step": "deploy",
                        "status": "error",
                        "error": str(exc),
                    }
                )
                raise

        with setup_server_connection(mode=machine.mode, target=machine.target) as base_url:
            client = PipelineClient(base_url=base_url, timeout=timeout)
            _fetch_worker_log(base_url, out / "worker_before.log")

            stages_to_run = list(PIPELINE_STAGES)
            # Restore audio/color from local cache when available.
            for stage in ("audio", "color"):
                cache_path = cache_dir(machine.name) / f"{machine.video}_{stage}.json"
                timing_cache = cache_dir(machine.name) / f"{machine.video}_{stage}_timings.json"
                if cache_path.is_file() and not is_done(experiment_id, machine.name, stage):
                    print(f"[INFO] Restoring cached {stage} for {machine.name}")
                    _restore_cached_stage(base_url, stage, machine.video, cache_path)
                    write_json(out / f"{machine.video}_{stage}.json", json.loads(cache_path.read_text(encoding="utf-8")))
                    if timing_cache.is_file():
                        timing_doc = json.loads(timing_cache.read_text(encoding="utf-8"))
                        timing_doc["cached"] = True
                        write_json(out / f"{machine.video}_{stage}_timings.json", timing_doc)
                    append_progress(
                        {
                            "experiment_id": experiment_id,
                            "machine": machine.name,
                            "step": stage,
                            "status": "ok",
                            "cached": True,
                        }
                    )
                    stages_to_run = [s for s in stages_to_run if s != stage]

            for stage in stages_to_run:
                if is_done(experiment_id, machine.name, stage):
                    continue
                print(f"[INFO] {machine.name}: running stage {stage}")
                _fetch_worker_log(base_url, out / "worker.log")
                result = client.execute_stage(stage, machine.video)
                if isinstance(result, dict) and result.get("status") == "failed":
                    write_json(out / f"{stage}_error.json", result)
                    append_progress(
                        {
                            "experiment_id": experiment_id,
                            "machine": machine.name,
                            "step": stage,
                            "status": "error",
                            "error": result.get("error"),
                        }
                    )
                    raise RuntimeError(f"{machine.name} stage {stage} failed: {result.get('error')}")

                document = result.get("result", result) if isinstance(result, dict) else result
                write_json(out / f"{machine.video}_{stage}.json", document)
                if isinstance(result, dict) and "timings" in result:
                    write_json(out / f"{machine.video}_{stage}_timings.json", result["timings"])

                # Pull sidecar / raw / timings files when present.
                for suffix in (
                    f"{machine.video}_{stage}_sidecar.json",
                    f"{machine.video}_{stage}_raw.txt",
                    f"{machine.video}_{stage}_timings.json",
                    f"{machine.video}_vlm_context.txt",
                    f"{machine.video}_dialogue_summary.txt",
                ):
                    if stage != "vlm" and (
                        suffix.endswith("_vlm_context.txt")
                        or suffix.endswith("_dialogue_summary.txt")
                    ):
                        continue
                    if stage != "vlm" and suffix.endswith("_sidecar.json"):
                        continue
                    _download_artifact(base_url, stage, suffix, out / suffix)

                if stage in ("audio", "color") and isinstance(document, (dict, list)):
                    _cache_stage_result(machine.name, machine.video, stage, document)
                    timing_path = out / f"{machine.video}_{stage}_timings.json"
                    if timing_path.is_file():
                        write_json(
                            cache_dir(machine.name) / timing_path.name,
                            json.loads(timing_path.read_text(encoding="utf-8")),
                        )

                append_progress(
                    {
                        "experiment_id": experiment_id,
                        "machine": machine.name,
                        "step": stage,
                        "status": "ok",
                    }
                )
                _fetch_worker_log(base_url, out / "worker.log")

            # Merge timings
            timing_paths = sorted(out.glob(f"{machine.video}_*_timings.json"))
            write_json(out / "timings.json", merge_timings(timing_paths))

            # Benchmark as soon as the LLM profile exists.
            profile_path = out / f"{machine.video}_llm.json"
            score_path = out / "score.json"
            if profile_path.is_file() and not is_done(experiment_id, machine.name, "score"):
                profile = json.loads(profile_path.read_text(encoding="utf-8"))
                score = score_profile(profile, machine.summary_path)
                save_score(score, score_path)
                append_progress(
                    {
                        "experiment_id": experiment_id,
                        "machine": machine.name,
                        "step": "score",
                        "status": "ok",
                        "overall": score.get("overall"),
                    }
                )

            _fetch_worker_log(base_url, out / "worker.log")
            _cli_log_snapshot(machine, out / "cli_final.log")

        append_progress(
            {
                "experiment_id": experiment_id,
                "machine": machine.name,
                "step": "complete",
                "status": "ok",
            }
        )
        status["status"] = "ok"
        return status
    except Exception as exc:
        status["status"] = "error"
        status["error"] = str(exc)
        write_json(out / "error.json", status)
        append_progress(
            {
                "experiment_id": experiment_id,
                "machine": machine.name,
                "step": "complete",
                "status": "error",
                "error": str(exc),
            }
        )
        _cli_log_snapshot(machine, out / "cli_error.log")
        return status
