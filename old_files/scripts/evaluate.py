#!/usr/bin/env python3
"""Run metadata inference on Colab, then evaluate and retune params with a Cursor SDK agent.

The agent prompt lives in scripts/eval_prompt.md. The agent rewrites
src/open_enricher/metadata_params.json in full after each run.
CURSOR_API_KEY is loaded from repo-root `.env.eval` (not from `.env`, which bootstrap uploads).

Example from the repository root:

  # .env.eval contains: CURSOR_API_KEY=cursor_...
  pip install -e ".[eval]"
  python scripts/evaluate.py \\
    --max-iters 3 \\
    --session enricher \\
    --gpu T4 \\
    --auth adc \\
    --timeout 7200 \\
    --install-timeout 600
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ENV_EVAL_PATH = ROOT / ".env.eval"


def _load_config_module():
    """Load config.py without importing open_enricher.__init__ (avoids torch/numpy)."""

    import importlib.util

    path = SRC / "open_enricher" / "config.py"
    spec = importlib.util.spec_from_file_location("open_enricher_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_config = _load_config_module()
METADATA_PARAMS_PATH = _config.METADATA_PARAMS_PATH
load_metadata_params = _config.load_metadata_params

OUTPUTS_DIR = ROOT / "outputs"
BOOTSTRAP = ROOT / "colab_cli" / "bootstrap.py"
EVAL_PROMPT_PATH = Path(__file__).resolve().parent / "eval_prompt.md"
AGENT_TOOLS = ["read", "grep", "glob", "ls", "edit"]
AGENT_MODEL = "composer-2.5"
MAX_VALIDATION_RETRIES = 3


def load_eval_prompt() -> str:
    """Fill scripts/eval_prompt.md with the fixed model and video-processor settings."""

    if not EVAL_PROMPT_PATH.is_file():
        raise FileNotFoundError(f"Eval prompt file does not exist: {EVAL_PROMPT_PATH}")
    template = EVAL_PROMPT_PATH.read_text(encoding="utf-8").strip()
    if not template:
        raise ValueError(f"Eval prompt file is empty: {EVAL_PROMPT_PATH}")
    visual_token_budget = _config.METADATA_VIDEO_LONGEST_EDGE // (32 * 32 * 2)
    return template.format(
        model_id=_config.MODEL_ID,
        dtype=_config.MODEL_DTYPE,
        device_map=_config.MODEL_DEVICE_MAP,
        attn_implementation=_config.MODEL_ATTN_IMPLEMENTATION,
        video_name=_config.DRIVE_VIDEO,
        video_seconds=_config.METADATA_VIDEO_SECONDS,
        min_frames=_config.METADATA_MIN_FRAMES,
        max_frames=_config.METADATA_MAX_FRAMES,
        max_video_tokens=_config.METADATA_MAX_VIDEO_TOKENS,
        video_shortest_edge=_config.METADATA_VIDEO_SHORTEST_EDGE,
        video_longest_edge=_config.METADATA_VIDEO_LONGEST_EDGE,
        visual_token_budget=visual_token_budget,
    )

RETRY_PROMPT = """The metadata_params.json you wrote failed validation:

{error}

Rewrite the entire src/open_enricher/metadata_params.json with all seven keys inside the allowed ranges, then end with exactly one JSON object:
{{"passed": <bool>, "rationale": "<short reason>"}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Colab metadata inference loop with Cursor SDK evaluation.",
    )
    parser.add_argument("--max-iters", required=True, type=int, help="Maximum inference+eval iterations.")
    parser.add_argument("--session", required=True, help="Named Colab CLI session.")
    parser.add_argument("--gpu", required=True, help="GPU type, for example T4, L4, or A100.")
    parser.add_argument("--auth", required=True, help="Colab CLI auth method, for example adc.")
    parser.add_argument("--timeout", required=True, type=int, help="Seconds for inference exec.")
    parser.add_argument(
        "--install-timeout",
        required=True,
        type=int,
        help="Seconds for remote pip / install_local exec.",
    )
    return parser.parse_args()


def run_bootstrap(args: argparse.Namespace, *, skip_setup: bool) -> None:
    cmd = [
        sys.executable,
        str(BOOTSTRAP),
        "--session", args.session,
        "--gpu", args.gpu,
        "--auth", args.auth,
        "--timeout", str(args.timeout),
        "--install-timeout", str(args.install_timeout),
    ]
    if skip_setup:
        cmd.append("--skip-setup")
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, check=False, cwd=str(ROOT))
    if result.returncode != 0:
        raise RuntimeError(f"colab_cli/bootstrap.py failed with exit code {result.returncode}")


def latest_metadata_artifacts() -> tuple[Path, Path]:
    """Return (metadata.txt, sidecar.json) from the newest stamp under outputs/."""

    if not OUTPUTS_DIR.is_dir():
        raise FileNotFoundError(f"Outputs directory missing: {OUTPUTS_DIR}")

    stamp_dirs = sorted(
        (
            path
            for path in OUTPUTS_DIR.iterdir()
            if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", path.name)
        ),
        key=lambda path: path.name,
    )
    if not stamp_dirs:
        raise FileNotFoundError(f"No run directories under {OUTPUTS_DIR}")

    metadata_dir = stamp_dirs[-1] / "metadata"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"No metadata directory in latest run: {metadata_dir}")

    txt_files = sorted(metadata_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No metadata .txt in {metadata_dir}")
    txt_path = txt_files[0]
    sidecar = txt_path.with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing sidecar JSON: {sidecar}")
    return txt_path, sidecar


def parse_verdict(text: str) -> dict:
    """Extract the final {\"passed\": ..., \"rationale\": ...} object from agent text."""

    matches = list(re.finditer(r"\{[^{}]*\"passed\"[^{}]*\}", text, flags=re.DOTALL))
    if not matches:
        raise ValueError(f"Agent reply has no passed/rationale JSON object:\n{text}")
    payload = json.loads(matches[-1].group(0))
    if "passed" not in payload or type(payload["passed"]) is not bool:
        raise ValueError(f"Verdict missing boolean passed: {payload!r}")
    if "rationale" not in payload or not isinstance(payload["rationale"], str):
        raise ValueError(f"Verdict missing string rationale: {payload!r}")
    return {"passed": payload["passed"], "rationale": payload["rationale"]}


def agent_turn(agent, message: str) -> dict:
    run = agent.send(message)
    result = run.wait()
    if result.status == "error":
        raise RuntimeError(f"Cursor agent run failed: {result.id}")
    text = result.result or ""
    print(text)
    return parse_verdict(text)


def validate_params_file() -> None:
    load_metadata_params(METADATA_PARAMS_PATH)


def main() -> None:
    args = parse_args()
    if args.max_iters <= 0:
        raise SystemExit("--max-iters must be a positive int")

    # Local only: bootstrap packs `.env`, never `.env.eval`.
    load_dotenv(ENV_EVAL_PATH)
    api_key = os.environ.get("CURSOR_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            f"CURSOR_API_KEY is not set. Put it in {ENV_EVAL_PATH} "
            "(or export it in the process environment)."
        )

    try:
        from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
    except ImportError as exc:
        raise SystemExit(
            'cursor-sdk is not installed. Install with: pip install -e ".[eval]"'
        ) from exc

    validate_params_file()
    eval_prompt = load_eval_prompt()

    with Agent.create(
        AgentOptions(
            model=AGENT_MODEL,
            api_key=api_key,
            tools=AGENT_TOOLS,
            local=LocalAgentOptions(cwd=str(ROOT)),
        )
    ) as agent:
        for iteration in range(args.max_iters):
            print(f"=== Iteration {iteration + 1}/{args.max_iters} ===")
            run_bootstrap(args, skip_setup=iteration > 0)

            txt_path, sidecar = latest_metadata_artifacts()
            print("Latest metadata:", txt_path)
            print("Sidecar:", sidecar)

            if iteration == 0:
                message = eval_prompt
            else:
                message = "A new Colab run finished with the params you wrote.\n" + eval_prompt
            message += (
                f"\nLatest metadata file: {txt_path.relative_to(ROOT)}\n"
                f"Sidecar: {sidecar.relative_to(ROOT)}\n"
                f"Params file: {METADATA_PARAMS_PATH.relative_to(ROOT)}\n"
            )

            verdict = agent_turn(agent, message)

            # Re-prompt until the rewritten JSON validates, without another Colab run.
            for _ in range(MAX_VALIDATION_RETRIES):
                try:
                    validate_params_file()
                    break
                except (ValueError, FileNotFoundError, json.JSONDecodeError) as err:
                    print("Params validation failed:", err)
                    verdict = agent_turn(agent, RETRY_PROMPT.format(error=err))
            else:
                validate_params_file()  # raise the last failure if still invalid

            print("Verdict:", verdict)
            if verdict["passed"]:
                print("Evaluation passed; stopping.")
                return

            print("Evaluation failed; continuing with updated metadata_params.json")

        print(f"Reached --max-iters ({args.max_iters}) without a passed verdict.")


if __name__ == "__main__":
    main()
