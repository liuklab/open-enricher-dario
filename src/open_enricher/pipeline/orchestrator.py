"""Local dispatcher: walk STAGE_ORDER and POST each stage to the remote worker."""

from __future__ import annotations

import argparse
import json
import sys

from open_enricher.pipeline.order import STAGE_ORDER
from open_enricher.remote.client import post_stage


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ordered pipeline stages on a remote FastAPI worker.",
    )
    parser.add_argument("--host", required=True, help="Worker Tailscale IPv4 or hostname.")
    parser.add_argument("--port", type=int, default=8000, help="Worker HTTP port (default 8000).")
    parser.add_argument(
        "--video",
        required=True,
        help="Video file name with extension, resolved on the worker dataset root.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Seconds to wait for each stage response (default 3600).",
    )
    return parser.parse_args(argv)


def run_pipeline(host: str, port: int, video: str, timeout: int) -> dict[str, object]:
    """POST each stage in STAGE_ORDER and return a map of stage name to result."""

    base_url = f"http://{host}:{port}"
    results: dict[str, object] = {}
    for name in STAGE_ORDER:
        print(f"Running stage: {name}")
        result = post_stage(base_url, name, video, timeout=timeout)
        results[name] = result
        print(json.dumps(result, indent=2, ensure_ascii=True))
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run_pipeline(args.host, args.port, args.video, args.timeout)
    except Exception as exc:
        print(f"Pipeline stopped: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
