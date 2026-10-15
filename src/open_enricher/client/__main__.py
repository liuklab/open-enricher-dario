"""Entrypoint for the client module."""

from __future__ import annotations

import sys

from src.open_enricher.client.cli_parser import parse_args
from src.open_enricher.client.orchestrator import PipelineOrchestrator
from src.open_enricher.client.network import PipelineClient, setup_server_connection

def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    
    print(f"[INFO] Initializing connection via {config.mode.upper()}...")
    
    try:
        with setup_server_connection(mode=config.mode, target=config.target) as base_url:
            
            client = PipelineClient(base_url=base_url, timeout=config.timeout)
            orchestrator = PipelineOrchestrator(client=client, config=config)

            print(f"[INFO] Target video: {config.video}")
            print(f"[INFO] Stages queue: {config.stages}")
            
            orchestrator.run_jobs()
            
            print("[SUCCESS] All requested jobs completed successfully.")
            
    except Exception as exc:
        print(f"[ERROR] Pipeline stopped abruptly: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

if __name__ == "__main__":
    main()