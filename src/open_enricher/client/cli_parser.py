"""Command Line Interface parsing."""

from __future__ import annotations

import argparse

from src.open_enricher.client.pipeline_cmd import PipelineConfig
from src.open_enricher.client.pipeline_order import STAGE_ORDER

def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    """Parses CLI arguments and returns a strict config object."""
    parser = argparse.ArgumentParser(description="Run specific pipeline jobs remotely.")
    
    arguments = [
        ("--mode", {
            "choices": ["tailscale", "colab-cli"], 
            "required": True, 
            "help": "Networking mode to communicate with the server."
        }),
        ("--target", {
            "required": True, 
            "help": "Tailscale IP address or Colab CLI session name."
        }),
        ("--video", {"required": True, "help": "Video file name with extension."}),
        ("--timeout", {"type": int, "default": 3600, "help": "Timeout in seconds."}),
        ("--stages", {
            "nargs": "+", 
            "default": list(STAGE_ORDER), 
            "help": f"Stages to run. Default: {' '.join(STAGE_ORDER)}"
        })
    ]
    
    for arg, kwargs in arguments:
        parser.add_argument(arg, **kwargs)
        
    args = parser.parse_args(argv)
    
    return PipelineConfig(
        mode=args.mode, 
        target=args.target, 
        video=args.video, 
        timeout=args.timeout,
        stages=args.stages
    )