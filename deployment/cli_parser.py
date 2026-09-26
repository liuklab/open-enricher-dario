"""CLI interface parsing and validation."""

import argparse

def parse_args() -> argparse.Namespace:
    """
    Parses CLI arguments provided by the user.
    
    Returns:
        argparse.Namespace: The parsed arguments containing 'mode' and 'target'.
    """
    parser = argparse.ArgumentParser(
        description="Deploy the Open-Enricher codebase to a remote Google Colab instance."
    )
    parser.add_argument(
        "--mode", 
        choices=["tailscale", "colab-cli"], 
        required=True, 
        help="The networking strategy to use for deployment."
    )
    parser.add_argument(
        "--target", 
        required=True, 
        help="The Tailscale IP address or Colab CLI session name."
    )
    
    return parser.parse_args()