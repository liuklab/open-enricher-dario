"""CLI interface parsing and validation."""

import argparse

def parse_args() -> argparse.Namespace:
    """
    Parses CLI arguments provided by the user.
    
    Returns:
        argparse.Namespace: The parsed arguments containing 'mode' and 'target'.
    """
    parser = argparse.ArgumentParser(
        description="Deploy the Open-Enricher worker to Colab, Kaggle, or a Tailscale host."
    )
    parser.add_argument(
        "--mode",
        choices=["tailscale", "colab-cli", "kaggle"],
        required=True,
        help="Network strategy used to upload the checkout and start the worker."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Tailscale IP, Colab session name, or Kaggle kernel slug."
    )
    parser.add_argument(
        "--gpu",
        default="T4",
        help="GPU used when colab-cli or kaggle creates a session. T4 maps to NvidiaTeslaT4 on Kaggle.",
    )
    
    return parser.parse_args()