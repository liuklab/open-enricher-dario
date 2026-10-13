"""Install and join a Tailscale network in a Linux container such as Colab."""

from __future__ import annotations

import os
import subprocess
import sys
import time


def connect(auth_key: str, *, hostname: str = "colab-worker") -> str:
    """Install Tailscale if needed, authenticate, and return the IPv4 address."""

    if not auth_key or not auth_key.strip():
        raise ValueError("Tailscale auth key is empty")

    print("Installing Tailscale...")
    subprocess.run(
        "curl -fsSL https://tailscale.com/install.sh | sh",
        shell=True,
        check=True,
        capture_output=True,
    )

    # Userspace networking is required in containers without /dev/net/tun.
    subprocess.Popen(
        ["tailscaled", "--tun=userspace-networking", "--socks5-server=localhost:1055"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    print("Authenticating to the Tailscale network...")
    subprocess.run(
        ["tailscale", "up", "--authkey", auth_key, "--hostname", hostname],
        check=True,
    )

    ip_address = subprocess.check_output(["tailscale", "ip", "-4"]).decode().strip()
    if not ip_address:
        raise RuntimeError("tailscale ip -4 returned an empty address")
    print(f"Tailscale connected. Worker IPv4: {ip_address}")
    return ip_address


def main() -> None:
    """Read TS_AUTH_KEY or TAILSCALE_KEY from the environment and connect."""

    auth_key = os.environ.get("TS_AUTH_KEY") or os.environ.get("TAILSCALE_KEY")
    if not auth_key:
        print(
            "Set TS_AUTH_KEY or TAILSCALE_KEY in the environment.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    connect(auth_key)


if __name__ == "__main__":
    main()
