"""Emit a Python snippet that copies selected env vars onto the Colab kernel.

google-colab-cli 0.6 has no `colab exec --env`. The kernel is reused, so this
snippet is executed once before install_package / runners.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

FORWARD_KEYS = (
    "HF_TOKEN",
    "GITHUB_TOKEN",
    "GIT_TOKEN",
    "OPEN_ENRICHER_GIT_URL",
    "OPEN_ENRICHER_GIT_REF",
    "OPEN_ENRICHER_REPO",
    "OPEN_ENRICHER_DATA",
    "OPEN_ENRICHER_OUTPUT",
    "OPEN_ENRICHER_VIDEO",
    "OPEN_ENRICHER_MODEL",
    "OPEN_ENRICHER_NUM_FRAMES",
    "OPEN_ENRICHER_MAX_NEW_TOKENS",
)


def render_snippet() -> str:
    lines = ["import os"]
    count = 0
    for key in FORWARD_KEYS:
        value = os.environ.get(key)
        if value:
            lines.append(f"os.environ[{key!r}] = {value!r}")
            count += 1
    lines.append(f"print('Injected', {count}, 'environment variables')")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a kernel env-inject snippet.")
    parser.add_argument("-o", "--output", help="Output path. Default: stdout.")
    args = parser.parse_args()
    text = render_snippet()
    if args.output:
        path = Path(args.output)
        path.write_text(text, encoding="utf-8")
        path.chmod(0o600)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
