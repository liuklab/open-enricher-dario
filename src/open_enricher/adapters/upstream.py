"""Install the pinned upstream inference modules in the shared model cache."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

from open_enricher import storage

TALKNET_REV = "6d6821479af485e251c4991487e40573b42181b4"
LAVIS_REV = "05cb670f47296522e24e7727e366b4a997d28582"
SOURCES = {
    "talknet": (
        "TaoRuijie/TalkNet-ASD",
        TALKNET_REV,
        (
            "model/audioEncoder.py",
            "model/visualEncoder.py",
            "model/attentionLayer.py",
            "model/talkNetModel.py",
            "loss.py",
            "LICENSE.md",
        ),
    ),
    "lavis": (
        "salesforce/LAVIS",
        LAVIS_REV,
        (
            "lavis/common/dist_utils.py",
            "lavis/models/eva_vit.py",
            "lavis/models/blip2_models/Qformer.py",
            "LICENSE.txt",
        ),
    ),
}
LAVIS_PACKAGES = (
    "transformers==4.37.2",
    "tokenizers==0.15.2",
    "huggingface-hub==0.25.2",
    "timm==0.4.12",
)


def runtime_path(name: str) -> Path:
    revision = SOURCES[name][1]
    tag = f"-{sys.implementation.cache_tag}-v1" if name == "lavis" else ""
    return storage.models_cache() / "external" / "upstream" / f"{name}-{revision}{tag}"


def activate(name: str) -> None:
    """Expose the pinned modules inside the optional adapter worker."""
    directory = runtime_path(name)
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing {name} runtime at {directory}. "
            f"Run python -m open_enricher.adapters.upstream {name}"
        )
    paths = [directory, directory / "deps"] if name == "lavis" else [directory]
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def install(name: str) -> None:
    destination = runtime_path(name)
    if destination.is_dir():
        print(f"Already installed at {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    repo, revision, files = SOURCES[name]
    with tempfile.TemporaryDirectory(dir=destination.parent) as work:
        staged = Path(work) / "runtime"
        for file in files:
            url = f"https://raw.githubusercontent.com/{repo}/{revision}/{file}"
            target = staged / file
            target.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(url, timeout=45) as response:
                target.write_bytes(response.read())
        if name == "talknet":
            (staged / "model" / "__init__.py").touch()
        else:
            # Import just the published encoder modules. The full LAVIS package
            # initializer eagerly imports unrelated datasets and language models.
            (staged / "lavis" / "__init__.py").touch()
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-deps",
                 "--target", str(staged / "deps"), *LAVIS_PACKAGES],
                check=True,
            )
        try:
            staged.rename(destination)
        except OSError:
            if not destination.is_dir():
                raise
    print(f"Installed {name} at {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("all", *SOURCES), nargs="?", default="all")
    selection = parser.parse_args().model
    for model in SOURCES if selection == "all" else (selection,):
        install(model)
