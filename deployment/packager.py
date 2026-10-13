"""Packaging logic to prepare the local codebase for deployment."""

import shutil
import tarfile
import tempfile
from pathlib import Path

PACK_FILES = ("pyproject.toml", "requirements.txt", "README.md")
SRC_DIR_NAME = "src"


class ProjectPackager:
    """Handles the creation of a clean deployment tarball."""

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def create_tarball(self, dest_path: Path) -> None:
        print(f"Packing local checkout into {dest_path.name}...")

        with tempfile.TemporaryDirectory() as tmp_dir:
            staging = Path(tmp_dir) / "staging"
            staging.mkdir()

            src_path = self.root_dir / SRC_DIR_NAME
            if src_path.exists():
                shutil.copytree(
                    src_path,
                    staging / SRC_DIR_NAME,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".pytest_cache")
                )

            for file_name in PACK_FILES:
                file_path = self.root_dir / file_name
                if file_path.exists():
                    shutil.copy2(file_path, staging / file_name)

            with tarfile.open(dest_path, "w:gz") as archive:
                archive.add(staging, arcname=".")