"""Kaggle input mount resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from deployment.kaggle_worker import mounted_dataset


def test_cli_mount_wins_over_editor_path(tmp_path: Path) -> None:
    owner_slug = tmp_path / "datasets" / "ada" / "open-enricher-code"
    short = tmp_path / "open-enricher-code"
    owner_slug.mkdir(parents=True)
    short.mkdir()
    (owner_slug / "launch.sh").write_text("#!/bin/bash\n", encoding="utf-8")

    found = mounted_dataset("ada/open-enricher-code", tmp_path)

    assert found == owner_slug


def test_editor_mount_is_the_fallback(tmp_path: Path) -> None:
    short = tmp_path / "suits-videos"
    short.mkdir()

    found = mounted_dataset("ada/suits-videos", tmp_path)

    assert found == short


def test_missing_dataset_names_both_paths(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="datasets/ada/missing"):
        mounted_dataset("ada/missing", tmp_path)
