"""Pipeline spec and benchmark weight helpers."""

from __future__ import annotations

import json
from pathlib import Path

from src.open_enricher.experiments.benchmark import _weighted_overall, prompt_hash
from src.open_enricher.experiments.config import SCORE_WEIGHTS, VARIANTS_DIR
from src.open_enricher.remote.stages.pipeline_spec import load_pipeline_spec


def test_default_pipeline_spec_loads() -> None:
    spec = load_pipeline_spec()
    assert spec.segmentation in ("interval", "scenes")
    assert spec.dialogue_color_routing in ("vlm", "llm")


def test_variants_are_valid() -> None:
    for path in sorted(VARIANTS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        # Strip orchestrator notes before validation through the loader.
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        tmp = path.parent / f".tmp_{path.name}"
        try:
            tmp.write_text(json.dumps(clean), encoding="utf-8")
            spec = load_pipeline_spec(tmp)
            assert spec.experiment_id
            assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9
        finally:
            if tmp.exists():
                tmp.unlink()


def test_weighted_overall() -> None:
    dims = {
        key: {"score": 5, "note": "x"}
        for key in SCORE_WEIGHTS
    }
    overall, normalized = _weighted_overall(dims)
    assert overall == 5.0
    assert normalized == 1.0


def test_prompt_hash_stable() -> None:
    assert len(prompt_hash()) == 16
