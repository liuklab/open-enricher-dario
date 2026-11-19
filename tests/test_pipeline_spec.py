"""Pipeline spec validation for packaged experiment variants.

No GPU required. Loads each variant JSON through ``load_pipeline_spec``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from open_enricher.experiments.archive import list_variants
from open_enricher.experiments.config import VARIANTS_DIR
from open_enricher.stages.pipeline_spec import load_pipeline_spec


def test_default_pipeline_spec_loads() -> None:
    """Packaged pipeline_spec.json must validate."""
    # Clear override so the packaged default is used.
    os.environ.pop("OPEN_ENRICHER_PIPELINE_SPEC", None)
    spec = load_pipeline_spec()
    assert spec.experiment_id == "01-scene-context"
    assert spec.interval_seconds == 300
    assert spec.vlm_memory_scenes is None
    assert spec.visual_only is False
    assert spec.narrate_from_visual is False


def test_variants_are_valid() -> None:
    """Every experiment variant JSON must load as a PipelineSpec."""
    paths = list_variants(VARIANTS_DIR)
    assert paths, "expected packaged variants"
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        clean = {key: value for key, value in data.items() if not str(key).startswith("_")}
        tmp = path.parent / f".tmp_{path.name}"
        try:
            tmp.write_text(json.dumps(clean), encoding="utf-8")
            spec = load_pipeline_spec(tmp)
            assert spec.experiment_id
            assert spec.title
            if spec.vlm_memory_scenes is not None:
                assert spec.vlm_memory_scenes > 0
            assert isinstance(spec.visual_only, bool)
            assert isinstance(spec.narrate_from_visual, bool)
        finally:
            if tmp.exists():
                tmp.unlink()


def test_absent_feature_flags_keep_the_legacy_path() -> None:
    """Variants 01-08 do not opt into BaSSL, the graph, or visual memory."""
    os.environ.pop("OPEN_ENRICHER_PIPELINE_SPEC", None)
    spec = load_pipeline_spec()
    assert spec.features.any_enabled is False
    assert spec.allocation_mode() is None
    assert spec.history_fraction() == 0.0


def test_list_variants_sorted() -> None:
    """Variant listing is deterministic and non-empty."""
    names = [path.name for path in list_variants(VARIANTS_DIR)]
    assert names == sorted(names)
    assert names[0].startswith("01-")
