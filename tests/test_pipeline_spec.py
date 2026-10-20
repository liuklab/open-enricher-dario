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
    assert spec.segmentation in ("interval", "scenes")
    assert spec.dialogue_color_routing in ("vlm", "llm")


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
        finally:
            if tmp.exists():
                tmp.unlink()


def test_list_variants_sorted() -> None:
    """Variant listing is deterministic and non-empty."""
    names = [path.name for path in list_variants(VARIANTS_DIR)]
    assert names == sorted(names)
    assert names[0].startswith("01-")
