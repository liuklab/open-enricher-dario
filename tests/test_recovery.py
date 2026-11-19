"""Recovery: explicit stages, manifests, child crashes, and missing weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from open_enricher.adapters.bassl import segment_video
from open_enricher.adapters.isolate import call_in_child
from open_enricher.experiments.config import PIPELINE_STAGES, VARIANTS_DIR
from open_enricher.experiments.stages import stages_for_spec
from open_enricher.manifest import profile_decision
from open_enricher.stages.pipeline_spec import Checkpoints, PipelineSpec, load_pipeline_spec
from open_enricher import storage


def test_control_does_not_insert_optional_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPEN_ENRICHER_PIPELINE_SPEC", raising=False)
    spec = load_pipeline_spec(VARIANTS_DIR / "06-memory-2.json")
    assert stages_for_spec(PIPELINE_STAGES, spec) == list(PIPELINE_STAGES)


def test_combined_variant_inserts_optional_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPEN_ENRICHER_PIPELINE_SPEC", raising=False)
    spec = load_pipeline_spec(VARIANTS_DIR / "12-combined.json")
    names = stages_for_spec(PIPELINE_STAGES, spec)
    assert names[:3] == ["timeline", "audio", "bassl"]
    assert "character_graph" in names
    assert "visual_memory" in names
    assert names[-2:] == ["vlm", "llm"]


def test_explicit_subset_is_not_expanded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPEN_ENRICHER_PIPELINE_SPEC", raising=False)
    spec = load_pipeline_spec(VARIANTS_DIR / "12-combined.json")
    assert stages_for_spec(("vlm", "llm"), spec) == ["vlm", "llm"]


def test_stale_profile_is_rejected_and_degraded_is_not_scored() -> None:
    manifest = {
        "profile_kind": "degraded",
        "artifacts": [{"name": "episode_llm.json"}],
    }
    assert profile_decision(manifest, "episode_llm.json") == "degraded"
    assert profile_decision(manifest, "old_llm.json") == "rejected"
    assert profile_decision(None, "episode_llm.json") is None


def test_atomic_write_replaces_the_destination(tmp_path: Path) -> None:
    target = tmp_path / "note.json"
    target.write_text("old", encoding="utf-8")
    storage.atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"
    assert list(tmp_path.glob(".note.json.*")) == []


def test_child_crash_is_a_fallback() -> None:
    result = call_in_child(
        "open_enricher.adapters.isolate",
        "crash_payload",
        {"ok": True},
        timeout=30,
    )
    assert result["status"] == "fallback"


def test_missing_bassl_checkpoint_does_not_raise(tmp_path: Path) -> None:
    spec = PipelineSpec(
        experiment_id="12-combined",
        title="combined",
        interval_seconds=300,
        vlm_memory_scenes=2,
        visual_only=False,
        narrate_from_visual=False,
        checkpoints=Checkpoints(bassl=str(tmp_path)),
    )
    result = segment_video(tmp_path / "clip.mkv", 12.0, spec)
    assert result["status"] == "fallback"
    json.dumps(result)
