"""TalkNet windows, the checkpoint graph, and speaker edges."""

from __future__ import annotations

import os
from pathlib import Path

import numpy
import pytest

from open_enricher.adapters.talknet_asd import (
    TalkNetASD,
    forward_scores,
    load_talknet,
    mfcc,
    select_windows,
)
from open_enricher.graph import compact_account, speaker_edges


def test_one_face_yields_several_one_second_windows() -> None:
    tracks = []
    for index in range(20):
        tracks.append(
            {
                "track_id": f"f{index}",
                "start": 20 + index * 0.2,
                "bbox": [10, 10, 110, 130],
            }
        )
    for index in range(5):
        tracks.append(
            {
                "track_id": f"g{index}",
                "start": 20 + index * 0.2,
                "bbox": [400, 10, 500, 130],
            }
        )
    windows = select_windows(tracks, max_tracks=4)
    assert len(windows) == 4
    assert sum(1 for row in windows if str(row["track_id"]).startswith("f")) >= 2
    for row in windows:
        assert row["end"] - row["start"] == pytest.approx(1.0)


def test_two_hits_reach_the_profile_account() -> None:
    decisions = [
        {"track_id": "a", "character_id": "c0001"},
        {"track_id": "b", "character_id": "c0001"},
    ]
    scores = [
        {"track_id": "a", "start": 20.0, "end": 21.0, "score": 1.2, "reliable": True},
        {"track_id": "b", "start": 22.0, "end": 23.0, "score": 0.8, "reliable": True},
        {"track_id": "c", "start": 22.0, "end": 23.0, "score": -0.4, "reliable": False},
    ]
    lines = [
        {"start_timestamp": 20.2, "end_timestamp": 20.8, "speaker": "SPEAKER_00", "text": "So."},
        {"start_timestamp": 22.1, "end_timestamp": 22.9, "speaker": "SPEAKER_00", "text": "Back then."},
    ]
    edges = speaker_edges(scores, decisions, lines)
    assert [(edge.start, edge.end) for edge in edges] == [(20.2, 20.8), (22.1, 22.9)]
    assert all(
        (edge.source, edge.target, edge.status) == ("c0001", "SPEAKER_00", "supported")
        for edge in edges
    )
    account = compact_account(
        [{"character_id": "c0001", "alias": None, "tracks": ["a", "b"]}],
        [edge.to_dict() for edge in edges],
    )
    assert "c0001" in account
    assert "speaks SPEAKER_00" in account


def test_one_hit_stays_unresolved() -> None:
    edges = speaker_edges(
        [{"track_id": "a", "start": 1.0, "end": 2.0, "score": 1.0, "reliable": True}],
        [{"track_id": "a", "character_id": "c0001"}],
        [{"start_timestamp": 1.1, "end_timestamp": 1.5, "speaker": "SPEAKER_01"}],
    )
    assert edges == []


def test_mfcc_frame_count_matches_the_talknet_rate() -> None:
    # 1.00 s is 99 frames at 25 ms / 10 ms. The scorer reads 1.25 s so a
    # one-second TalkNet window still has 100 MFCC rows.
    tone = numpy.zeros(20000, dtype=numpy.int16)
    features = mfcc(tone)
    assert features.shape[1] == 13
    assert features.shape[0] >= 100


def test_unloaded_forward_returns_one_logit_per_frame() -> None:
    network = TalkNetASD().eval()
    audio = numpy.zeros((100, 13), dtype=numpy.float32)
    visual = numpy.zeros((25, 112, 112), dtype=numpy.float32)
    scores = forward_scores(network, audio, visual)
    assert scores.shape == (25,)
    assert numpy.isfinite(scores).all()


def _checkpoint() -> Path | None:
    roots = []
    override = os.environ.get("OPEN_ENRICHER_DATASET_ROOT")
    if override:
        roots.append(Path(override))
    roots.append(Path("/SLURM_shared/st_mdp26/open-enricher/dataset"))
    for root in roots:
        path = root / "models_cache" / "external" / "talknet" / "pretrain_TalkSet.model"
        if path.is_file():
            return path
    return None


def test_checkpoint_matches_the_vendored_graph() -> None:
    path = _checkpoint()
    if path is None:
        pytest.skip("TalkNet checkpoint is not on this machine")
    network, device = load_talknet(path, device="cpu")
    assert device == "cpu"
    assert network.lossAV.FC.weight.shape == (2, 256)
    audio = numpy.zeros((100, 13), dtype=numpy.float32)
    visual = numpy.full((25, 112, 112), 128, dtype=numpy.float32)
    scores = forward_scores(network, audio, visual)
    assert scores.shape == (25,)
    assert numpy.isfinite(scores).all()
