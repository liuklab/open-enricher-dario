"""Relative face size uses one fixed cutoff, not a per-video percentile."""

from open_enricher.face_size import (
    MIN_FACE_SIDE_RATIO,
    face_is_relevant,
    relative_face_side,
)


def test_larger_side_is_divided_by_the_matching_frame_side() -> None:
    wide = relative_face_side([0, 0, 200, 80], 1920, 1080)
    tall = relative_face_side([0, 0, 40, 200], 1920, 1080)
    assert wide == 200 / 1920
    assert tall == 200 / 1080


def test_faces_below_the_fixed_ratio_are_dropped() -> None:
    assert MIN_FACE_SIDE_RATIO == 0.13050925925925927
    assert face_is_relevant([0, 0, 100, 100], 1920, 1080) is False
    assert face_is_relevant([0, 0, 400, 200], 1920, 1080) is True
    # Exactly the constant is kept.
    side = MIN_FACE_SIDE_RATIO * 1920
    assert face_is_relevant([0, 0, side, 10], 1920, 1080) is True
