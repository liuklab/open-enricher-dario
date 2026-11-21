"""Drop InsightFace faces that are small on the frame.

The cutoff is the 50th percentile of relative box size, measured once on
the HIMYM S01E02 middle 30 seconds (144 frames, 846 detections, 1920x1080).
It is a fixed constant. Do not recompute a percentile for each video.
"""

from __future__ import annotations

# Larger box side divided by the matching frame side. Linear 50th
# percentile of the mid-episode smoke: the average of the 423rd and 424th
# sorted ratios (0.1303703703703704 and 0.13064814814814815).
MIN_FACE_SIDE_RATIO = 0.13050925925925927


def relative_face_side(
    bbox: list[float],
    frame_width: float,
    frame_height: float,
) -> float | None:
    """Larger face-box side divided by the frame side in that direction.

    A wide box is compared with the frame width. A tall box is compared
    with the frame height. Returns ``None`` when the box or the frame
    has no positive size.
    """
    if len(bbox) < 4 or frame_width <= 0 or frame_height <= 0:
        return None
    box_width = float(bbox[2]) - float(bbox[0])
    box_height = float(bbox[3]) - float(bbox[1])
    if box_width <= 0 or box_height <= 0:
        return None
    if box_width >= box_height:
        return box_width / float(frame_width)
    return box_height / float(frame_height)


def face_is_relevant(
    bbox: list[float],
    frame_width: float,
    frame_height: float,
    *,
    minimum: float = MIN_FACE_SIDE_RATIO,
) -> bool:
    """True when the face is at least ``minimum`` of the frame.

    Faces below the constant are dropped. A face exactly on the constant
    is kept.
    """
    ratio = relative_face_side(bbox, frame_width, frame_height)
    return ratio is not None and ratio >= minimum
