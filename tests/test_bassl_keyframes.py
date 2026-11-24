"""The BaSSL demo requests repeated frames, including across neighbouring batches."""

from pathlib import Path

import cv2
import numpy as np

from open_enricher.adapters.bassl import IndexedKeyframes


class Timecode:
    def __init__(self, index: int) -> None:
        self.index = index

    def get_frames(self) -> int:
        return self.index


def test_repeated_and_overlapping_shots_decode_at_the_requested_indices(tmp_path: Path) -> None:
    video = tmp_path / "shots.avi"
    encoder = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert encoder.isOpened()
    for index in range(64):
        encoder.write(np.full((32, 32, 3), index * 3, dtype=np.uint8))
    encoder.release()

    reference = cv2.VideoCapture(str(video))
    pixels = []
    while True:
        ok, frame = reference.read()
        if not ok:
            break
        pixels.append(frame[0, 0, 0])
    reference.release()
    assert len(pixels) == 64

    with IndexedKeyframes(str(video)) as decoder:
        first = decoder.read_batch({0: [Timecode(1), Timecode(1), Timecode(62)]}, 0, 2)
        second = decoder.read_batch({1: [Timecode(2), Timecode(2), Timecode(3)]}, 1, 2)
    values = [np.asarray(image)[0, 0, 0] for image in first[0] + second[0]]
    np.testing.assert_array_equal(values, [pixels[i] for i in (1, 1, 62, 2, 2, 3)])
