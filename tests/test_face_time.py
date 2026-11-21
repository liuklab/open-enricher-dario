"""Face times are instants. Stuffed milliseconds do not come back."""

from open_enricher.face_time import face_instant, face_span


def test_stuffed_milliseconds_collapse_to_whole_seconds() -> None:
    face = face_instant(178.178)
    assert face == {"time": "00:02:58", "seconds": 178}
    assert "end" not in face
    assert face_instant(3.003) == {"time": "00:00:03", "seconds": 3}
    assert face_instant(1202.202) == {"time": "00:20:02", "seconds": 1202}


def test_real_frame_fraction_is_kept() -> None:
    # 15540 / 23.976 is the middle-segment frame, not a stuffed millisecond.
    face = face_instant(15540 / 23.976)
    assert face["time"] == "00:10:48"
    assert face["seconds"] != 648
    assert abs(face["seconds"] - 648.148148) < 1e-6
    assert "end" not in face


def test_whole_seconds_stay_beside_the_clock() -> None:
    assert face_instant(90) == {"time": "00:01:30", "seconds": 90}


def test_end_is_kept_only_when_it_is_a_later_instant() -> None:
    assert face_span(178.178, 178.178) == {"time": "00:02:58", "seconds": 178}
    assert face_span(178.178, 182.182) == {
        "time": "00:02:58",
        "seconds": 178,
        "end": "00:03:02",
        "end_seconds": 182,
    }
    opened = face_span(648.038038, 651.166166)
    assert opened["time"] == "00:10:48"
    assert opened["end"] == "00:10:51"
    assert opened["seconds"] != opened["end_seconds"]
