"""Character identity, vocatives, and edge de-duplication."""

from __future__ import annotations

from open_enricher.graph import (
    CharacterNode,
    Edge,
    TrackObservation,
    apply_alias,
    assign_character_ids,
    compact_account,
    dedupe_edges,
    interpret_mention,
    link_observations,
    link_speaker,
    speaker_edges,
    visible_characters,
)


def test_returning_face_keeps_one_id_across_a_cut() -> None:
    tracks = [
        TrackObservation("t1", [1.0, 0.0], 0.0, 1.0, "shot-a"),
        TrackObservation("t2", [0.99, 0.01], 2.0, 3.0, "shot-b"),
    ]
    nodes, decisions = assign_character_ids(tracks, threshold=0.45, margin=0.08)
    assert len(nodes) == 1
    assert decisions[1]["decision"] == "merged"
    assert decisions[1]["character_id"] == nodes[0].character_id


def test_faces_visible_together_are_not_merged() -> None:
    tracks = [
        TrackObservation("t1", [1.0, 0.0], 0.0, 1.0, "shot", co_visible=("t2",)),
        TrackObservation("t2", [1.0, 0.0], 0.0, 1.0, "shot", co_visible=("t1",)),
    ]
    nodes, _decisions = assign_character_ids(tracks, threshold=0.45, margin=0.08)
    assert len(nodes) == 2


def test_vocative_does_not_rename_the_speaker() -> None:
    role, name = interpret_mention("Richard, come here")
    assert role == "addressee"
    assert name == "Richard"
    node = CharacterNode(character_id="c0001", embedding=[1.0, 0.0], alias=None)
    assert apply_alias(node, role, name) is None
    self_role, self_name = interpret_mention("I am Richard Hendricks")
    assert self_role == "self_introduction"
    assert apply_alias(node, self_role, self_name) == "Richard Hendricks"


def test_nearby_samples_of_one_face_become_one_track() -> None:
    face = [1.0, 0.0]
    other = [0.0, 1.0]
    tracks = [
        TrackObservation("f0", face, 0.0, 1.0, "frame"),
        TrackObservation("f1", [0.98, 0.02], 1.0, 2.0, "frame"),
        TrackObservation("f2", face, 2.0, 3.0, "frame"),
        TrackObservation("g0", other, 0.0, 1.0, "frame", co_visible=("f0",)),
    ]
    pooled = link_observations(tracks, threshold=0.6, gap=2.0)
    assert len(pooled) == 2
    kept_track = next(track for track in pooled if track.samples == 3)
    assert kept_track.end == 3.0
    nodes, _decisions = assign_character_ids(pooled, threshold=0.45, margin=0.08)
    visible = visible_characters(nodes)
    assert len(visible) == 1
    assert visible[0].samples == 3


def test_one_frame_faces_stay_out_of_the_published_graph() -> None:
    tracks = [
        TrackObservation(f"f{index}", [float(index), 1.0], float(index), float(index) + 0.2, "frame")
        for index in range(5)
    ]
    nodes, _decisions = assign_character_ids(tracks, threshold=0.99, margin=0.5)
    assert visible_characters(nodes) == []
    account = compact_account(
        [{"character_id": f"c{index:04d}", "samples": 1, "tracks": ["f"]} for index in range(20)],
        [],
    )
    assert "c0001" in account
    assert "8 further character(s) were left out." in account


def test_sentence_initial_words_are_not_names() -> None:
    role, name = interpret_mention("Okay, wait.")
    assert role == "third_party"
    assert name is None
    role, name = interpret_mention("It was Robin.")
    assert role == "third_party"
    assert name == "Robin"


def test_one_alignment_stays_unresolved_and_edges_do_not_duplicate() -> None:
    linked = link_speaker(
        [{"speaker": "SPEAKER_00", "character_id": "c0001", "reliable": True}]
    )
    assert linked == {}
    edge = Edge(
        "speaks",
        "c0001",
        "u1",
        0.0,
        1.0,
        ["u1"],
        0.9,
        "supported",
    )
    assert len(dedupe_edges([edge, edge])) == 1


def test_speaker_edges_use_original_face_sample_ids_after_pooling() -> None:
    observations = [
        TrackObservation(f"f{index}", [1.0, 0.0], index, index + 1, "shot")
        for index in range(3)
    ]
    pooled = link_observations(observations)
    nodes, decisions = assign_character_ids(pooled, threshold=0.45, margin=0.08)
    scores = [
        {"track_id": f"f{index}", "start": index, "end": index + 1, "reliable": True}
        for index in range(2)
    ]
    lines = [{"speaker": "SPEAKER_00", "start_timestamp": 0, "end_timestamp": 2}]

    edges = speaker_edges(scores, decisions, lines)

    assert len(edges) == 1
    assert (edges[0].source, edges[0].target) == (nodes[0].character_id, "SPEAKER_00")


def test_speaker_with_three_supported_candidates_stays_unresolved() -> None:
    alignments = [
        {"speaker": "SPEAKER_00", "character_id": character, "reliable": True}
        for character in ("c0001", "c0002", "c0003")
        for _ in range(2)
    ]
    assert link_speaker(alignments) == {}


def test_one_talknet_window_cannot_count_twice_across_dialogue_lines() -> None:
    scores = [{"track_id": "f0", "start": 0, "end": 1, "reliable": True}]
    decisions = [{"track_id": "f0", "character_id": "c0001"}]
    lines = [
        {"speaker": "SPEAKER_00", "start_timestamp": 0, "end_timestamp": 0.4},
        {"speaker": "SPEAKER_00", "start_timestamp": 0.5, "end_timestamp": 1},
    ]
    assert speaker_edges(scores, decisions, lines) == []
