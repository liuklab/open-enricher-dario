"""Character identity, vocatives, and edge de-duplication."""

from __future__ import annotations

from open_enricher.graph import (
    CharacterNode,
    Edge,
    TrackObservation,
    apply_alias,
    assign_character_ids,
    dedupe_edges,
    interpret_mention,
    link_speaker,
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
