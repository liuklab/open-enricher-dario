"""Film-local character graph: tracks, aliases and timed edges.

Matching is cosine similarity against exemplars already seen in this
title. A merge needs a best score at or above the threshold and a margin
over the next candidate. Two tracks that are visible together are not
merged. Ambiguous tracks stay provisional. Name mentions stay distinct
from the film-local id, the face track and the diarized speaker label.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


_SELF_RE = re.compile(
    r"\b(?:i am|i'm|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
    re.IGNORECASE,
)
_VOCATIVE_RE = re.compile(
    r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*[,!:]",
)


def cosine(left: list[float], right: list[float]) -> float:
    """Cosine similarity. Orthogonal or empty vectors score 0."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass
class TrackObservation:
    """One face track pooled inside a shot."""

    track_id: str
    embedding: list[float]
    start: float
    end: float
    shot_id: str
    co_visible: tuple[str, ...] = ()


@dataclass
class CharacterNode:
    """A film-local identity. The alias may stay empty."""

    character_id: str
    embedding: list[float]
    weight: int = 1
    alias: str | None = None
    provisional: bool = False
    tracks: list[str] = field(default_factory=list)


@dataclass
class Edge:
    """One supported or unresolved relation."""

    kind: str
    source: str
    target: str
    start: float
    end: float
    evidence_ids: list[str]
    confidence: float | None
    status: str

    def key(self) -> tuple[Any, ...]:
        """Identity used to keep a resume from duplicating the same edge."""
        return (
            self.kind,
            self.source,
            self.target,
            round(self.start, 3),
            round(self.end, 3),
            tuple(self.evidence_ids),
            self.status,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "source": self.source,
            "target": self.target,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "status": self.status,
        }


def interpret_mention(text: str) -> tuple[str, str | None]:
    """
    Classify one utterance without a text model.

    Returns ``(role, name)``. ``role`` is ``self_introduction``,
    ``addressee`` or ``third_party``. A vocative such as
    ``Richard, come here`` is an addressee and does not name the speaker.
    ``I am Richard Hendricks`` names the speaker.
    """
    stripped = text.strip()
    self_match = _SELF_RE.search(stripped)
    if self_match:
        return "self_introduction", _title_name(self_match.group(1))
    vocative = _VOCATIVE_RE.match(stripped)
    if vocative:
        return "addressee", _title_name(vocative.group(1))
    proper = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", stripped)
    if proper:
        return "third_party", _title_name(proper.group(1))
    return "third_party", None


def _title_name(value: str) -> str:
    return " ".join(part.capitalize() for part in value.split())


def apply_alias(node: CharacterNode, role: str, name: str | None) -> str | None:
    """
    Update a speaker alias only for a self-introduction.

    An addressee or a third-party mention leaves the speaker alias as it
    was. Returns the alias after the update.
    """
    if role == "self_introduction" and name:
        node.alias = name
    return node.alias


def assign_character_ids(
    tracks: list[TrackObservation],
    *,
    threshold: float,
    margin: float,
) -> tuple[list[CharacterNode], list[dict[str, Any]]]:
    """
    Match tracks to film-local exemplars in chronological order.

    The observations behind each decision are returned so a merge can be
    revised. Co-visible tracks cannot join the same character.
    """
    ordered = sorted(tracks, key=lambda track: (track.start, track.track_id))
    nodes: list[CharacterNode] = []
    decisions: list[dict[str, Any]] = []
    track_to_character: dict[str, str] = {}
    for track in ordered:
        blocked = {
            track_to_character[other]
            for other in track.co_visible
            if other in track_to_character
        }
        scored: list[tuple[float, CharacterNode]] = []
        for node in nodes:
            if node.character_id in blocked:
                continue
            scored.append((cosine(track.embedding, node.embedding), node))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0][0] if scored else 0.0
        second = scored[1][0] if len(scored) > 1 else 0.0
        accept = bool(scored) and best >= threshold and (best - second) >= margin
        if accept:
            node = scored[0][1]
            node.tracks.append(track.track_id)
            node.weight += 1
            node.embedding = _average(node.embedding, track.embedding, node.weight)
            node.provisional = False
            decision = "merged"
        else:
            node = CharacterNode(
                character_id=f"c{len(nodes) + 1:04d}",
                embedding=list(track.embedding),
                tracks=[track.track_id],
                provisional=not accept,
            )
            nodes.append(node)
            decision = "provisional" if scored else "new"
        track_to_character[track.track_id] = node.character_id
        decisions.append(
            {
                "track_id": track.track_id,
                "character_id": node.character_id,
                "decision": decision,
                "best": round(best, 6),
                "second": round(second, 6),
                "blocked": sorted(blocked),
            }
        )
    return nodes, decisions


def _average(current: list[float], new: list[float], weight: int) -> list[float]:
    if weight <= 1:
        return list(new)
    previous = weight - 1
    return [
        (left * previous + right) / weight
        for left, right in zip(current, new)
    ]


def align_speaker_scores(
    scores: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    lines: list[Any],
) -> list[dict[str, Any]]:
    """Pair reliable TalkNet windows with overlapping diarized lines.

    A row that already names ``speaker`` and ``character_id`` is kept.
    Otherwise the face track is resolved through ``decisions``. One window
    can hit several lines. ``link_speaker`` still wants repeated hits.
    """
    track_to_character = {
        str(row.get("track_id")): str(row.get("character_id"))
        for row in decisions
        if row.get("track_id") and row.get("character_id")
    }
    aligned: list[dict[str, Any]] = []
    for score in scores:
        if not isinstance(score, dict) or not score.get("reliable"):
            continue
        character_id = str(score.get("character_id") or "")
        if not character_id:
            character_id = track_to_character.get(str(score.get("track_id") or ""), "")
        if not character_id:
            continue
        speaker = str(score.get("speaker") or "")
        if speaker and speaker != "UNKNOWN":
            aligned.append({**score, "speaker": speaker, "character_id": character_id, "reliable": True})
            continue
        start = float(score.get("start") or 0)
        end = float(score.get("end") or start)
        for line in lines:
            if not isinstance(line, dict):
                continue
            line_speaker = str(line.get("speaker") or "")
            if not line_speaker or line_speaker == "UNKNOWN":
                continue
            line_start = float(line.get("start_timestamp") or line.get("start") or 0)
            line_end = float(line.get("end_timestamp") or line.get("end") or line_start)
            if min(end, line_end) - max(start, line_start) <= 0:
                continue
            aligned.append(
                {
                    "speaker": line_speaker,
                    "character_id": character_id,
                    "reliable": True,
                    "track_id": score.get("track_id"),
                    "score": score.get("score"),
                    "start": max(start, line_start),
                    "end": min(end, line_end),
                }
            )
    return aligned


def speaker_edges(
    scores: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    lines: list[Any],
) -> list[Edge]:
    """Supported ``speaks`` edges for labels that clear ``link_speaker``."""
    alignments = align_speaker_scores(scores, decisions, lines)
    linked = link_speaker(alignments)
    edges: list[Edge] = []
    for speaker, character_id in linked.items():
        hits = [
            row
            for row in alignments
            if row.get("reliable")
            and str(row.get("speaker") or "") == speaker
            and str(row.get("character_id") or "") == character_id
        ]
        if not hits:
            continue
        start = min(float(row.get("start") or 0) for row in hits)
        end = max(float(row.get("end") or row.get("start") or 0) for row in hits)
        values = [
            float(row["score"])
            for row in hits
            if isinstance(row.get("score"), (int, float))
        ]
        confidence = round(sum(values) / len(values), 4) if values else None
        edges.append(
            Edge(
                "speaks",
                character_id,
                speaker,
                start,
                end,
                [speaker],
                confidence,
                "supported",
            )
        )
    return edges


def link_speaker(
    alignments: list[dict[str, Any]],
    *,
    minimum_hits: int = 2,
) -> dict[str, str]:
    """
    Connect a diarized speaker label to a character after repeated hits.

    ``alignments`` items carry ``speaker``, ``character_id`` and ``reliable``.
    A label with fewer than ``minimum_hits`` reliable alignments stays
    unresolved and is omitted.
    """
    counts: dict[tuple[str, str], int] = {}
    for row in alignments:
        if not row.get("reliable"):
            continue
        speaker = str(row.get("speaker") or "")
        character_id = str(row.get("character_id") or "")
        if not speaker or speaker == "UNKNOWN" or not character_id:
            continue
        counts[(speaker, character_id)] = counts.get((speaker, character_id), 0) + 1
    linked: dict[str, str] = {}
    for (speaker, character_id), hits in counts.items():
        if hits < minimum_hits:
            continue
        previous = linked.get(speaker)
        if previous is not None and previous != character_id:
            linked.pop(speaker, None)
            continue
        linked[speaker] = character_id
    return linked


def dedupe_edges(edges: list[Edge]) -> list[Edge]:
    """Drop edges that repeat the same key. Order of first occurrence is kept."""
    seen: set[tuple[Any, ...]] = set()
    unique: list[Edge] = []
    for edge in edges:
        key = edge.key()
        if key in seen:
            continue
        seen.add(key)
        unique.append(edge)
    return unique


def relations_for_interval(
    edges: list[dict[str, Any]],
    start: float,
    end: float,
    *,
    names: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """Return supported edges that overlap ``[start, end)``."""
    names = names or {}
    picked: list[dict[str, Any]] = []
    for edge in edges:
        if edge.get("status") != "supported":
            continue
        edge_start = float(edge.get("start") or 0)
        edge_end = float(edge.get("end") or edge_start)
        if min(end, edge_end) - max(start, edge_start) <= 0 and not (
            edge_start == edge_end and start <= edge_start < end
        ):
            continue
        row = dict(edge)
        source = str(edge.get("source") or "")
        if source in names and names[source]:
            row["source_alias"] = names[source]
        picked.append(row)
    return picked


def compact_account(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> str:
    """One short paragraph of supported facts for the profile writer."""
    lines: list[str] = []
    for node in nodes:
        alias = node.get("alias") or "unnamed"
        tracks = node.get("tracks") or []
        lines.append(
            f"{node.get('character_id')} ({alias}) appears in {len(tracks)} face track(s)."
        )
    supported = [edge for edge in edges if edge.get("status") == "supported"]
    for edge in supported[:12]:
        lines.append(
            f"{edge.get('source')} {edge.get('kind')} {edge.get('target')} "
            f"at {edge.get('start')}-{edge.get('end')}."
        )
    if not lines:
        return ""
    return " ".join(lines)
