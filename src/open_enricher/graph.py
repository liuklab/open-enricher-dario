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

import numpy as np


_SELF_RE = re.compile(
    r"\b(?i:i am|i'm|my name is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
)
_VOCATIVE_RE = re.compile(
    r"^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s*[,!:]",
)
_PROPER_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
# Sentence-initial closed-class words looked like names on the full HIMYM
# transcript (Okay, It, You, So). They are not character aliases.
_NAME_STOP = frozenset(
    {
        "a", "all", "an", "and", "are", "but", "bye", "for", "from", "goodbye", "hello",
        "her", "hey", "hi", "him",
        "his", "how", "if", "in", "is", "it", "its", "just", "let", "like", "man", "me",
        "my", "no", "not", "now", "of", "oh", "ok", "okay", "on", "or", "our",
        "right", "she", "so", "that", "the", "their", "them", "then", "there", "they",
        "this", "to", "uh", "um", "wait", "we", "well", "what", "when", "who", "why",
        "wow", "yeah", "yes", "you", "your",
    }
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
    samples: int = 1
    source_track_ids: tuple[str, ...] = ()


@dataclass
class CharacterNode:
    """A film-local identity. The alias may stay empty."""

    character_id: str
    embedding: list[float]
    weight: int = 1
    alias: str | None = None
    provisional: bool = False
    tracks: list[str] = field(default_factory=list)
    samples: int = 1


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


def _accept_name(value: str) -> str | None:
    """Return a title-cased name when none of its tokens are stopwords."""
    tokens = [part for part in value.split() if part]
    if not tokens or any(part.lower() in _NAME_STOP for part in tokens):
        return None
    return _title_name(value)


def interpret_mention(text: str) -> tuple[str, str | None]:
    """
    Classify one utterance without a text model.

    Returns ``(role, name)``. ``role`` is ``self_introduction``,
    ``addressee`` or ``third_party``. A vocative such as
    ``Richard, come here`` is an addressee and does not name the speaker.
    ``I am Richard Hendricks`` names the speaker. A capitalized closed-class
    word such as ``Okay`` or ``It`` is not a name.
    """
    stripped = text.strip()
    self_match = _SELF_RE.search(stripped)
    if self_match:
        name = _accept_name(self_match.group(1))
        if name:
            return "self_introduction", name
    vocative = _VOCATIVE_RE.match(stripped)
    if vocative:
        name = _accept_name(vocative.group(1))
        if name:
            return "addressee", name
    for proper in _PROPER_RE.finditer(stripped):
        # Capitalization alone is weak evidence at the start of an utterance.
        if not proper.start() and " " not in proper.group(1):
            continue
        name = _accept_name(proper.group(1))
        if name:
            return "third_party", name
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
    node_indexes: dict[str, int] = {}
    width = len(ordered[0].embedding) if ordered else 0
    uniform_width = width > 0 and all(len(track.embedding) == width for track in ordered)
    # Store unit vectors so scoring does not recompute every exemplar norm.
    exemplars = np.zeros((len(ordered), width)) if uniform_width else None

    for track in ordered:
        blocked = {
            track_to_character[other]
            for other in track.co_visible
            if other in track_to_character
        }
        best_index = None
        best = second = 0.0
        if exemplars is not None and nodes:
            vector = np.asarray(track.embedding)
            norm = np.linalg.norm(vector)
            if norm:
                vector = vector / norm
            scores = np.einsum("ij,j->i", exemplars[:len(nodes)], vector)
            for character_id in blocked:
                scores[node_indexes[character_id]] = -np.inf
            index = int(np.argmax(scores))
            if scores[index] != -np.inf:
                best_index = index
                best = float(scores[index])
                scores[index] = -np.inf
                runner_up = float(scores.max())
                second = runner_up if runner_up != -np.inf else 0.0
        elif exemplars is None:
            scored = sorted(
                (
                    (cosine(track.embedding, node.embedding), index)
                    for index, node in enumerate(nodes)
                    if node.character_id not in blocked
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if scored:
                best, best_index = scored[0]
                second = scored[1][0] if len(scored) > 1 else 0.0

        if best_index is not None and best >= threshold and (best - second) >= margin:
            node = nodes[best_index]
            node.tracks.append(track.track_id)
            node.weight += 1
            node.samples += max(1, track.samples)
            node.embedding = _average(node.embedding, track.embedding, node.weight)
            node.provisional = False
            decision = "merged"
        else:
            node = CharacterNode(
                character_id=f"c{len(nodes) + 1:04d}",
                embedding=list(track.embedding),
                tracks=[track.track_id],
                provisional=True,
                samples=max(1, track.samples),
            )
            node_indexes[node.character_id] = len(nodes)
            nodes.append(node)
            decision = "provisional" if best_index is not None else "new"
        if exemplars is not None:
            vector = np.asarray(node.embedding)
            norm = np.linalg.norm(vector)
            exemplars[node_indexes[node.character_id]] = vector / norm if norm else vector
        track_to_character[track.track_id] = node.character_id
        decisions.append(
            {
                "track_id": track.track_id,
                "character_id": node.character_id,
                "source_track_ids": list(track.source_track_ids),
                "decision": decision,
                "best": round(best, 6),
                "second": round(second, 6),
                "blocked": sorted(blocked),
            }
        )
    return nodes, decisions


def link_observations(
    tracks: list[TrackObservation],
    *,
    threshold: float = 0.6,
    gap: float = 2.0,
) -> list[TrackObservation]:
    """Collapse nearby samples of one face into a single track.

    Faces in the same frame stay on different tracks. A later sample joins
    the recent track with the highest cosine at or above ``threshold`` when
    it starts within ``gap`` seconds of that track. The returned tracks are
    what character assignment sees, so a person who stays in shot is one
    observation instead of one identity per frame.
    """
    ordered = sorted(tracks, key=lambda track: (track.start, track.track_id))
    groups: list[list[TrackObservation]] = []
    for track in ordered:
        if groups and abs(groups[-1][0].start - track.start) < 1e-3:
            groups[-1].append(track)
        else:
            groups.append([track])
    chains: list[dict[str, Any]] = []
    for group in groups:
        stamp = group[0].start
        active = [
            index
            for index, chain in enumerate(chains)
            if stamp - float(chain["end"]) <= gap
        ]
        pairs: list[tuple[float, int, int]] = []
        for local_index, track in enumerate(group):
            for chain_index in active:
                score = cosine(track.embedding, chains[chain_index]["embedding"])
                if score >= threshold:
                    pairs.append((score, local_index, chain_index))
        pairs.sort(key=lambda item: item[0], reverse=True)
        used_local: set[int] = set()
        used_chain: set[int] = set()
        assignment: dict[int, int] = {}
        for _score, local_index, chain_index in pairs:
            if local_index in used_local or chain_index in used_chain:
                continue
            used_local.add(local_index)
            used_chain.add(chain_index)
            assignment[local_index] = chain_index
        frame_chains: list[int] = []
        for local_index, track in enumerate(group):
            chain_index = assignment.get(local_index)
            if chain_index is None:
                chains.append(
                    {
                        "embedding": list(track.embedding),
                        "start": track.start,
                        "end": track.end,
                        "shot_id": track.shot_id,
                        "samples": max(1, track.samples),
                        "ids": list(track.source_track_ids or (track.track_id,)),
                        "co": set(),
                    }
                )
                chain_index = len(chains) - 1
            else:
                chain = chains[chain_index]
                incoming = max(1, track.samples)
                previous = int(chain["samples"])
                chain["samples"] = previous + incoming
                chain["embedding"] = [
                    (left * previous + right * incoming) / chain["samples"]
                    for left, right in zip(chain["embedding"], track.embedding)
                ]
                chain["end"] = max(float(chain["end"]), track.end)
                chain["ids"].extend(track.source_track_ids or (track.track_id,))
            frame_chains.append(chain_index)
        for left in frame_chains:
            for right in frame_chains:
                if left != right:
                    chains[left]["co"].add(right)
    pooled: list[TrackObservation] = []
    ids = [f"p{index + 1:04d}" for index in range(len(chains))]
    for index, chain in enumerate(chains):
        pooled.append(
            TrackObservation(
                track_id=ids[index],
                embedding=list(chain["embedding"]),
                start=float(chain["start"]),
                end=float(chain["end"]),
                shot_id=str(chain["shot_id"]),
                co_visible=tuple(ids[other] for other in sorted(chain["co"])),
                samples=int(chain["samples"]),
                source_track_ids=tuple(chain["ids"]),
            )
        )
    return pooled


def visible_characters(nodes: list[CharacterNode]) -> list[CharacterNode]:
    """Keep faces that recur or stay on screen. One-frame leftovers drop out.

    A kept node was merged across tracks, was sampled at least three times,
    or received an alias. The full HIMYM pass otherwise published thousands
    of single-frame provisional ids.
    """
    return [
        node
        for node in nodes
        if node.weight >= 2 or node.samples >= 3 or bool(node.alias)
    ]


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
        str(track_id): str(row["character_id"])
        for row in decisions
        if row.get("character_id")
        for track_id in (row.get("track_id"), *(row.get("source_track_ids") or ()))
        if track_id
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
                    "window": (start, end),
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
        groups: list[list[dict[str, Any]]] = []
        group_end = -math.inf
        for row in sorted(hits, key=lambda item: float(item.get("start") or 0)):
            start = float(row.get("start") or 0)
            end = float(row.get("end") or start)
            if not groups or start > group_end:
                groups.append([])
                group_end = end
            groups[-1].append(row)
            group_end = max(group_end, end)

        for group in groups:
            values = [
                float(row["score"])
                for row in group
                if isinstance(row.get("score"), (int, float))
            ]
            evidence_ids = list(dict.fromkeys(
                f"{row['track_id']}@{row['window'][0]:.3f}-{row['window'][1]:.3f}"
                for row in group
                if row.get("window") and row.get("track_id")
            ))
            edges.append(
                Edge(
                    "speaks",
                    character_id,
                    speaker,
                    float(group[0].get("start") or 0),
                    max(float(row.get("end") or row.get("start") or 0) for row in group),
                    evidence_ids or [speaker],
                    round(sum(values) / len(values), 4) if values else None,
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
    windows: dict[tuple[str, str], set[object]] = {}
    for index, row in enumerate(alignments):
        if not row.get("reliable"):
            continue
        speaker = str(row.get("speaker") or "")
        character_id = str(row.get("character_id") or "")
        if not speaker or speaker == "UNKNOWN" or not character_id:
            continue
        windows.setdefault((speaker, character_id), set()).add(row.get("window", index))
    candidates: dict[str, list[str]] = {}
    for (speaker, character_id), hits in windows.items():
        if len(hits) >= minimum_hits:
            candidates.setdefault(speaker, []).append(character_id)
    return {
        speaker: identities[0]
        for speaker, identities in candidates.items()
        if len(identities) == 1
    }


def visible_speech_edges(
    observations: list[TrackObservation],
    decisions: list[dict[str, Any]],
    speech_edges: list[Edge],
    kept_ids: set[str],
) -> list[Edge]:
    """Record speech with another published character visible in the same frame."""
    characters = {
        track_id: row["character_id"]
        for row in decisions
        if row.get("character_id") in kept_ids
        for track_id in row.get("source_track_ids") or ()
    }
    frames: dict[float, dict[str, str]] = {}
    for observation in observations:
        character = characters.get(observation.track_id)
        if character:
            frames.setdefault(observation.start, {})[character] = observation.track_id

    edges: list[Edge] = []
    for speech in speech_edges:
        if speech.source not in kept_ids or speech.status != "supported":
            continue
        for timestamp, visible in frames.items():
            if not speech.start <= timestamp <= speech.end or speech.source not in visible:
                continue
            for other, track_id in visible.items():
                if other == speech.source:
                    continue
                edges.append(
                    Edge(
                        "speaks_in_presence_of",
                        speech.source,
                        other,
                        timestamp,
                        timestamp,
                        [*speech.evidence_ids, visible[speech.source], track_id],
                        speech.confidence,
                        "supported",
                    )
                )
    return edges


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
    *,
    limit: int = 12,
) -> str:
    """One short paragraph of supported facts for the profile writer.

    Only the heaviest characters are listed. The HIMYM graph named every
    provisional face and the profile prompt no longer fit in GPU memory.
    """
    ranked = sorted(
        nodes,
        key=lambda node: (
            -(
                int(node["samples"])
                if isinstance(node.get("samples"), int)
                else len(node.get("tracks") or [])
            ),
            str(node.get("character_id") or ""),
        ),
    )
    shown = ranked[:limit]
    lines: list[str] = []
    for node in shown:
        alias = node.get("alias") or "unnamed"
        if isinstance(node.get("samples"), int):
            count = int(node["samples"])
        else:
            count = len(node.get("tracks") or [])
        lines.append(
            f"{node.get('character_id')} ({alias}) appears in {count} face sample(s)."
        )
    hidden = len(ranked) - len(shown)
    if hidden:
        lines.append(f"{hidden} further character(s) were left out.")
    supported = [edge for edge in edges if edge.get("status") == "supported"]
    for edge in sorted(supported, key=lambda edge: edge.get("kind") != "speaks_in_presence_of")[:8]:
        lines.append(
            f"{edge.get('source')} {edge.get('kind')} {edge.get('target')} "
            f"at {edge.get('start')}-{edge.get('end')}."
        )
    if not lines:
        return ""
    return " ".join(lines)
