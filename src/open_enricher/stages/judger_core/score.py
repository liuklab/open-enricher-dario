"""Pure, deterministic scoring of a VideoProfile against upstream evidence.

No randomness, no network, no GPU. Same inputs always produce the same report.
Dimension definitions and literature mapping live in ``docs/judger_references.md``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from pydantic import ValidationError

from open_enricher.stages.judger_core import criteria as C
from open_enricher.stages.video_profile import VideoProfile

# Word tokens: letters and digits only after lowercasing.
_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Hex color like #rrggbb or #rgb.
_HEX_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def round_score(value: float) -> float:
    """Clamp to [0, 1] and round to four decimal places for stable JSON."""
    return round(max(0.0, min(1.0, value)), 4)


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens; light plural fold (trailing 's' only)."""
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
            tokens.append(raw[:-1])
        else:
            tokens.append(raw)
    return tokens


def content_tokens(text: str) -> List[str]:
    """Tokens after removing English stopwords (Ochoa information content)."""
    return [t for t in tokenize(text) if t not in C.ENGLISH_STOPWORDS]


def is_placeholder(value: str) -> bool:
    """True when the string is empty or a known schema/filler placeholder."""
    return value.strip().lower() in C.PLACEHOLDER_VALUES


def field_as_text(value: Any) -> str:
    """Flatten a leaf value (str or list of str) into one string for scoring."""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    if value is None:
        return ""
    return str(value)


def get_by_path(profile: Dict[str, Any], path: str) -> Any:
    """Read a dotted path such as ``recsys_metadata.primary_genre``."""
    node: Any = profile
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Evidence bag
# ---------------------------------------------------------------------------

def _walk_strings(obj: Any, into: List[str]) -> None:
    """Collect every string leaf from nested JSON for the evidence bag."""
    if isinstance(obj, str):
        into.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _walk_strings(value, into)
    elif isinstance(obj, list):
        for item in obj:
            _walk_strings(item, into)


def build_evidence_bag(evidence: Dict[str, Any]) -> Set[str]:
    """Token set drawn from audio / color / vlm JSON (grounding proxy)."""
    strings: List[str] = []
    for payload in evidence.values():
        _walk_strings(payload, strings)
    bag: Set[str] = set()
    for text in strings:
        bag.update(content_tokens(text))
    return bag


def extract_palette_rgbs(color_payload: Any) -> List[Tuple[int, int, int]]:
    """Parse ``#rrggbb`` values from the color-stage JSON list."""
    rgbs: List[Tuple[int, int, int]] = []
    if not isinstance(color_payload, list):
        return rgbs
    for entry in color_payload:
        if not isinstance(entry, dict):
            continue
        palette = entry.get("color_palette", [])
        if not isinstance(palette, list):
            continue
        for hex_color in palette:
            if not isinstance(hex_color, str):
                continue
            match = _HEX_RE.search(hex_color)
            if not match:
                continue
            digits = match.group(1)
            if len(digits) == 3:
                digits = "".join(ch * 2 for ch in digits)
            rgbs.append(
                (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
            )
    return rgbs


def _rgb_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Per-field checks
# ---------------------------------------------------------------------------

def check_presence(value: Any) -> float:
    """Completeness: non-empty after trim; placeholders score 0."""
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        if not items:
            return 0.0
        # All list items must be non-placeholders.
        return 1.0 if all(not is_placeholder(item) for item in items) else 0.0
    text = field_as_text(value).strip()
    if not text or is_placeholder(text):
        return 0.0
    return 1.0


def check_cardinality(path: str, value: Any) -> Optional[float]:
    """List length inside the declared window; strings skip this check."""
    if path not in C.LIST_CARDINALITY:
        return None
    if not isinstance(value, list):
        return 0.0
    low, high = C.LIST_CARDINALITY[path]
    n = len(value)
    return 1.0 if low <= n <= high else 0.0


def check_conformance(path: str, value: Any) -> Optional[float]:
    """Closed vocabulary for pacing / genre / age; free-text fields skip."""
    vocab = C.CONFORMANCE_FIELDS.get(path)
    if vocab is None:
        return None
    text = field_as_text(value).strip()
    if not text:
        return 0.0
    # Case-insensitive membership against the fixed vocabulary.
    lowered = {entry.lower(): entry for entry in vocab}
    return 1.0 if text.lower() in lowered else 0.0


def check_specificity(value: Any) -> float:
    """Information content: content tokens, generic-filler and duplicate penalties."""
    if isinstance(value, list):
        items = [str(item).strip() for item in value]
        if not items:
            return 0.0
        # Duplicate list items (case-folded) reduce specificity.
        lowered = [item.lower() for item in items]
        unique_ratio = len(set(lowered)) / len(lowered)
        text = " ".join(items)
    else:
        unique_ratio = 1.0
        text = field_as_text(value)

    tokens = content_tokens(text)
    if not tokens:
        return 0.0

    distinct = set(tokens)
    filler_count = sum(1 for t in distinct if t in C.GENERIC_FILLERS)
    filler_ratio = filler_count / max(len(distinct), 1)
    # Reward having at least SPECIFICITY_MIN_CONTENT_TOKENS distinct tokens.
    richness = min(1.0, len(distinct) / float(C.SPECIFICITY_MIN_CONTENT_TOKENS))
    score = richness * (1.0 - 0.5 * filler_ratio) * unique_ratio
    return round_score(score)


def check_grounding(value: Any, evidence_bag: Set[str]) -> float:
    """Fraction of content tokens that also appear in the evidence bag."""
    tokens = content_tokens(field_as_text(value))
    if not tokens:
        return 0.0
    if not evidence_bag:
        # No upstream evidence: cannot support any claim.
        return 0.0
    hits = sum(1 for t in tokens if t in evidence_bag)
    return round_score(hits / len(tokens))


def score_field(
    path: str,
    value: Any,
    evidence_bag: Set[str],
) -> Dict[str, Any]:
    """Run applicable checks for one leaf and return score, checks, notes."""
    checks: Dict[str, float] = {}
    notes: List[str] = []

    presence = check_presence(value)
    checks["presence"] = presence
    if presence < 1.0:
        notes.append("empty_or_placeholder")

    cardinality = check_cardinality(path, value)
    if cardinality is not None:
        checks["cardinality"] = cardinality
        if cardinality < 1.0:
            notes.append("cardinality_out_of_range")

    conformance = check_conformance(path, value)
    if conformance is not None:
        checks["conformance"] = conformance
        if conformance < 1.0:
            notes.append("outside_closed_vocabulary")

    specificity = check_specificity(value)
    checks["specificity"] = specificity
    if specificity < 0.5:
        notes.append("low_information_content")

    grounding = check_grounding(value, evidence_bag)
    checks["grounding"] = grounding
    if grounding < 0.4:
        notes.append("weak_evidence_overlap")

    # Field score is the mean of the checks that applied.
    field_score = round_score(sum(checks.values()) / len(checks))
    return {
        "score": field_score,
        "checks": {k: round_score(v) for k, v in checks.items()},
        "evidence_sources": [],  # filled by the caller with provenance labels
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Cross-field consistency
# ---------------------------------------------------------------------------

def _jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard over content tokens of two string lists (softer than exact items)."""
    ta: Set[str] = set()
    tb: Set[str] = set()
    for item in a:
        text = str(item).strip()
        if text:
            ta.update(content_tokens(text))
    for item in b:
        text = str(item).strip()
        if text:
            tb.update(content_tokens(text))
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def _band_score(value: float, low: float, high: float) -> float:
    """1.0 inside [low, high]; linear falloff to 0 outside."""
    if low <= value <= high:
        return 1.0
    if value < low:
        return round_score(value / low) if low > 0 else 0.0
    # value > high: distance from high toward 1.0
    if high >= 1.0:
        return 1.0
    return round_score(max(0.0, 1.0 - (value - high) / (1.0 - high)))


def score_consistency(
    profile: Dict[str, Any],
    color_payload: Any,
) -> Tuple[float, List[str]]:
    """Cross-field rules: tags↔topics Jaccard, sub_genres vs genre, color words."""
    notes: List[str] = []
    parts: List[float] = []

    tags = get_by_path(profile, "search_and_discovery.semantic_tags") or []
    topics = get_by_path(profile, "search_and_discovery.main_topics") or []
    if isinstance(tags, list) and isinstance(topics, list) and tags and topics:
        j = _jaccard(tags, topics)
        band = _band_score(j, C.TAGS_TOPICS_JACCARD_LOW, C.TAGS_TOPICS_JACCARD_HIGH)
        parts.append(band)
        if band < 1.0:
            notes.append(f"tags_topics_jaccard={round_score(j)}")
    else:
        parts.append(0.0)
        notes.append("tags_or_topics_missing")

    primary = str(get_by_path(profile, "recsys_metadata.primary_genre") or "").strip().lower()
    sub_genres = get_by_path(profile, "recsys_metadata.sub_genres") or []
    if isinstance(sub_genres, list) and sub_genres and primary:
        # Fail when every sub-genre is only a repeat of the primary genre.
        only_repeat = all(str(s).strip().lower() == primary for s in sub_genres)
        parts.append(0.0 if only_repeat else 1.0)
        if only_repeat:
            notes.append("sub_genres_repeat_primary")
    elif isinstance(sub_genres, list) and sub_genres:
        parts.append(1.0)
    else:
        parts.append(0.0)
        notes.append("sub_genres_missing")

    # Color-word rule: skip when no color words or no color evidence.
    visual = str(get_by_path(profile, "content_summary.visual_style") or "")
    named = [
        name
        for name in C.COLOR_NAME_TO_RGB
        if re.search(rf"\b{re.escape(name)}\b", visual.lower())
    ]
    palette = extract_palette_rgbs(color_payload)
    if named and palette:
        matched = False
        for name in named:
            target = C.COLOR_NAME_TO_RGB[name]
            if any(
                _rgb_distance(target, rgb) <= C.COLOR_MATCH_MAX_DISTANCE for rgb in palette
            ):
                matched = True
                break
        parts.append(1.0 if matched else 0.0)
        if not matched:
            notes.append("visual_style_color_mismatch")
    # else: rule skipped; do not append a score

    if not parts:
        return 0.0, notes
    return round_score(sum(parts) / len(parts)), notes


# ---------------------------------------------------------------------------
# Dimension aggregation
# ---------------------------------------------------------------------------

def _weighted_mean(pairs: Sequence[Tuple[float, float]]) -> float:
    """Weight pairs of (score, weight); empty → 0."""
    total_w = sum(w for _, w in pairs)
    if total_w <= 0:
        return 0.0
    return round_score(sum(s * w for s, w in pairs) / total_w)


def _field_evidence_sources(
    path: str,
    value: Any,
    evidence: Dict[str, Any],
    evidence_bags: Dict[str, Set[str]],
) -> List[str]:
    """Which upstream stages share at least one content token with this field."""
    tokens = set(content_tokens(field_as_text(value)))
    if not tokens:
        return []
    sources: List[str] = []
    for stage_name in ("audio", "vlm", "color"):
        if stage_name not in evidence:
            continue
        bag = evidence_bags.get(stage_name, set())
        if tokens & bag:
            sources.append(stage_name)
    return sources


def score_profile(
    profile_data: Any,
    evidence: Optional[Dict[str, Any]] = None,
    *,
    video: str = "",
    profile_file: str = "",
) -> Dict[str, Any]:
    """Score a profile dict (or raw JSON) against optional upstream evidence.

    Always returns a full report dict. If the payload fails ``VideoProfile``
    validation, ``accessibility`` is 0 and ``overall`` is 0.
    """
    evidence = evidence or {}
    evidence_present = {
        "audio": "audio" in evidence,
        "color": "color" in evidence,
        "vlm": "vlm" in evidence,
    }

    # ---- accessibility gate -------------------------------------------------
    accessibility = 0.0
    profile: Dict[str, Any] = {}
    validation_error: Optional[str] = None
    try:
        if isinstance(profile_data, str):
            validated = VideoProfile.model_validate_json(profile_data)
        elif isinstance(profile_data, VideoProfile):
            validated = profile_data
        else:
            validated = VideoProfile.model_validate(profile_data)
        profile = validated.model_dump()
        accessibility = 1.0
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as err:
        validation_error = str(err)
        # Keep a best-effort dict so field paths can still be inspected.
        if isinstance(profile_data, dict):
            profile = profile_data
        elif isinstance(profile_data, str):
            try:
                parsed = json.loads(profile_data)
                if isinstance(parsed, dict):
                    profile = parsed
            except json.JSONDecodeError:
                profile = {}

    evidence_bag = build_evidence_bag(evidence)
    evidence_bags = {
        name: build_evidence_bag({name: payload})
        for name, payload in evidence.items()
    }

    fields_report: Dict[str, Any] = {}
    completeness_pairs: List[Tuple[float, float]] = []
    conformance_pairs: List[Tuple[float, float]] = []
    specificity_pairs: List[Tuple[float, float]] = []
    grounding_pairs: List[Tuple[float, float]] = []
    provenance_hits = 0
    provenance_total = 0

    for path in C.FIELD_PATHS:
        weight = C.FIELD_WEIGHTS[path]
        value = get_by_path(profile, path)
        entry = score_field(path, value, evidence_bag)
        sources = _field_evidence_sources(path, value, evidence, evidence_bags)
        entry["evidence_sources"] = sources
        fields_report[path] = entry

        checks = entry["checks"]
        # Completeness = presence (+ cardinality when applicable).
        comp_parts = [checks["presence"]]
        if "cardinality" in checks:
            comp_parts.append(checks["cardinality"])
        completeness_pairs.append((sum(comp_parts) / len(comp_parts), weight))

        if "conformance" in checks:
            conformance_pairs.append((checks["conformance"], weight))

        specificity_pairs.append((checks["specificity"], weight))
        grounding_pairs.append((checks["grounding"], weight))

        provenance_total += 1
        if sources:
            provenance_hits += 1

    consistency_score, consistency_notes = score_consistency(
        profile, evidence.get("color")
    )
    if consistency_notes:
        # Attach cross-field notes under a synthetic key for the report.
        fields_report["_consistency"] = {
            "score": consistency_score,
            "checks": {},
            "evidence_sources": [],
            "notes": consistency_notes,
        }

    dimensions = {
        "completeness": {
            "score": _weighted_mean(completeness_pairs),
            "weight": C.DIMENSION_WEIGHTS["completeness"],
        },
        "conformance": {
            "score": _weighted_mean(conformance_pairs) if conformance_pairs else 0.0,
            "weight": C.DIMENSION_WEIGHTS["conformance"],
        },
        "consistency": {
            "score": consistency_score,
            "weight": C.DIMENSION_WEIGHTS["consistency"],
        },
        "specificity": {
            "score": _weighted_mean(specificity_pairs),
            "weight": C.DIMENSION_WEIGHTS["specificity"],
        },
        "grounding": {
            "score": _weighted_mean(grounding_pairs),
            "weight": C.DIMENSION_WEIGHTS["grounding"],
        },
        "provenance": {
            "score": round_score(
                provenance_hits / provenance_total if provenance_total else 0.0
            ),
            "weight": C.DIMENSION_WEIGHTS["provenance"],
        },
        "accessibility": {
            "score": accessibility,
            "weight": C.DIMENSION_WEIGHTS["accessibility"],
        },
    }

    if accessibility < 1.0:
        overall = 0.0
    else:
        overall_pairs = [
            (dimensions[name]["score"], dimensions[name]["weight"])
            for name in (
                "completeness",
                "conformance",
                "consistency",
                "specificity",
                "grounding",
                "provenance",
            )
        ]
        overall = _weighted_mean(overall_pairs)

    usable = (
        accessibility == 1.0
        and dimensions["completeness"]["score"] >= C.USABLE_COMPLETENESS_MIN
        and dimensions["grounding"]["score"] >= C.USABLE_GROUNDING_MIN
        and overall >= C.USABLE_OVERALL_MIN
    )

    report: Dict[str, Any] = {
        "video": video,
        "profile_file": profile_file,
        "evidence_present": evidence_present,
        "fields": fields_report,
        "dimensions": dimensions,
        "overall": overall,
        "usable": usable,
        "thresholds": {
            "completeness": C.USABLE_COMPLETENESS_MIN,
            "grounding": C.USABLE_GROUNDING_MIN,
            "overall": C.USABLE_OVERALL_MIN,
        },
        "method": {
            "deterministic": True,
            "sources": list(C.METHOD_SOURCES),
            "limitations": C.METHOD_LIMITATIONS,
        },
    }
    if validation_error is not None:
        report["validation_error"] = validation_error
    return report
