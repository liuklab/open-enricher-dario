"""CPU helpers for the LLM judger: title parsing, weighting, report assembly.

No torch and no network. The GPU stage calls ``build_report`` after the model
returns a JSON document of per-dimension scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

from open_enricher.stages.judger_core import criteria as C


def title_from_filename(video_filename: str) -> str:
    """
    Derive a human-readable title from a video file name.

    Underscores and dots in the stem become spaces so that names such as
    ``Stranger_Things.S01E03.mkv`` become ``Stranger Things S01E03``.

    Args:
        video_filename: Exact file name under ``video_inputs``.

    Returns:
        Title string passed to the judger LLM. Never empty for a non-empty name.
    """
    stem = Path(video_filename).stem.strip()
    if not stem:
        return video_filename.strip() or "unknown"
    title = stem.replace("_", " ").replace(".", " ")
    return " ".join(title.split())


def clamp_score(value: Any) -> int | None:
    """
    Coerce a raw LLM score to an integer in ``[SCORE_MIN, SCORE_MAX]``.

    Args:
        value: Raw JSON value (int, float, or numeric string).

    Returns:
        Clamped int, or None when the value cannot be interpreted.
    """
    try:
        if isinstance(value, bool):
            return None
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    rounded = int(round(number))
    return max(C.SCORE_MIN, min(C.SCORE_MAX, rounded))


def weighted_overall(dimension_scores: Mapping[str, int]) -> Tuple[float, float]:
    """
    Combine per-dimension scores with ``DIMENSION_WEIGHTS``.

    Args:
        dimension_scores: Mapping of dimension name to integer score in 1-5.
            Missing dimensions are treated as SCORE_MIN.

    Returns:
        ``(overall_1_to_5, overall_normalized_0_to_1)``.
    """
    total = 0.0
    for name, weight in C.DIMENSION_WEIGHTS.items():
        score = dimension_scores.get(name, C.SCORE_MIN)
        total += weight * float(score)
    overall = round(total, 4)
    # Map [1, 5] -> [0, 1].
    normalized = round((overall - C.SCORE_MIN) / (C.SCORE_MAX - C.SCORE_MIN), 4)
    return overall, max(0.0, min(1.0, normalized))


def build_report(
    *,
    video_filename: str,
    title: str,
    profile_file: str,
    llm_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build the archived judger report from a parsed LLM JSON payload.

    Args:
        video_filename: Exact source video file name.
        title: Title string that was shown to the judger LLM.
        profile_file: Name of the LLM profile JSON that was scored.
        llm_payload: Parsed model output. Expected shape::

            {
              "dimensions": {
                "<name>": {"score": 1-5, "note": "..."},
                ...
              },
              "summary": "optional short overall comment"
            }

    Returns:
        Report dict with weighted ``overall``, ``overall_normalized``,
        ``usable``, per-dimension scores, weights, and method metadata.
        Does not call a model.
    """
    raw_dims = llm_payload.get("dimensions")
    if not isinstance(raw_dims, dict):
        raw_dims = {}

    dimensions: Dict[str, Dict[str, Any]] = {}
    scores_for_weight: Dict[str, int] = {}
    parse_errors: list[str] = []

    for name in C.DIMENSION_ORDER:
        entry = raw_dims.get(name)
        if not isinstance(entry, dict):
            entry = {}
        score = clamp_score(entry.get("score"))
        if score is None:
            parse_errors.append(f"missing_or_invalid_score:{name}")
            score = C.SCORE_MIN
        note = entry.get("note")
        if note is None:
            note = ""
        dimensions[name] = {
            "score": score,
            "note": str(note).strip(),
            "weight": C.DIMENSION_WEIGHTS[name],
        }
        scores_for_weight[name] = score

    overall, normalized = weighted_overall(scores_for_weight)
    fidelity = scores_for_weight["fidelity"]
    usable = overall >= C.USABLE_OVERALL_MIN and fidelity >= C.USABLE_FIDELITY_MIN

    summary = llm_payload.get("summary")
    if summary is None:
        summary = ""

    return {
        "video": video_filename,
        "title": title,
        "profile_file": profile_file,
        "dimensions": dimensions,
        "overall": overall,
        "overall_normalized": normalized,
        "usable": usable,
        "summary": str(summary).strip(),
        "weights": dict(C.DIMENSION_WEIGHTS),
        "score_scale": {"min": C.SCORE_MIN, "max": C.SCORE_MAX},
        "thresholds": {
            "usable_overall_min": C.USABLE_OVERALL_MIN,
            "usable_fidelity_min": C.USABLE_FIDELITY_MIN,
        },
        "method": {
            "kind": "llm_empty_context",
            "uses_synopsis": False,
            "uses_pipeline_evidence": False,
            "title_source": "video_filename",
        },
        "parse_errors": parse_errors,
    }


def failed_report(
    *,
    video_filename: str,
    title: str,
    profile_file: str | None,
    error: str,
) -> Dict[str, Any]:
    """
    Build a zeroed report when the judger cannot score the profile.

    Args:
        video_filename: Exact source video file name.
        title: Title derived from the file name.
        profile_file: Profile file name when known.
        error: Human-readable failure reason.

    Returns:
        Report with minimum scores, ``usable=False``, and the error string.
    """
    dimensions = {
        name: {
            "score": C.SCORE_MIN,
            "note": error,
            "weight": weight,
        }
        for name, weight in C.DIMENSION_WEIGHTS.items()
    }
    overall, normalized = weighted_overall(
        {name: C.SCORE_MIN for name in C.DIMENSION_ORDER}
    )
    return {
        "video": video_filename,
        "title": title,
        "profile_file": profile_file,
        "dimensions": dimensions,
        "overall": overall,
        "overall_normalized": normalized,
        "usable": False,
        "summary": error,
        "weights": dict(C.DIMENSION_WEIGHTS),
        "score_scale": {"min": C.SCORE_MIN, "max": C.SCORE_MAX},
        "thresholds": {
            "usable_overall_min": C.USABLE_OVERALL_MIN,
            "usable_fidelity_min": C.USABLE_FIDELITY_MIN,
        },
        "method": {
            "kind": "llm_empty_context",
            "uses_synopsis": False,
            "uses_pipeline_evidence": False,
            "title_source": "video_filename",
        },
        "error": error,
        "parse_errors": [],
    }
