"""Frame budgets for BaSSL scenes and for the visual-memory reserve.

The title budget is the number of frames variant 06 would have indexed,
including its short final window. History, when requested, is reserved
first. The remaining slots are split by duration or by a reproducible
importance score. Counts are pairs, because Qwen's temporal patch size
is 2. A same-boundaries duration mode separates segmentation from
importance weighting.
"""

from __future__ import annotations

from typing import Any

from open_enricher.stages.vlm_core.frames import FRAME_FACTOR, sampled_frame_count
from open_enricher.timeline import Interval, control_windows, fixed_windows


def control_frame_budget(
    duration: float,
    interval_seconds: int,
    *,
    video_fps: float,
    sample_fps: float,
    min_frames: int,
    max_frames: int,
) -> int:
    """
    Sum the frames the 06 sampler would keep for this title.

    Each control window is measured as ``round(length * video_fps)``
    frames at ``video_fps``, then passed through ``sampled_frame_count``.
    """
    if video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {video_fps}")
    total = 0
    for start, end in control_windows(duration, interval_seconds):
        length = max(0.0, end - start)
        frames = max(1, int(round(length * video_fps)))
        total += sampled_frame_count(
            frames,
            video_fps,
            sample_fps,
            min_frames,
            max_frames,
        )
    return total


def _largest_remainder(weights: list[float], total: int) -> list[int]:
    """Split ``total`` integer units in proportion to ``weights``."""
    if total <= 0 or not weights:
        return [0 for _ in weights]
    safe = [weight if weight > 0 else 0.0 for weight in weights]
    mass = sum(safe)
    if mass <= 0:
        safe = [1.0 for _ in weights]
        mass = float(len(weights))
    raw = [total * weight / mass for weight in safe]
    floors = [int(value) for value in raw]
    left = total - sum(floors)
    order = sorted(
        range(len(raw)),
        key=lambda index: (raw[index] - floors[index], -index),
        reverse=True,
    )
    for index in order[:left]:
        floors[index] += 1
    return floors


def _normalize(values: list[float]) -> list[float]:
    """Divide by the sum. A zero sum becomes a uniform vector."""
    mass = sum(value for value in values if value > 0)
    if mass <= 0:
        if not values:
            return []
        share = 1.0 / len(values)
        return [share for _ in values]
    return [max(0.0, value) / mass for value in values]


def importance_weights(
    durations: list[float],
    narrative: list[float] | None,
    cuts_per_minute: list[float] | None,
    visual_change: list[float] | None,
    *,
    pacing_cap: float,
) -> tuple[list[float], dict[str, Any]]:
    """
    Combine duration, narrative evidence and editing pace.

    Each cue is normalized inside the title. Pace is the mean of the
    available visual cues and is then capped at ``pacing_cap`` of the
    weight mass. Missing cues are recorded and dropped from the mix.
    """
    missing: list[str] = []
    named: list[tuple[str, float, list[float]]] = [
        ("duration", 0.5, _normalize([max(0.0, value) for value in durations])),
    ]
    if narrative is None:
        missing.append("narrative")
    else:
        named.append(
            ("narrative", 0.3, _normalize([max(0.0, value) for value in narrative]))
        )
    pace_sources: list[list[float]] = []
    if cuts_per_minute is None:
        missing.append("cuts_per_minute")
    else:
        pace_sources.append(_normalize([max(0.0, value) for value in cuts_per_minute]))
    if visual_change is None:
        missing.append("visual_change")
    else:
        pace_sources.append(_normalize([max(0.0, value) for value in visual_change]))
    if pace_sources:
        pace = [
            sum(source[index] for source in pace_sources) / len(pace_sources)
            for index in range(len(durations))
        ]
        named.append(("pace", 0.2, _normalize(pace)))
    else:
        missing.append("pace")
    weight_sum = sum(weight for _, weight, _ in named)
    shares: list[tuple[float, list[float]]] = []
    pace_mass = 0.0
    for name, weight, values in named:
        share = weight / weight_sum
        if name == "pace" and pacing_cap >= 0 and share > pacing_cap:
            share = pacing_cap
        if name == "pace":
            pace_mass = share
        shares.append((share, values))
    share_sum = sum(share for share, _ in shares) or 1.0
    combined = [0.0 for _ in durations]
    for share, values in shares:
        adjusted = share / share_sum
        for index, value in enumerate(values):
            combined[index] += adjusted * value
    record = {
        "missing": missing,
        "pacing_cap": pacing_cap,
        "pace_mass": round(pace_mass, 6),
        "weights": [round(value, 6) for value in combined],
    }
    return combined, record


def group_for_budget(
    intervals: list[Interval],
    slot_pairs: int,
    *,
    min_frames: int,
) -> list[list[Interval]]:
    """
    Group adjacent intervals until each group can hold ``min_frames``.

    ``slot_pairs`` counts frame pairs still available for current evidence.
    """
    if not intervals:
        return []
    minimum = min_frames if min_frames >= FRAME_FACTOR else FRAME_FACTOR
    if minimum % FRAME_FACTOR:
        minimum -= minimum % FRAME_FACTOR
    capacity = max(0, slot_pairs * FRAME_FACTOR)
    if capacity < minimum:
        return [list(intervals)]
    max_groups = max(1, capacity // minimum)
    if len(intervals) <= max_groups:
        return [[item] for item in intervals]
    # Merge the shortest adjacent pair until the group count fits.
    pending = [[item] for item in intervals]
    while len(pending) > max_groups:
        sizes = [
            sum(item.end - item.start for item in group)
            for group in pending
        ]
        pair_index = min(
            range(len(pending) - 1),
            key=lambda index: (sizes[index] + sizes[index + 1], index),
        )
        merged = pending[pair_index] + pending[pair_index + 1]
        pending = pending[:pair_index] + [merged] + pending[pair_index + 2 :]
    return pending


def _spread(start: float, end: float, count: int, events: list[float]) -> list[float]:
    """Place ``count`` timestamps inside ``[start, end)``, near events first."""
    if count <= 0 or end <= start:
        return []
    span = end - start
    chosen: list[float] = []
    for event in events:
        if start <= event < end:
            chosen.append(event)
        if len(chosen) >= count:
            break
    if len(chosen) >= count:
        return sorted(chosen)[:count]
    remaining = count - len(chosen)
    for step in range(remaining):
        # Stay strictly inside the half-open interval.
        fraction = (step + 0.5) / remaining
        stamp = start + span * fraction
        if stamp >= end:
            stamp = end - min(1e-3, span / 2)
        chosen.append(stamp)
    unique = sorted(set(round(stamp, 3) for stamp in chosen))
    while len(unique) < count:
        unique.append(round(min(end - 1e-3, unique[-1] + 1e-3), 3))
    return unique[:count]


def allocate_frames(
    intervals: list[Interval],
    *,
    budget: int,
    history_fraction: float,
    mode: str,
    min_frames: int = 2,
    max_frames_per_call: int = 40,
    max_seconds: float = 300.0,
    narrative: list[float] | None = None,
    cuts_per_minute: list[float] | None = None,
    visual_change: list[float] | None = None,
    events: list[list[float]] | None = None,
    pacing_cap: float = 0.35,
    scoring_failed: bool = False,
) -> dict[str, Any]:
    """
    Build the Qwen frame plan for one title.

    Args:
        intervals: Validated scenes or fixed windows, in order.
        budget: Frames variant 06 would have used for the title.
        history_fraction: Share reserved for memory images. Zero keeps
            every slot on the current interval.
        mode: ``adaptive`` or ``duration``. Adaptive falls back to
            duration when ``scoring_failed`` is set.
        min_frames: Coverage reserve per group, before proportional fill.
        max_frames_per_call: Split a long group into bounded Qwen calls.
        max_seconds: Split a group that is longer than this.
        narrative: Optional per-interval narrative scores, aligned.
        cuts_per_minute: Optional editing-pace cue.
        visual_change: Optional sampled-image change cue.
        events: Optional supported timestamps inside each interval.
        pacing_cap: Maximum share of the weight that pace may take.
        scoring_failed: Force the duration weights and record the reason.

    Returns:
        A JSON-ready plan. ``groups`` is what the VLM iterates. Scene ids
        with no sampled frame are listed in ``unsampled_scene_ids``.
    """
    if mode not in {"adaptive", "duration"}:
        raise ValueError(f"mode must be adaptive or duration, got {mode}")
    pair_budget = max(0, budget - (budget % FRAME_FACTOR))
    history_frames = 0
    if history_fraction > 0 and pair_budget > 0:
        history_frames = int(pair_budget * history_fraction)
        history_frames -= history_frames % FRAME_FACTOR
        # Keep at least one pair for the current scene when the budget allows.
        if history_frames >= pair_budget and pair_budget >= FRAME_FACTOR:
            history_frames = pair_budget - FRAME_FACTOR
    current = pair_budget - history_frames
    durations = [max(0.0, item.end - item.start) for item in intervals]
    effective = "duration" if scoring_failed or mode == "duration" else "adaptive"
    weight_record: dict[str, Any] = {"missing": [], "weights": [], "pacing_cap": pacing_cap}
    if effective == "duration":
        weights = _normalize(durations)
        if scoring_failed:
            weight_record["missing"] = ["importance"]
        weight_record["weights"] = [round(value, 6) for value in weights]
    else:
        weights, weight_record = importance_weights(
            durations,
            narrative,
            cuts_per_minute,
            visual_change,
            pacing_cap=pacing_cap,
        )
    reserve = min_frames if min_frames % FRAME_FACTOR == 0 else min_frames - 1
    if reserve < FRAME_FACTOR:
        reserve = FRAME_FACTOR
    groups_of = group_for_budget(intervals, current // FRAME_FACTOR, min_frames=reserve)
    group_weights: list[float] = []
    group_durations: list[float] = []
    for group in groups_of:
        ids = {item.id for item in group}
        indexes = [index for index, item in enumerate(intervals) if item.id in ids]
        group_weights.append(sum(weights[index] for index in indexes) if weights else 1.0)
        group_durations.append(sum(durations[index] for index in indexes))
    pair_slots = current // FRAME_FACTOR
    minimum_pairs = reserve // FRAME_FACTOR
    base = [0 for _ in groups_of]
    if groups_of and pair_slots >= len(groups_of) * minimum_pairs:
        base = [minimum_pairs for _ in groups_of]
        extra = _largest_remainder(group_weights, pair_slots - sum(base))
        base = [left + right for left, right in zip(base, extra)]
    elif groups_of:
        base = _largest_remainder(group_weights, pair_slots)
    counts = [pairs * FRAME_FACTOR for pairs in base]
    event_rows = events or [[] for _ in intervals]
    calls: list[dict[str, Any]] = []
    unsampled: list[str] = []
    for group_index, (group, count) in enumerate(zip(groups_of, counts), start=1):
        scene_ids = [item.id for item in group]
        if count <= 0:
            unsampled.extend(scene_ids)
        start = group[0].start
        end = group[-1].end
        group_events: list[float] = []
        for item in group:
            if item.id in {interval.id for interval in intervals}:
                position = next(
                    index for index, interval in enumerate(intervals) if interval.id == item.id
                )
                group_events.extend(event_rows[position] if position < len(event_rows) else [])
        pieces = _split_call(start, end, count, max_frames_per_call, max_seconds)
        for piece_index, (piece_start, piece_end, piece_count) in enumerate(pieces, start=1):
            stamps = _spread(piece_start, piece_end, piece_count, group_events)
            calls.append(
                {
                    "group_id": f"g{group_index:04d}",
                    "call_id": f"g{group_index:04d}c{piece_index:02d}",
                    "scene_ids": scene_ids,
                    "parent_id": group[0].parent_id,
                    "start": round(piece_start, 3),
                    "end": round(piece_end, 3),
                    "frames": piece_count,
                    "timestamps": stamps,
                    "no_frame": piece_count == 0,
                    "duration": round(group_durations[group_index - 1], 3),
                }
            )
    return {
        "budget": budget,
        "paired_budget": pair_budget,
        "history_frames": history_frames,
        "current_frames": current,
        "mode": effective,
        "requested_mode": mode,
        "scoring_failed": scoring_failed,
        "weights": weight_record,
        "groups": calls,
        "unsampled_scene_ids": unsampled,
        "allocated_frames": sum(call["frames"] for call in calls),
    }


def _split_call(
    start: float,
    end: float,
    count: int,
    max_frames: int,
    max_seconds: float,
) -> list[tuple[float, float, int]]:
    """Split one group into bounded Qwen calls that keep the same scene ids."""
    length = max(0.0, end - start)
    if count <= 0:
        return [(start, end, 0)]
    max_frames -= max_frames % FRAME_FACTOR
    if max_frames < FRAME_FACTOR:
        max_frames = FRAME_FACTOR
    parts = 1
    if length > max_seconds:
        parts = int(length // max_seconds) + (1 if length % max_seconds else 0)
    if count > max_frames:
        parts = max(parts, (count + max_frames - 1) // max_frames)
    if parts <= 1:
        return [(start, end, count)]
    pair_counts = _largest_remainder([1.0] * parts, count // FRAME_FACTOR)
    frames = [pairs * FRAME_FACTOR for pairs in pair_counts]
    bounds: list[tuple[float, float, int]] = []
    cursor = start
    for index, frame_count in enumerate(frames):
        if index == len(frames) - 1:
            piece_end = end
        else:
            piece_end = start + length * (index + 1) / parts
        bounds.append((cursor, piece_end, frame_count))
        cursor = piece_end
    return bounds


def uniform_timestamps(start: float, end: float, count: int) -> list[float]:
    """Even timestamps inside one span. Used when importance sampling falls back."""
    return _spread(start, end, count, [])
