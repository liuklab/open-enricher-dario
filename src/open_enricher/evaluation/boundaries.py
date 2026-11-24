"""Score archived BaSSL scene starts against independently annotated timestamps."""

import argparse
import json
import math
from pathlib import Path


def match_boundaries(predicted: list[float], reference: list[float], tolerance: float) -> dict:
    """Match sorted boundaries one-to-one within a fixed time tolerance."""
    predictions = iter(sorted(predicted))
    annotations = iter(sorted(reference))
    prediction = next(predictions, None)
    annotation = next(annotations, None)
    matched, missed, extra = [], [], []
    while prediction is not None and annotation is not None:
        if prediction < annotation - tolerance:
            extra.append(prediction)
            prediction = next(predictions, None)
        elif prediction > annotation + tolerance:
            missed.append(annotation)
            annotation = next(annotations, None)
        else:
            matched.append([prediction, annotation])
            prediction = next(predictions, None)
            annotation = next(annotations, None)
    if prediction is not None:
        extra.extend([prediction, *predictions])
    if annotation is not None:
        missed.extend([annotation, *annotations])
    true_positives = len(matched)
    return {
        "matches": matched,
        "false_splits": extra,
        "missed_boundaries": missed,
        "precision": true_positives / len(predicted) if predicted else None,
        "recall": true_positives / len(reference) if reference else None,
        "f1": 2 * true_positives / (len(predicted) + len(reference)) if predicted or reference else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bassl_archive", type=Path)
    parser.add_argument("reference", type=Path, help="JSON list of interior boundary times in seconds")
    parser.add_argument("--tolerance", type=float, default=1.0)
    args = parser.parse_args()
    archive = json.loads(args.bassl_archive.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    if archive.get("status") not in {"success", "degraded"} or archive.get("replaced_all"):
        parser.error("BaSSL did not produce this timeline")
    duration = archive["duration"]
    if not math.isfinite(args.tolerance) or args.tolerance < 0:
        parser.error("tolerance must be a finite nonnegative number")
    if not isinstance(reference, list) or any(
        type(time) not in (int, float) or not 0 < time < duration for time in reference
    ) or len(reference) != len(set(reference)):
        parser.error("reference must contain unique interior boundary times")
    predicted = [row["start"] for row in archive["intervals"] if 0 < row["start"] < duration]
    print(json.dumps(match_boundaries(predicted, reference, args.tolerance), indent=2))


if __name__ == "__main__":
    main()
