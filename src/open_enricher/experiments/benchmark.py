"""Clean-context Cursor SDK benchmark. Persists only the weighted score JSON."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src.open_enricher.experiments.config import (
    BENCHMARK_MODEL,
    PROMPTS_DIR,
    SCORE_WEIGHTS,
)


def prompt_hash() -> str:
    text = (PROMPTS_DIR / "benchmark.md").read_bytes()
    return hashlib.sha256(text).hexdigest()[:16]


def _weighted_overall(scores: dict[str, Any]) -> tuple[float, float]:
    total = 0.0
    for key, weight in SCORE_WEIGHTS.items():
        entry = scores.get(key) or {}
        value = float(entry.get("score", 0))
        total += value * weight
    overall = round(total, 4)
    normalized = round((overall - 1.0) / 4.0, 4) if overall > 0 else 0.0
    return overall, max(0.0, min(1.0, normalized))


def _zero_score(reason: str) -> dict[str, Any]:
    dims = {
        key: {"score": 0, "note": reason}
        for key in SCORE_WEIGHTS
    }
    return {
        "dimensions": dims,
        "weights": dict(SCORE_WEIGHTS),
        "overall": 0.0,
        "overall_normalized": 0.0,
        "prompt_hash": prompt_hash(),
        "model": BENCHMARK_MODEL,
        "error": reason,
    }


def score_profile(
    profile: dict[str, Any] | None,
    summary_path: Path,
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run one clean-context Agent.prompt and return the weighted score document."""
    if not profile:
        return _zero_score("missing_profile")
    if not summary_path.is_file():
        return _zero_score(f"missing_summary:{summary_path}")

    key = api_key or os.environ.get("CURSOR_API_KEY", "").strip()
    if not key:
        return _zero_score("missing_CURSOR_API_KEY")

    prompt_text = (PROMPTS_DIR / "benchmark.md").read_text(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp(prefix="open-enricher-bench-"))
    try:
        (tmp / "profile.json").write_text(
            json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(summary_path, tmp / "episode_summary.txt")
        (tmp / "benchmark.md").write_text(prompt_text, encoding="utf-8")

        try:
            from cursor_sdk import Agent, AgentOptions, LocalAgentOptions
        except ImportError:
            return _zero_score("cursor_sdk_not_installed")

        instruction = (
            f"{prompt_text}\n\n"
            "The working directory contains only profile.json, episode_summary.txt, "
            "and this prompt. Write score.json in the working directory."
        )
        try:
            result = Agent.prompt(
                instruction,
                AgentOptions(
                    api_key=key,
                    model=BENCHMARK_MODEL,
                    local=LocalAgentOptions(cwd=str(tmp)),
                ),
            )
        except Exception as exc:
            return _zero_score(f"agent_startup_failed:{exc}")

        status = getattr(result, "status", None)
        if status == "error":
            return _zero_score(f"agent_run_failed:{getattr(result, 'id', '')}")

        score_path = tmp / "score.json"
        if not score_path.is_file():
            # Some agents print JSON in the result text instead of writing a file.
            text = getattr(result, "result", None) or ""
            if isinstance(text, str) and "{" in text:
                start = text.find("{")
                end = text.rfind("}")
                if end > start:
                    score_path.write_text(text[start : end + 1], encoding="utf-8")
            if not score_path.is_file():
                return _zero_score("score_json_missing")

        try:
            raw = json.loads(score_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return _zero_score(f"score_json_invalid:{exc}")

        dimensions: dict[str, Any] = {}
        for key_name in SCORE_WEIGHTS:
            entry = raw.get(key_name)
            if not isinstance(entry, dict):
                return _zero_score(f"missing_dimension:{key_name}")
            try:
                score_val = int(entry.get("score"))
            except (TypeError, ValueError):
                return _zero_score(f"bad_score:{key_name}")
            if score_val < 1 or score_val > 5:
                return _zero_score(f"score_out_of_range:{key_name}")
            dimensions[key_name] = {
                "score": score_val,
                "note": str(entry.get("note") or "").strip(),
            }

        overall, normalized = _weighted_overall(dimensions)
        return {
            "dimensions": dimensions,
            "weights": dict(SCORE_WEIGHTS),
            "overall": overall,
            "overall_normalized": normalized,
            "prompt_hash": prompt_hash(),
            "model": BENCHMARK_MODEL,
            "deterministic": False,
            "role": "benchmark",
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save_score(document: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
