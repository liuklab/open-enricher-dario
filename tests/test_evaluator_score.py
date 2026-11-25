"""GPU-free and API-free tests for the external evaluator helpers.

Does not load torch or call Cursor. Run from the repository root:

    PYTHONPATH=src python -m pytest tests/test_evaluator_score.py -q
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from open_enricher.evaluation import criteria as C
from open_enricher.evaluation.prompt import system_prompt, user_prompt
from open_enricher.evaluation.providers import PROVIDERS, get_provider
from open_enricher.evaluation.run import evaluate_video
from open_enricher.evaluation.score import (
    build_report,
    failed_report,
    title_from_filename,
    weighted_overall,
)
from open_enricher.experiments.config import PIPELINE_STAGES


class FakeProvider:
    """Deterministic provider for unit tests."""

    name = "fake"
    model = "fake-model"

    def __init__(self, payload: dict | None = None, *, fail: bool = False) -> None:
        self.payload = payload or {
            "dimensions": {
                "completeness": {"score": 5, "note": "full"},
                "specificity": {"score": 4, "note": "ok"},
                "fidelity": {"score": 5, "note": "matches"},
                "consistency": {"score": 4, "note": "ok"},
                "recsys_usefulness": {"score": 5, "note": "useful"},
            },
            "summary": "Strong profile.",
        }
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if self.fail:
            raise RuntimeError("provider boom")
        return json.dumps(self.payload)


class TestEvaluatorHelpers(unittest.TestCase):
    def test_pipeline_stages_exclude_evaluator(self) -> None:
        self.assertEqual(PIPELINE_STAGES, ("audio", "dialogue", "vlm", "llm"))
        self.assertNotIn("evaluator", PIPELINE_STAGES)

    def test_title_from_filename(self) -> None:
        self.assertEqual(
            title_from_filename("Stranger_Things.S01E03.mkv"),
            "Stranger Things S01E03",
        )
        self.assertEqual(title_from_filename("Suits.S01E02.mkv"), "Suits S01E02")

    def test_weights_sum_to_one(self) -> None:
        self.assertAlmostEqual(sum(C.DIMENSION_WEIGHTS.values()), 1.0, places=9)

    def test_weighted_overall(self) -> None:
        scores = {name: 5 for name in C.DIMENSION_ORDER}
        overall, normalized = weighted_overall(scores)
        self.assertEqual(overall, 5.0)
        self.assertEqual(normalized, 1.0)

        low = {name: 1 for name in C.DIMENSION_ORDER}
        overall_low, normalized_low = weighted_overall(low)
        self.assertEqual(overall_low, 1.0)
        self.assertEqual(normalized_low, 0.0)

    def test_build_report_applies_weights_and_synopsis_flag(self) -> None:
        payload = {
            "dimensions": {
                "completeness": {"score": 5, "note": "full"},
                "specificity": {"score": 4, "note": "ok"},
                "fidelity": {"score": 5, "note": "matches title"},
                "consistency": {"score": 4, "note": "ok"},
                "recsys_usefulness": {"score": 5, "note": "useful"},
            },
            "summary": "Strong profile for the titled episode.",
        }
        report = build_report(
            video_filename="Stranger_Things.S01E03.mkv",
            title="Stranger Things S01E03",
            profile_file="Stranger_Things.S01E03.mkv_llm.json",
            llm_payload=payload,
            uses_synopsis=True,
            provider="cursor",
            model="grok-4.7",
        )
        self.assertTrue(report["method"]["uses_synopsis"])
        self.assertFalse(report["method"]["uses_pipeline_evidence"])
        self.assertEqual(report["method"]["provider"], "cursor")
        self.assertEqual(report["title"], "Stranger Things S01E03")
        self.assertTrue(report["usable"])
        expected, _ = weighted_overall(
            {name: report["dimensions"][name]["score"] for name in C.DIMENSION_ORDER}
        )
        self.assertEqual(report["overall"], expected)

    def test_failed_report_not_usable(self) -> None:
        report = failed_report(
            video_filename="demo.mkv",
            title="demo",
            profile_file=None,
            error="missing profile",
        )
        self.assertFalse(report["usable"])
        self.assertEqual(report["error"], "missing profile")
        self.assertEqual(report["overall"], 1.0)

    def test_prompt_requires_nonempty_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonempty reference synopsis"):
            user_prompt(title="Demo", profile={}, synopsis="  ")

    def test_prompt_with_synopsis(self) -> None:
        user = user_prompt(
            title="Suits S01E02",
            profile={"recsys_metadata": {"primary_genre": "Drama"}},
            synopsis="Mike plays air hockey with Wyatt.",
        )
        self.assertIn("sole factual reference", system_prompt())
        self.assertIn("Reference synopsis", user)
        self.assertIn("Suits S01E02", user)
        self.assertIn("primary_genre", user)
        self.assertIn("Mike plays air hockey with Wyatt.", user)

    def test_prompt_loaded_from_markdown(self) -> None:
        from open_enricher.evaluation.prompt import _SYSTEM_PATH, _USER_PATH

        self.assertTrue(_SYSTEM_PATH.is_file())
        self.assertTrue(_USER_PATH.is_file())
        system_text = _SYSTEM_PATH.read_text(encoding="utf-8")
        user_text = _USER_PATH.read_text(encoding="utf-8")
        self.assertIn("{{SCORE_MIN}}", system_text)
        self.assertIn("{{TITLE}}", user_text)
        self.assertIn("{{PROFILE_JSON}}", user_text)
        self.assertIn("{{SYNOPSIS}}", user_text)

    def test_provider_registry_has_cursor(self) -> None:
        self.assertIn("cursor", PROVIDERS)

    def test_get_provider_unknown_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_provider("not-a-provider")

    def test_evaluate_video_with_fake_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            llm_dir = tmp_path / "llms_results"
            llm_dir.mkdir()
            synopsis = "A client hires a lawyer."
            (tmp_path / "demo.mkv_summary.txt").write_text(synopsis, encoding="utf-8")
            profile = {"recsys_metadata": {"primary_genre": "Drama"}}
            profile_path = llm_dir / "demo.mkv_llm.json"
            profile_path.write_text(json.dumps(profile), encoding="utf-8")

            eval_dir = tmp_path / "evaluation"
            eval_dir.mkdir()

            fake = FakeProvider()
            with (
                patch(
                    "open_enricher.evaluation.run.resolve_profile_path",
                    return_value=profile_path,
                ),
                patch(
                    "open_enricher.evaluation.run.resolve_output_paths",
                    return_value=(
                        eval_dir / "demo.mkv_evaluation.json",
                        eval_dir / "demo.mkv_evaluation_raw.txt",
                    ),
                ),
            ):
                result = evaluate_video("demo.mkv", provider=fake, synopsis_dir=tmp_path)

            self.assertEqual(result["status"], "success")
            self.assertTrue(result["usable"])
            self.assertEqual(len(fake.calls), 1)
            report = json.loads(
                (eval_dir / "demo.mkv_evaluation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(report["method"]["uses_synopsis"])
            self.assertIn(synopsis, fake.calls[0][1])

    def test_evaluate_video_requires_synopsis_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            llm_dir = tmp_path / "llms_results"
            llm_dir.mkdir()
            profile_path = llm_dir / "demo.mkv_llm.json"
            profile_path.write_text(json.dumps({"x": 1}), encoding="utf-8")
            eval_dir = tmp_path / "evaluation"
            eval_dir.mkdir()
            synopsis_dir = tmp_path / "synopsis"
            synopsis_dir.mkdir()

            fake = FakeProvider()
            with (
                patch(
                    "open_enricher.evaluation.run.resolve_profile_path",
                    return_value=profile_path,
                ),
                patch(
                    "open_enricher.evaluation.run.resolve_output_paths",
                    return_value=(
                        eval_dir / "score.json",
                        eval_dir / "raw.txt",
                    ),
                ),
            ):
                result = evaluate_video(
                    "demo.mkv",
                    provider=fake,
                    synopsis_dir=synopsis_dir,
                )

            self.assertEqual(result["status"], "failed")
            self.assertIn("Missing synopsis", result["error"])
            self.assertEqual(len(fake.calls), 0)


if __name__ == "__main__":
    unittest.main()
