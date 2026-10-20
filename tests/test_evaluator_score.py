"""GPU-free tests for LLM evaluator helpers (title, weights, report assembly).

Does not load torch or call a model. Run from the repository root:

    PYTHONPATH=src python -m pytest tests/test_evaluator_score.py -q
"""

from __future__ import annotations

import unittest

from open_enricher.stages.evaluator_core import criteria as C
from open_enricher.stages.evaluator_core.prompt import system_prompt, user_prompt
from open_enricher.stages.evaluator_core.score import (
    build_report,
    failed_report,
    title_from_filename,
    weighted_overall,
)


class TestEvaluatorHelpers(unittest.TestCase):
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

    def test_build_report_applies_weights(self) -> None:
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
        )
        self.assertFalse(report["method"]["uses_synopsis"])
        self.assertFalse(report["method"]["uses_pipeline_evidence"])
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

    def test_prompt_excludes_synopsis_and_includes_title(self) -> None:
        system = system_prompt()
        self.assertIn("empty conversation", system.lower())
        self.assertIn("Do NOT invent or request a synopsis", system)
        user = user_prompt(
            title="Stranger Things S01E03",
            profile={"recsys_metadata": {"primary_genre": "Drama"}},
        )
        self.assertIn("Stranger Things S01E03", user)
        self.assertNotIn("synopsis", user.lower())
        self.assertNotIn("transcript", user.lower())

    def test_prompt_loaded_from_markdown(self) -> None:
        from open_enricher.stages.evaluator_core.prompt import _SYSTEM_PATH, _USER_PATH

        self.assertTrue(_SYSTEM_PATH.is_file())
        self.assertTrue(_USER_PATH.is_file())
        system_text = _SYSTEM_PATH.read_text(encoding="utf-8")
        user_text = _USER_PATH.read_text(encoding="utf-8")
        self.assertIn("{{SCORE_MIN}}", system_text)
        self.assertIn("{{TITLE}}", user_text)
        self.assertIn("{{PROFILE_JSON}}", user_text)


if __name__ == "__main__":
    unittest.main()
