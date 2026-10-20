"""Dependency-free tests for the deterministic metadata judger.

Requires only pydantic (already a project dependency). No Drive, no GPU, no torch.
Run from the repository root:

    PYTHONPATH=src python -m unittest tests.test_judger_score -v
"""

from __future__ import annotations

import copy
import unittest

from open_enricher.stages.judger_core.score import score_profile


def _grounded_profile() -> dict:
    """Filled profile whose tokens overlap the evidence below."""
    return {
        "recsys_metadata": {
            "primary_genre": "Drama",
            "sub_genres": ["Mystery", "Thriller"],
            "pacing": "Moderate",
            "mood": ["tense", "eerie"],
        },
        "search_and_discovery": {
            "semantic_tags": ["teenagers", "forest", "police", "hospital"],
            "main_topics": ["teenagers", "mysterious events"],
        },
        "audience_profiling": {
            "target_age_group": "Young Adults",
            "viewer_interests": ["supernatural mystery", "ensemble drama"],
        },
        "content_summary": {
            "visual_style": "Dim warm lighting with dark forest greens and muted blues",
            "narrative_hook": "Teenagers navigate eerie forest events while police investigate.",
        },
    }


def _generic_profile() -> dict:
    """Schema-valid but thin: placeholders, empty lists, and generic fillers."""
    return {
        "recsys_metadata": {
            "primary_genre": "n/a",
            "sub_genres": [],
            "pacing": "Moderate",
            "mood": [],
        },
        "search_and_discovery": {
            "semantic_tags": ["video", "scene", "content"],
            "main_topics": ["various things"],
        },
        "audience_profiling": {
            "target_age_group": "unknown",
            "viewer_interests": ["stuff"],
        },
        "content_summary": {
            "visual_style": "nice video",
            "narrative_hook": "something happens in the video",
        },
    }


def _evidence() -> dict:
    return {
        "audio": [
            {
                "start_timestamp": 0.0,
                "end_timestamp": 2.5,
                "speaker": "SPEAKER_00",
                "text": "The teenagers went into the dark forest near the hospital.",
            }
        ],
        "color": [
            {
                "scene_id": 1,
                "timestamp": 12.0,
                "frame_file": "scene_1.jpg",
                "color_palette": ["#284028", "#3c3c3c", "#5064c8"],
            }
        ],
        "vlm": {
            "short_summary": "Teenagers and police officers encounter eerie forest and hospital scenes.",
            "tags": ["teenagers", "police", "forest"],
            "mood": ["tense", "eerie"],
            "setting": "dark forest and hospital corridor",
            "genres": ["drama", "mystery"],
        },
    }


class TestJudgerScore(unittest.TestCase):
    def test_grounded_beats_generic_on_key_dimensions(self) -> None:
        evidence = _evidence()
        grounded = score_profile(_grounded_profile(), evidence, video="demo.mp4")
        generic = score_profile(_generic_profile(), evidence, video="demo.mp4")

        self.assertEqual(grounded["dimensions"]["accessibility"]["score"], 1.0)
        self.assertEqual(generic["dimensions"]["accessibility"]["score"], 1.0)

        self.assertGreater(
            grounded["dimensions"]["completeness"]["score"],
            generic["dimensions"]["completeness"]["score"],
        )
        self.assertGreater(
            grounded["dimensions"]["specificity"]["score"],
            generic["dimensions"]["specificity"]["score"],
        )
        self.assertGreater(
            grounded["dimensions"]["grounding"]["score"],
            generic["dimensions"]["grounding"]["score"],
        )
        self.assertGreater(grounded["overall"], generic["overall"])

    def test_scoring_is_deterministic(self) -> None:
        profile = _grounded_profile()
        evidence = _evidence()
        first = score_profile(profile, evidence, video="demo.mp4", profile_file="demo_llm.json")
        second = score_profile(
            copy.deepcopy(profile),
            copy.deepcopy(evidence),
            video="demo.mp4",
            profile_file="demo_llm.json",
        )
        self.assertEqual(first, second)

    def test_invalid_profile_zeroes_overall(self) -> None:
        report = score_profile({"broken": True}, {}, video="bad.mp4")
        self.assertEqual(report["dimensions"]["accessibility"]["score"], 0.0)
        self.assertEqual(report["overall"], 0.0)
        self.assertFalse(report["usable"])
        self.assertIn("validation_error", report)

    def test_usable_flag_on_grounded_profile(self) -> None:
        report = score_profile(_grounded_profile(), _evidence(), video="demo.mp4")
        self.assertTrue(report["method"]["deterministic"])
        # Grounded fixture should clear the hypothesized usability gate.
        self.assertTrue(report["usable"])
        self.assertGreaterEqual(
            report["dimensions"]["completeness"]["score"],
            report["thresholds"]["completeness"],
        )


if __name__ == "__main__":
    unittest.main()
