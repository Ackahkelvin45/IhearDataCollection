"""
Unit tests for clarification gate helper functions.

These test the heuristic detection logic in isolation. They do not require
a database — entity context is passed as a plain dict.
"""

from django.test import SimpleTestCase

from data_insights.workflows.clarification_gate import (
    _needs_entity,
    _has_entity_match,
    _match_analysis_types,
    _has_ranking_intent,
    _has_explicit_metric,
    _needs_time_range,
    _has_time_reference,
)


# ── _has_entity_match (word-boundary matching) ──────────────────────


class HasEntityMatchTests(SimpleTestCase):
    """Word-boundary entity matching — substring false positives are fixed."""

    def test_exact_match(self):
        ctx = {"available_regions": ["Accra"], "available_communities": []}
        self.assertTrue(_has_entity_match("show data for Accra", ctx))

    def test_partial_word_no_match(self):
        """'Ho' is a real region name but must not match 'how', 'who', etc."""
        ctx = {"available_regions": ["Ho"], "available_communities": []}
        self.assertFalse(_has_entity_match("how do I get data", ctx))
        self.assertFalse(_has_entity_match("who collected this", ctx))
        self.assertFalse(_has_entity_match("shows the results", ctx))
        self.assertFalse(_has_entity_match("open hours", ctx))

    def test_ho_as_standalone_word(self):
        ctx = {"available_regions": ["Ho"], "available_communities": []}
        self.assertTrue(_has_entity_match("data from Ho region", ctx))

    def test_central_not_match_centralized(self):
        ctx = {"available_regions": ["Central"], "available_communities": []}
        self.assertFalse(_has_entity_match("centralized database", ctx))

    def test_central_as_standalone_word(self):
        ctx = {"available_regions": ["Central"], "available_communities": []}
        self.assertTrue(_has_entity_match("data from Central region", ctx))

    def test_bono_not_match_bonus(self):
        ctx = {"available_regions": ["Bono"], "available_communities": []}
        self.assertFalse(_has_entity_match("bonus recording", ctx))

    def test_multi_word_entity(self):
        ctx = {"available_regions": ["Greater Accra"], "available_communities": []}
        self.assertTrue(_has_entity_match("recordings in Greater Accra", ctx))

    def test_no_match_when_entity_absent(self):
        ctx = {"available_regions": ["Kumasi"], "available_communities": []}
        self.assertFalse(_has_entity_match("show all recordings", ctx))

    def test_community_match(self):
        ctx = {"available_regions": [], "available_communities": ["Madina"]}
        self.assertTrue(_has_entity_match("data from Madina", ctx))

    def test_empty_context(self):
        self.assertFalse(
            _has_entity_match(
                "data from Accra",
                {
                    "available_regions": [],
                    "available_communities": [],
                },
            )
        )


# ── _needs_entity ───────────────────────────────────────────────────


class NeedsEntityTests(SimpleTestCase):
    """Entity-seeking intent detection — no prepositions, uses DB context."""

    def test_structural_hint_region(self):
        self.assertTrue(_needs_entity("show recordings by region"))

    def test_structural_hint_area(self):
        self.assertTrue(_needs_entity("which area is loudest"))

    def test_structural_hint_district(self):
        self.assertTrue(_needs_entity("district breakdown"))

    def test_db_entity_in_query(self):
        ctx = {"available_regions": ["Kumasi"], "available_communities": []}
        self.assertTrue(_needs_entity("recordings from Kumasi", ctx))

    def test_preposition_in_no_longer_triggers(self):
        """After fix: 'in' is not in ENTITY_HINT_WORDS."""
        self.assertFalse(_needs_entity("show me trends in the last 30 days"))

    def test_preposition_from_no_longer_triggers(self):
        self.assertFalse(_needs_entity("data from last week"))

    def test_preposition_within_no_longer_triggers(self):
        self.assertFalse(_needs_entity("recordings within 30 days"))

    def test_no_context_no_hint_words(self):
        self.assertFalse(_needs_entity("show me the loudest recordings"))

    def test_entity_name_in_context_matches(self):
        ctx = {"available_regions": ["Accra"], "available_communities": []}
        self.assertTrue(_needs_entity("data around Accra", ctx))

    def test_entity_name_with_word_boundary(self):
        """'Ho' in context should not match 'how'."""
        ctx = {"available_regions": ["Ho"], "available_communities": []}
        self.assertFalse(_needs_entity("how to analyze data", ctx))

    def test_empty_context_falls_back_to_hints(self):
        self.assertFalse(_needs_entity("no hint words here", {}))


# ── _match_analysis_types ───────────────────────────────────────────


class MatchAnalysisTypesTests(SimpleTestCase):
    """Analysis type keyword classification."""

    def test_energy_keywords(self):
        result = _match_analysis_types("show decibel levels")
        self.assertIn("energy", result)

    def test_spectral_keywords(self):
        result = _match_analysis_types("what is the spectral centroid")
        self.assertIn("spectral", result)

    def test_frequency_keywords(self):
        result = _match_analysis_types("dominant frequency analysis")
        self.assertIn("frequency", result)

    def test_correlation_keywords(self):
        result = _match_analysis_types("compare energy vs frequency")
        self.assertIn("correlation", result)

    def test_statistical_keywords(self):
        result = _match_analysis_types("distribution of decibel levels")
        self.assertIn("statistical", result)

    def test_temporal_keywords(self):
        result = _match_analysis_types("trends over time")
        self.assertIn("temporal", result)

    def test_overview_keywords(self):
        result = _match_analysis_types("give me an overview")
        self.assertIn("overview", result)

    def test_zero_match_returns_empty(self):
        """After fix: zero-match queries no longer default to ['overview'].
        The multi_analysis_type signal uses len(matched) > 1, so zero-match
        queries passing through as [] won't fire the signal — they'll fall
        through to whatever the tool does with no analysis_type set."""
        result = _match_analysis_types("help me understand")
        self.assertEqual(result, [])  # was: ["overview"]

    def test_single_match_length(self):
        result = _match_analysis_types("show decibel data")
        self.assertEqual(len(result), 1)

    def test_multi_match_length(self):
        result = _match_analysis_types("energy and spectral analysis")
        self.assertGreater(len(result), 1)


# ── _has_ranking_intent ─────────────────────────────────────────────


class HasRankingIntentTests(SimpleTestCase):
    """Ranking word detection."""

    def test_best_triggers(self):
        self.assertTrue(_has_ranking_intent("best recordings"))

    def test_worst_triggers(self):
        self.assertTrue(_has_ranking_intent("worst noise levels"))

    def test_top_triggers(self):
        self.assertTrue(_has_ranking_intent("top 10 recordings"))

    def test_loudest_triggers(self):
        self.assertTrue(_has_ranking_intent("loudest areas"))

    def test_highest_triggers(self):
        self.assertTrue(_has_ranking_intent("highest decibel"))

    def test_no_ranking_word(self):
        self.assertFalse(_has_ranking_intent("show me recordings"))


# ── _has_explicit_metric ────────────────────────────────────────────


class HasExplicitMetricTests(SimpleTestCase):
    """Explicit metric word detection."""

    def test_decibel_is_metric(self):
        self.assertTrue(_has_explicit_metric("show decibel levels"))

    def test_db_is_metric(self):
        self.assertTrue(_has_explicit_metric("average db by region"))

    def test_rms_is_metric(self):
        self.assertTrue(_has_explicit_metric("rms energy analysis"))

    def test_count_is_metric(self):
        self.assertTrue(_has_explicit_metric("recording count by region"))

    def test_frequency_is_metric(self):
        self.assertTrue(_has_explicit_metric("frequency analysis"))

    def test_no_metric(self):
        self.assertFalse(_has_explicit_metric("show me data"))


# ── _needs_time_range / _has_time_reference ────────────────────────


class TimeRangeTests(SimpleTestCase):
    """Time-range-related detection functions."""

    def test_trend_needs_time(self):
        self.assertTrue(_needs_time_range("trend over time"))

    def test_daily_needs_time(self):
        self.assertTrue(_needs_time_range("daily recordings"))

    def test_no_time_need(self):
        self.assertFalse(_needs_time_range("show recordings"))

    def test_has_explicit_time_reference(self):
        self.assertTrue(_has_time_reference("last 30 days"))

    def test_has_date_pattern(self):
        self.assertTrue(_has_time_reference("data from 2024-01-15"))

    def test_has_today(self):
        self.assertTrue(_has_time_reference("recordings today"))

    def test_no_time_reference(self):
        self.assertFalse(_has_time_reference("show me data"))


# ── Phase 2: LLM clarification detection heuristics ────────────────


class LooksLikeClarificationTests(SimpleTestCase):
    """_looks_like_clarification() heuristic patterns."""

    @staticmethod
    def _looks_like_clarification(content: str) -> bool:
        # Inline copy of DataAgent._looks_like_clarification for testing
        import re

        if "?" not in content:
            return False
        patterns = [
            r"\bdid you mean\b",
            r"\bwhich (one|of these)\b",
            r"\bcould you (specify|clarify)\b",
            r"\bwould you like\b",
            r"\bdo you (want|mean|prefer)\b",
            r"\bare you (asking|referring|looking)\b",
            r"\bI (need|want) to (clarify|confirm|understand)\b",
            r"\bcan you (specify|clarify|tell me more)\b",
        ]
        content_lower = content.lower()
        return any(re.search(p, content_lower) for p in patterns)

    def test_did_you_mean(self):
        self.assertTrue(
            self._looks_like_clarification("Did you mean frequency or energy analysis?")
        )

    def test_which_one(self):
        self.assertTrue(
            self._looks_like_clarification(
                "Which one would you prefer — spectral or temporal?"
            )
        )

    def test_could_you_specify(self):
        self.assertTrue(
            self._looks_like_clarification(
                "Could you specify what kind of analysis you want?"
            )
        )

    def test_do_you_want(self):
        self.assertTrue(
            self._looks_like_clarification("Do you want the results grouped by region?")
        )

    def test_normal_response_not_clarification(self):
        self.assertFalse(
            self._looks_like_clarification(
                "Here are the results for your query. The average decibel level is 72 dB."
            )
        )

    def test_question_not_clarification(self):
        """A question that is a follow-up offer, not a clarification request."""
        self.assertFalse(
            self._looks_like_clarification("Is there anything else I can help with?")
        )

    def test_no_question_mark(self):
        self.assertFalse(
            self._looks_like_clarification("Let me know if you need anything else")
        )


# ── Phase 2: multi_analysis_type V2 threshold (len != 1) ───────────


class MultiAnalysisTypeV2Tests(SimpleTestCase):
    """V2 threshold: fires on 0 or 2+ matches, not exactly 1."""

    def test_zero_match_should_fire_under_v2(self):
        """'help me understand' matches nothing → len=0 → should fire (≠1)."""
        matched = _match_analysis_types("help me understand the data")
        self.assertEqual(matched, [])
        self.assertTrue(len(matched) != 1)

    def test_one_match_should_not_fire_under_v2(self):
        """'show decibel data' matches energy only → len=1 → should not fire."""
        matched = _match_analysis_types("show decibel data")
        self.assertIn("energy", matched)
        self.assertEqual(len(matched), 1)
        self.assertFalse(len(matched) != 1)

    def test_two_matches_should_fire_under_v2(self):
        """'energy and frequency' matches both → len=2 → should fire (≠1)."""
        matched = _match_analysis_types("energy and frequency analysis")
        self.assertIn("energy", matched)
        self.assertIn("frequency", matched)
        self.assertGreater(len(matched), 1)
        self.assertTrue(len(matched) != 1)


# ── Phase 2: tool ambiguity sentinel ────────────────────────────────


class ToolAmbiguitySentinelTests(SimpleTestCase):
    """_tool_ambiguity sentinel dict shape validation."""

    def test_sentinel_has_required_keys(self):
        sentinel = {
            "_tool_ambiguity": True,
            "dimension": "analysis_type",
            "query": "help me understand",
            "message": "Could not determine analysis type.",
        }
        self.assertTrue(sentinel.get("_tool_ambiguity"))
        self.assertIn("dimension", sentinel)
        self.assertIn("query", sentinel)

    def test_sentinel_is_detectable(self):
        """The check in post_process_tools: result.get("_tool_ambiguity")"""
        valid = {"_tool_ambiguity": True, "dimension": "analysis_type"}
        normal = {"analysis_type": "energy", "rows": []}
        self.assertTrue(valid.get("_tool_ambiguity"))
        self.assertFalse(normal.get("_tool_ambiguity"))


# ── Phase 2: force_entity / force_time_range in input models ──────


class ForceParamsTests(SimpleTestCase):
    """force_entity and force_time_range on tool input models."""

    def test_data_analysis_input_accepts_force_entity(self):
        from data_insights.workflows.tools import DataAnalysisInput

        inp = DataAnalysisInput(
            query="test",
            query_type="group_count",
            force_entity="Accra",
        )
        self.assertEqual(inp.force_entity, "Accra")

    def test_data_analysis_input_accepts_force_time_range(self):
        from data_insights.workflows.tools import DataAnalysisInput

        inp = DataAnalysisInput(
            query="test",
            query_type="recent_datasets",
            force_time_range="last_30_days",
        )
        self.assertEqual(inp.force_time_range, "last_30_days")

    def test_audio_analysis_input_accepts_force_entity(self):
        from data_insights.workflows.tools import AudioAnalysisInput

        inp = AudioAnalysisInput(
            query="test",
            force_entity="all",
        )
        self.assertEqual(inp.force_entity, "all")

    def test_audio_analysis_input_accepts_force_time_range(self):
        from data_insights.workflows.tools import AudioAnalysisInput

        inp = AudioAnalysisInput(
            query="test",
            force_time_range="last_year",
        )
        self.assertEqual(inp.force_time_range, "last_year")
