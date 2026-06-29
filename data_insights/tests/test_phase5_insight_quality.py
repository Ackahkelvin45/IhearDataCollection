"""
Unit tests for Phase 5 "UX & insight quality" (chart/visualization fixes).

All tests are DB-free Django SimpleTestCases (no Postgres / OpenAI required).
They exercise the pure-Python chart classification, unit labelling, truncation
surfacing, numeric coercion and box-plot statistics so the visualization layer
can be trusted to represent the data honestly.

Issues covered (data-insights-fixes branch):

P5-2  Fractional 0..1 metrics that are NOT parts of a whole (RMS energy,
      entropy, correlation, mutual information) must NOT be drawn as pie/donut.
      Pie/donut is only chosen for non-negative components that sum to a
      meaningful total (a composition).

P5-3  Known units (e.g. dB) must be re-attached to chart axis labels and
      stat-card values/labels so a bare number like 62 is unambiguous.

P5-4  Top-N truncation must be visible: the chart/table config surfaces a
      caption / truncated flag and the original total count.

P5-5  Box-plot widgets emit real box statistics (min/q1/median/q3/max) instead
      of silently degrading to a mislabeled bar chart.

P5-6  Numeric coercion drops non-numeric / null y-values (returns None) instead
      of fabricating misleading zero-height bars, and the drop is surfaced.
"""

from django.test import SimpleTestCase

from data_insights.workflows.chart_builder import (
    select_chart_type,
    build_chart_config,
    unit_for_column,
    label_with_unit,
    _is_composition_data,
    _coerce_numeric,
    ChartDecision,
    TruncateSpec,
)
from data_insights.workflows.widget_composer import (
    _box_stats,
    _quartile,
    _build_table_widget,
)


# ── P5-2: chart-type classification (composition vs magnitude) ────────────────


class CompositionClassificationTests(SimpleTestCase):
    """P5-2: only true part-of-whole compositions become pie/donut."""

    def _rows(self, col, values):
        return [{"label": f"g{i}", col: v} for i, v in enumerate(values)]

    def test_rms_energy_in_0_1_is_not_composition(self):
        # RMS energy lives in [0,1] but is a magnitude, not a slice of a whole.
        rows = self._rows("avg_rms", [0.12, 0.08, 0.30, 0.21])
        self.assertFalse(_is_composition_data(rows, "avg_rms"))

    def test_entropy_in_0_1_is_not_composition(self):
        rows = self._rows("normalized_entropy", [0.4, 0.9, 0.1])
        self.assertFalse(_is_composition_data(rows, "normalized_entropy"))

    def test_correlation_in_0_1_is_not_composition(self):
        rows = self._rows("spearman_correlation", [0.5, 0.2, 0.9])
        self.assertFalse(_is_composition_data(rows, "spearman_correlation"))

    def test_mutual_information_score_is_not_composition(self):
        rows = self._rows("mi_score", [0.6, 0.3, 0.1])
        self.assertFalse(_is_composition_data(rows, "mi_score"))

    def test_fractions_summing_to_one_with_share_name_is_composition(self):
        rows = self._rows("share", [0.5, 0.3, 0.2])
        self.assertTrue(_is_composition_data(rows, "share"))

    def test_percentages_summing_to_100_is_composition(self):
        rows = self._rows("percentage", [50, 30, 20])
        self.assertTrue(_is_composition_data(rows, "percentage"))

    def test_share_values_not_summing_to_whole_is_not_composition(self):
        # Has a composition keyword but the values do not form a whole.
        rows = self._rows("share", [0.5, 0.5, 0.5])  # sums to 1.5
        self.assertFalse(_is_composition_data(rows, "share"))

    def test_negative_components_rejected(self):
        rows = self._rows("proportion", [0.6, -0.1, 0.5])
        self.assertFalse(_is_composition_data(rows, "proportion"))

    def test_plain_metric_without_composition_keyword_is_not_composition(self):
        rows = self._rows("avg_db", [0.4, 0.5, 0.1])
        self.assertFalse(_is_composition_data(rows, "avg_db"))

    def test_rms_energy_renders_as_bar_not_pie(self):
        rows = self._rows("avg_rms", [0.12, 0.08, 0.30])
        decision = select_chart_type(
            rows, ["label", "avg_rms"], {"x": "label", "y": "avg_rms"}
        )
        self.assertIsNotNone(decision)
        self.assertNotIn(decision.chart_type, ("pie_chart", "donut_chart"))
        self.assertEqual(decision.chart_type, "bar_chart")

    def test_real_composition_renders_as_pie(self):
        rows = [
            {"label": "A", "share": 0.5},
            {"label": "B", "share": 0.3},
            {"label": "C", "share": 0.2},
        ]
        decision = select_chart_type(
            rows, ["label", "share"], {"x": "label", "y": "share"}
        )
        self.assertIsNotNone(decision)
        self.assertEqual(decision.chart_type, "pie_chart")


# ── P5-3: unit labels ─────────────────────────────────────────────────────────


class UnitLabelTests(SimpleTestCase):
    """P5-3: known units are re-attached; unknown columns get no unit."""

    def test_db_columns_resolve_to_dB(self):
        for col in (
            "mean_db",
            "avg_db",
            "max_db",
            "min_db",
            "median_db",
            "decibel_level",
        ):
            self.assertEqual(unit_for_column(col), "dB", col)

    def test_frequency_columns_resolve_to_Hz(self):
        self.assertEqual(unit_for_column("avg_centroid"), "Hz")
        self.assertEqual(unit_for_column("spectral_bandwidth"), "Hz")

    def test_duration_resolves_to_seconds(self):
        self.assertEqual(unit_for_column("avg_duration"), "s")

    def test_unknown_column_has_no_unit(self):
        self.assertIsNone(unit_for_column("count"))
        self.assertIsNone(unit_for_column("dataset_count"))
        self.assertIsNone(unit_for_column(None))

    def test_label_with_unit_appends_once(self):
        self.assertEqual(label_with_unit("Avg Db", "dB"), "Avg Db (dB)")
        # Idempotent — does not double-append.
        self.assertEqual(label_with_unit("Avg Db (dB)", "dB"), "Avg Db (dB)")

    def test_label_with_unit_noop_when_unit_unknown(self):
        self.assertEqual(label_with_unit("Count", None), "Count")

    def test_chart_config_attaches_unit_to_axes(self):
        rows = [{"region": "North", "avg_db": 62}, {"region": "South", "avg_db": 55}]
        cfg = build_chart_config("bar_chart", rows, {"x": "region", "y": "avg_db"})
        fd = cfg["frontend_data"]
        self.assertEqual(fd["y_unit"], "dB")
        self.assertIn("(dB)", fd["y_label"])
        self.assertIn("(dB)", fd["title"])

    def test_chart_config_no_unit_for_count(self):
        rows = [{"region": "North", "count": 10}, {"region": "South", "count": 7}]
        cfg = build_chart_config("bar_chart", rows, {"x": "region", "y": "count"})
        fd = cfg["frontend_data"]
        self.assertIsNone(fd["y_unit"])
        self.assertNotIn("(", fd["y_label"])


# ── P5-4: truncation indicator ────────────────────────────────────────────────


class TruncationIndicatorTests(SimpleTestCase):
    """P5-4: truncation is visible (caption + flags + original total)."""

    def test_chart_decision_truncation_reflected_in_title_and_flags(self):
        rows = [{"label": f"g{i}", "count": 100 - i} for i in range(20)]
        decision = ChartDecision(
            chart_type="horizontal_bar_chart",
            truncate=TruncateSpec(limit=12, sort_by="count", sort_order="desc"),
        )
        cfg = build_chart_config(
            "horizontal_bar_chart", rows, {"x": "label", "y": "count"}, decision
        )
        fd = cfg["frontend_data"]
        self.assertTrue(fd["truncated"])
        self.assertEqual(fd["total_count"], 20)
        self.assertIn("of 20", fd["title"])
        self.assertIn("caption", fd)
        self.assertIn("12", fd["caption"])

    def test_hard_cap_surfaces_truncation_even_without_decision(self):
        rows = [{"label": f"g{i}", "count": i} for i in range(30)]
        cfg = build_chart_config("bar_chart", rows, {"x": "label", "y": "count"})
        fd = cfg["frontend_data"]
        self.assertTrue(fd["truncated"])
        self.assertEqual(fd["total_count"], 30)
        self.assertEqual(fd["shown_count"], 12)
        self.assertIn("of 30", fd["title"])
        self.assertIn("Showing top 12 of 30", fd["caption"])

    def test_no_truncation_no_caption(self):
        rows = [{"label": f"g{i}", "count": i} for i in range(4)]
        cfg = build_chart_config("bar_chart", rows, {"x": "label", "y": "count"})
        fd = cfg["frontend_data"]
        self.assertFalse(fd["truncated"])
        self.assertEqual(fd["total_count"], 4)
        self.assertNotIn("caption", fd)

    def test_table_widget_surfaces_truncation(self):
        shown = [{"a": i} for i in range(20)]
        widget = _build_table_widget("t1", "Preview", shown, 0, total_count=137)
        table = widget["data"]["table"]
        self.assertTrue(table["truncated"])
        self.assertEqual(table["total_count"], 137)
        self.assertEqual(table["shown_count"], 20)
        self.assertEqual(table["caption"], "Showing 20 of 137 rows")

    def test_table_widget_no_caption_when_not_truncated(self):
        shown = [{"a": i} for i in range(5)]
        widget = _build_table_widget("t1", "Full", shown, 0, total_count=5)
        table = widget["data"]["table"]
        self.assertFalse(table["truncated"])
        self.assertNotIn("caption", table)


# ── P5-5: box-plot statistics ─────────────────────────────────────────────────


class BoxStatsTests(SimpleTestCase):
    """P5-5: real box statistics instead of a silent bar-chart fallback."""

    def test_quartile_matches_linear_interpolation(self):
        s = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(_quartile(s, 0.25), 1.75)
        self.assertAlmostEqual(_quartile(s, 0.5), 2.5)
        self.assertAlmostEqual(_quartile(s, 0.75), 3.25)

    def test_box_stats_full_five_number_summary(self):
        stats = _box_stats([5, 1, 3, 2, 4])
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 5.0)
        self.assertEqual(stats["median"], 3.0)
        self.assertEqual(stats["count"], 5)
        self.assertLess(stats["q1"], stats["median"])
        self.assertGreater(stats["q3"], stats["median"])

    def test_box_stats_none_for_empty(self):
        self.assertIsNone(_box_stats([]))
        self.assertIsNone(_box_stats([None, "x"]))

    def test_box_stats_ignores_non_numeric(self):
        stats = _box_stats([1, None, 2, "bad", 3])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 3.0)


# ── P5-6: numeric coercion ────────────────────────────────────────────────────


class NumericCoercionTests(SimpleTestCase):
    """P5-6: non-numeric/null y-values are dropped, not coerced to 0."""

    def test_numeric_inputs_pass_through(self):
        self.assertEqual(_coerce_numeric(5), 5.0)
        self.assertEqual(_coerce_numeric(5.5), 5.5)
        self.assertEqual(_coerce_numeric(0), 0.0)
        self.assertEqual(_coerce_numeric(-3), -3.0)

    def test_numeric_strings_are_parsed(self):
        self.assertEqual(_coerce_numeric("42"), 42.0)
        self.assertEqual(_coerce_numeric("1,234.5"), 1234.5)
        self.assertEqual(_coerce_numeric(" 7 "), 7.0)

    def test_null_returns_none_not_zero(self):
        self.assertIsNone(_coerce_numeric(None))

    def test_non_numeric_returns_none_not_zero(self):
        self.assertIsNone(_coerce_numeric("n/a"))
        self.assertIsNone(_coerce_numeric("abc"))
        self.assertIsNone(_coerce_numeric(""))

    def test_booleans_rejected(self):
        self.assertIsNone(_coerce_numeric(True))
        self.assertIsNone(_coerce_numeric(False))

    def test_build_chart_config_drops_non_numeric_rows(self):
        rows = [
            {"label": "A", "avg_db": 60},
            {"label": "B", "avg_db": None},  # null aggregate — must NOT become 0
            {"label": "C", "avg_db": "n/a"},  # unparseable — must NOT become 0
            {"label": "D", "avg_db": 55},
        ]
        cfg = build_chart_config("bar_chart", rows, {"x": "label", "y": "avg_db"})
        fd = cfg["frontend_data"]
        # Only the two real numeric rows are plotted; no fabricated 0-bars.
        self.assertEqual(fd["data"], [60.0, 55.0])
        self.assertEqual(fd["labels"], ["A", "D"])
        self.assertEqual(fd["dropped_rows"], 2)
        self.assertNotIn(0.0, fd["data"])
        # The drop is surfaced to the user.
        self.assertIn("caption", fd)
        self.assertIn("omitted", fd["caption"])

    def test_genuine_zero_is_preserved(self):
        rows = [{"label": "A", "avg_db": 0}, {"label": "B", "avg_db": 5}]
        cfg = build_chart_config("bar_chart", rows, {"x": "label", "y": "avg_db"})
        fd = cfg["frontend_data"]
        self.assertEqual(fd["data"], [0.0, 5.0])
        self.assertEqual(fd["dropped_rows"], 0)
