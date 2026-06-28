"""
Unit tests for Phase 4 "Analytics correctness".

All tests are DB-free Django SimpleTestCases (no Postgres / OpenAI required).
They exercise the pure-math helpers extracted so that the statistically
load-bearing computations can be verified on known inputs, plus the corrected
tool-registration / output-labelling behaviour.

Issues covered (data-insights-fixes branch, data_insights/workflows/tools.py):

P4-1  Correlation matrix must align rows COMPLETE-CASE (row i of feature A and
      feature B come from the same observation), not drop nulls per-column and
      then position-truncate. ``complete_case_matrix`` enforces this.

P4-3  ``cumulative_energy`` must be a true running total across ordered periods
      (prefix sum), not a per-period sum. ``running_cumulative`` computes it.

P4-5  The full advanced ML toolset is registered on the happy path of
      ``get_agent_tools`` (previously only reachable via the except fallback).

P4-6  The correlation tool labels its sampling "random" (order_by("?") is not
      stratified).

P4-7  Stratified train/val/test allocation sums EXACTLY to the class size and
      never over-allocates small classes. ``stratified_split_counts`` uses a
      largest-remainder method.
"""

from django.test import SimpleTestCase

from data_insights.workflows.tools import (
    complete_case_matrix,
    running_cumulative,
    stratified_split_counts,
)


class RunningCumulativeTests(SimpleTestCase):
    """P4-3: cumulative_energy is a real running total (prefix sum)."""

    def test_basic_prefix_sum(self):
        self.assertEqual(running_cumulative([3, 1, 4]), [3.0, 4.0, 8.0])

    def test_is_monotonic_for_nonneg_inputs(self):
        out = running_cumulative([5, 2, 8, 1])
        # rms_energy is non-negative, so a true cumulative must be non-decreasing.
        self.assertTrue(all(out[i] <= out[i + 1] for i in range(len(out) - 1)))

    def test_final_equals_grand_total(self):
        vals = [2.5, 0.0, 7.5, 10.0]
        self.assertEqual(running_cumulative(vals)[-1], sum(vals))

    def test_none_treated_as_zero(self):
        # A period with no energy data must not break the running total.
        self.assertEqual(running_cumulative([2, None, 3]), [2.0, 2.0, 5.0])

    def test_empty(self):
        self.assertEqual(running_cumulative([]), [])

    def test_differs_from_per_period_sum(self):
        # The whole point of the fix: cumulative != per-period for >1 period.
        per_period = [3, 1, 4]
        cumulative = running_cumulative(per_period)
        self.assertNotEqual([float(x) for x in per_period], cumulative)


class CompleteCaseMatrixTests(SimpleTestCase):
    """P4-1: correlation alignment keeps only rows where ALL features non-null."""

    def test_drops_rows_with_any_null(self):
        rows = [
            {"a": 1, "b": 10},
            {"a": None, "b": 20},  # dropped (a null)
            {"a": 3, "b": None},  # dropped (b null)
            {"a": 4, "b": 40},
        ]
        cols, n = complete_case_matrix(rows, ["a", "b"])
        self.assertEqual(n, 2)
        self.assertEqual(cols["a"], [1.0, 4.0])
        self.assertEqual(cols["b"], [10.0, 40.0])

    def test_rows_stay_paired(self):
        # Row pairing must be preserved: the i-th value of every column comes
        # from the same source row. Independent null-drop + truncation would
        # mispair these.
        rows = [
            {"a": 1, "b": 100},
            {"a": 2, "b": None},  # dropped
            {"a": 3, "b": 300},
        ]
        cols, n = complete_case_matrix(rows, ["a", "b"])
        self.assertEqual(n, 2)
        # a=1 pairs with b=100, a=3 pairs with b=300 (NOT b=300 with a=2).
        self.assertEqual(list(zip(cols["a"], cols["b"])), [(1.0, 100.0), (3.0, 300.0)])

    def test_all_columns_same_length(self):
        rows = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": None, "c": 6},
            {"a": 7, "b": 8, "c": 9},
        ]
        cols, n = complete_case_matrix(rows, ["a", "b", "c"])
        lengths = {len(v) for v in cols.values()}
        self.assertEqual(lengths, {n})
        self.assertEqual(n, 2)

    def test_no_complete_rows(self):
        rows = [{"a": 1, "b": None}, {"a": None, "b": 2}]
        cols, n = complete_case_matrix(rows, ["a", "b"])
        self.assertEqual(n, 0)
        self.assertEqual(cols, {"a": [], "b": []})


class StratifiedSplitCountsTests(SimpleTestCase):
    """P4-7: per-class split sums exactly to the class size; no over-allocation."""

    def test_sums_exactly_for_many_sizes(self):
        for c in range(1, 257):
            tr, va, te = stratified_split_counts(c, 0.70, 0.15, 0.15)
            self.assertEqual(tr + va + te, c, f"class size {c} did not sum exactly")
            self.assertTrue(min(tr, va, te) >= 0)
            self.assertTrue(max(tr, va, te) <= c)

    def test_small_class_not_over_allocated(self):
        # The classic bug: c=2 previously produced 1/1/1 == 3 > 2.
        tr, va, te = stratified_split_counts(2, 0.70, 0.15, 0.15)
        self.assertEqual(tr + va + te, 2)
        self.assertLessEqual(tr + va + te, 2)

    def test_singleton_class(self):
        tr, va, te = stratified_split_counts(1, 0.70, 0.15, 0.15)
        self.assertEqual((tr, va, te), (1, 0, 0))

    def test_zero_class(self):
        self.assertEqual(stratified_split_counts(0, 0.70, 0.15, 0.15), (0, 0, 0))

    def test_respects_ratios_on_round_numbers(self):
        tr, va, te = stratified_split_counts(100, 0.70, 0.15, 0.15)
        self.assertEqual((tr, va, te), (70, 15, 15))

    def test_largest_remainder_distribution(self):
        # c=10 with 70/15/15 -> ideal 7.0 / 1.5 / 1.5. Floors 7/1/1 leave 1 unit
        # which goes to the largest remainder (a tie between val and test; first
        # in order wins) -> 7/2/1, summing to 10.
        tr, va, te = stratified_split_counts(10, 0.70, 0.15, 0.15)
        self.assertEqual(tr + va + te, 10)
        self.assertEqual(tr, 7)

    def test_alternate_ratios_sum_exactly(self):
        for c in [3, 4, 5, 11, 13, 17, 99]:
            tr, va, te = stratified_split_counts(c, 0.60, 0.20, 0.20)
            self.assertEqual(tr + va + te, c)


class ClassBalanceSplitRecommendationTests(SimpleTestCase):
    """P4-7 (class-balance variant): MLClassBalanceTool's split_recommendation
    must also sum EXACTLY to each class size and never over-allocate.

    The recommendation was previously built with independent
    ``max(1, round(c * pct))`` rounding, which produced e.g. 1/1/1 == 3 for a
    class of 2. It now delegates to ``stratified_split_counts``. The tool fetches
    class counts from the DB, so this verifies (a) statically that the tool uses
    the helper on the recommendation path, and (b) the exact-sum property the
    helper guarantees for the 70/15/15 ratio the tool hard-codes.
    """

    def test_70_15_15_sums_exactly_for_all_class_sizes(self):
        # MLClassBalanceTool hard-codes 70/15/15; assert the property holds for
        # every plausible per-class count, including the small classes that the
        # old rounding over-allocated.
        for c in range(1, 257):
            tr, va, te = stratified_split_counts(c, 0.70, 0.15, 0.15)
            self.assertEqual(tr + va + te, c, f"class size {c} did not sum exactly")
            self.assertLessEqual(max(tr, va, te), c)

    def test_class_balance_tool_uses_helper_for_recommendation(self):
        import inspect

        from data_insights.workflows.tools import MLClassBalanceTool

        src = inspect.getsource(MLClassBalanceTool._run)
        # The split recommendation must go through the exact-sum helper, not the
        # old independent-rounding (max(1, round(c * 0.70))) that over-allocated.
        self.assertIn("stratified_split_counts(c, 0.70, 0.15, 0.15)", src)
        self.assertNotIn("max(1, round(c * 0.70))", src)


class CorrelationSamplingLabelTests(SimpleTestCase):
    """P4-6: the correlation tool must not advertise/return "stratified" for
    order_by("?") sampling, and its input field must not claim stratification."""

    def test_tool_description_not_stratified(self):
        from data_insights.workflows.tools import MLCorrelationMatrixTool

        desc = MLCorrelationMatrixTool().description.lower()
        self.assertNotIn("stratified", desc)
        self.assertIn("random", desc)

    def test_label_column_field_not_stratified(self):
        from data_insights.workflows.tools import MLCorrelationMatrixInput

        field = MLCorrelationMatrixInput.model_fields["label_column"]
        self.assertNotIn("stratified sampling", (field.description or "").lower())


class AgentToolRegistrationTests(SimpleTestCase):
    """P4-5: the advanced ML tools must be reachable on the normal (happy) path.

    Instantiating the tools requires a live DB (some tools build a SQL agent at
    construction), so this is verified statically against the source of
    ``get_agent_tools`` instead — DB-free, matching the Phase 1/3 convention.
    The check confirms each advanced ML tool is constructed BEFORE the
    ``try:``/``except`` guard (the happy path), not only inside the fallback.
    """

    ADVANCED_ML_TOOLS = (
        "MLClassBalanceTool",
        "MLTrainTestSplitTool",
        "MLCorrelationMatrixTool",
        "MLStatisticalTestTool",
        "MLExportFeaturesTool",
        "MLFeatureImportanceTool",
    )

    def _happy_path_source(self):
        import inspect

        from data_insights.workflows.tools import get_agent_tools

        src = inspect.getsource(get_agent_tools)
        # The happy path is everything up to the first ``try:`` (the guarded
        # WebFetchTool import). The advanced ML tools must appear before it.
        before_try = src.split("try:", 1)[0]
        return before_try

    def test_advanced_ml_tools_on_happy_path(self):
        happy = self._happy_path_source()
        for tool_cls in self.ADVANCED_ML_TOOLS:
            self.assertIn(
                f"{tool_cls}()",
                happy,
                f"{tool_cls} must be registered on the happy path, not only the "
                "except fallback (P4-5)",
            )

    def test_get_agent_tools_returns_advanced_ml_when_db_available(self):
        # If a DB connection is available in this environment, additionally
        # assert the live tool list. Otherwise skip (kept DB-free by default).
        from django.db import connection

        try:
            connection.ensure_connection()
        except Exception:
            self.skipTest("No DB connection available; static check covers P4-5.")

        import data_insights.workflows.tools as tools_mod

        saved = tools_mod._all_tools_cache
        tools_mod._all_tools_cache = None
        try:
            names = {t.name for t in tools_mod.get_agent_tools()}
        finally:
            tools_mod._all_tools_cache = saved

        for expected in (
            "ml_class_balance",
            "ml_train_test_split",
            "ml_correlation_matrix",
            "ml_statistical_test",
            "ml_export_features",
            "ml_feature_importance",
        ):
            self.assertIn(expected, names)


class LazyAgentConstructionTests(SimpleTestCase):
    """Regression (P4-5 follow-up): constructing DataAnalysisTool and the agent
    tool set must NOT open a DB connection.

    DataAnalysisTool's SQL agent is now built lazily on first query instead of
    in a model_validator at construction time. Previously, registering the ML
    tools on the happy path meant DataAnalysisTool was always constructed, and
    its eager TextToSQLAgent opened a DB connection (schema reflection) — so the
    whole tool set (and any agent flow) failed whenever the DB was unreachable.
    These DB-free tests would have ERRORed before the lazy fix.
    """

    def test_data_analysis_tool_constructs_without_db(self):
        from data_insights.workflows.tools import DataAnalysisTool

        tool = DataAnalysisTool()
        # The compiled SQL agent must not be built at construction time.
        self.assertIsNone(tool.agent)

    def test_get_agent_tools_builds_db_free_and_includes_advanced_ml(self):
        import data_insights.workflows.tools as tools_mod

        saved = tools_mod._all_tools_cache
        tools_mod._all_tools_cache = None
        try:
            names = {t.name for t in tools_mod.get_agent_tools()}
        finally:
            tools_mod._all_tools_cache = saved

        for expected in (
            "data_analysis",
            "ml_class_balance",
            "ml_train_test_split",
            "ml_correlation_matrix",
            "ml_statistical_test",
            "ml_export_features",
            "ml_feature_importance",
        ):
            self.assertIn(expected, names)
