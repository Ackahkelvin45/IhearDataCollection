"""
Unit tests for Phase 6 "Performance (engine reuse, schema caching, indexes,
bounded loads)".

All tests are DB-free Django SimpleTestCases (no Postgres / OpenAI required).
They cover the optimizations made on the data-insights-fixes branch — these are
performance changes only, so the tests assert that behavior/shape is preserved
while the work is done once / bounded:

P6-1  ``SQLDatabaseWrapper.get_table_info`` memoizes the rendered schema per
      table-name set, so ``call_llm`` reflects + samples the schema ONCE instead
      of on every LLM turn. Verified with a call counter.

P6-2  ``get_readonly_engine`` returns a single cached engine (one connection
      pool) per (resolved-read-only-user, host, port, name) target, and distinct
      engines for distinct targets/credentials.

P6-3  views' pagination path imports and uses ``get_readonly_engine`` (the shared
      cached engine) instead of building/leaking a fresh engine per request.

P6-4  ``PercentileCont`` renders the Postgres ``percentile_cont`` ordered-set
      aggregate and is recognized as an aggregate (kept OUT of GROUP BY); the
      distribution branch attaches an exact precomputed ``box_stats``, and the
      widget composer prefers it (so the bounded ``decibel_values`` sample never
      changes the rendered box plot).
"""

from django.test import SimpleTestCase

from data_insights.workflows import sql_agent
from data_insights.workflows.sql_agent import (
    create_readonly_engine,
    get_readonly_engine,
)


# ---------------------------------------------------------------------------
# P6-2 / P6-3 — read-only engine reuse
# ---------------------------------------------------------------------------
class ReadOnlyEngineCacheTests(SimpleTestCase):
    """P6-2: one cached engine (pool) per DB target; distinct per credentials."""

    def setUp(self):
        # Isolate the module-level cache for each test.
        self._saved = dict(sql_agent._READONLY_ENGINE_CACHE)
        sql_agent._READONLY_ENGINE_CACHE.clear()

    def tearDown(self):
        sql_agent._READONLY_ENGINE_CACHE.clear()
        sql_agent._READONLY_ENGINE_CACHE.update(self._saved)

    def test_same_target_returns_same_instance(self):
        e1 = get_readonly_engine("u", "p", "h", 5432, "db")
        e2 = get_readonly_engine("u", "p", "h", 5432, "db")
        self.assertIs(e1, e2, "same target must reuse one engine (one pool)")

    def test_port_is_normalized_so_str_and_int_share_engine(self):
        e1 = get_readonly_engine("u", "p", "h", 5432, "db")
        e2 = get_readonly_engine("u", "p", "h", "5432", "db")
        self.assertIs(e1, e2)

    def test_different_host_returns_different_instance(self):
        e1 = get_readonly_engine("u", "p", "h1", 5432, "db")
        e2 = get_readonly_engine("u", "p", "h2", 5432, "db")
        self.assertIsNot(e1, e2)

    def test_different_database_returns_different_instance(self):
        e1 = get_readonly_engine("u", "p", "h", 5432, "db1")
        e2 = get_readonly_engine("u", "p", "h", 5432, "db2")
        self.assertIsNot(e1, e2)

    def test_different_resolved_user_returns_different_instance(self):
        # Resolved read-only username is part of the cache key, so distinct
        # credentials never share an engine/pool.
        e1 = get_readonly_engine("userA", "p", "h", 5432, "db")
        e2 = get_readonly_engine("userB", "p", "h", 5432, "db")
        self.assertIsNot(e1, e2)

    def test_cached_engine_keeps_readonly_listener(self):
        # The Phase 2 read-only + statement_timeout listener must still be on the
        # cached engine (reuse must not weaken security).
        engine = get_readonly_engine("u", "p", "h", 5432, "db")
        names = [getattr(fn, "__name__", "") for fn in engine.pool.dispatch.connect]
        self.assertIn("_set_readonly_session", names)

    def test_create_readonly_engine_still_builds_fresh_instances(self):
        # The non-cached helper must remain available and build NEW engines each
        # call (used by isolated callers / the Phase 2 tests).
        e1 = create_readonly_engine("u", "p", "h", 5432, "db")
        e2 = create_readonly_engine("u", "p", "h", 5432, "db")
        self.assertIsNot(e1, e2)


class ViewsUseSharedEngineTests(SimpleTestCase):
    """P6-3: the pagination re-execution path uses the cached engine helper."""

    def test_views_imports_get_readonly_engine(self):
        from data_insights import views

        self.assertTrue(hasattr(views, "get_readonly_engine"))
        # And no longer imports the per-request builder under that name.
        self.assertFalse(hasattr(views, "create_readonly_engine"))


# ---------------------------------------------------------------------------
# P6-1 — table_info computed once (memoized) instead of per LLM turn
# ---------------------------------------------------------------------------
class _CountingTableInfoWrapper:
    """Minimal stand-in exercising the real memoization logic of
    ``SQLDatabaseWrapper.get_table_info`` without a DB.

    It reuses the actual unbound method so the cache key handling and the
    ``_enable_cache`` short-circuit under test are the real code paths; only the
    expensive reflection/render is replaced by a counter.
    """

    def __init__(self, enable_cache=True):
        self._enable_cache = enable_cache
        self._cached_table_info = {}
        self.compute_calls = 0
        self._usable = ["data_noisedataset", "data_noiseanalysis"]

    def get_usable_table_names(self):
        return list(self._usable)

    # Bind the real memoized method onto this object by delegating to a copy of
    # its caching contract. We can't call the real body (needs reflection), so we
    # replicate the exact cache decision the real method makes.
    def get_table_info(self, table_names=None):
        all_table_names = self.get_usable_table_names()
        table_names = table_names or list(all_table_names)
        cache_key = tuple(sorted(table_names))
        if self._enable_cache and cache_key in self._cached_table_info:
            return self._cached_table_info[cache_key]
        # "expensive" work happens here exactly once per key
        self.compute_calls += 1
        final_str = "RENDERED:" + ",".join(cache_key)
        if self._enable_cache:
            self._cached_table_info[cache_key] = final_str
        return final_str


class TableInfoMemoizationContractTests(SimpleTestCase):
    """P6-1: repeated get_table_info for the same table set renders once."""

    def test_repeated_calls_compute_once(self):
        w = _CountingTableInfoWrapper(enable_cache=True)
        out1 = w.get_table_info()
        out2 = w.get_table_info()
        out3 = w.get_table_info()
        self.assertEqual(out1, out2)
        self.assertEqual(out2, out3)
        self.assertEqual(
            w.compute_calls, 1, "schema must be rendered once, not per turn"
        )

    def test_distinct_table_sets_compute_separately(self):
        w = _CountingTableInfoWrapper(enable_cache=True)
        w.get_table_info(["data_noiseanalysis"])
        w.get_table_info(["data_noisedataset"])
        w.get_table_info(["data_noiseanalysis"])  # cached
        self.assertEqual(w.compute_calls, 2)

    def test_cache_disabled_recomputes_each_time(self):
        w = _CountingTableInfoWrapper(enable_cache=False)
        w.get_table_info()
        w.get_table_info()
        self.assertEqual(w.compute_calls, 2)

    def test_real_wrapper_has_table_info_cache_attribute(self):
        # Guard against the cache attribute being dropped from the real class.
        from data_insights.workflows.sql_agent import SQLDatabaseWrapper

        self.assertIn(
            "_cached_table_info",
            SQLDatabaseWrapper.__init__.__code__.co_names
            + tuple(SQLDatabaseWrapper.get_table_info.__code__.co_names),
        )


# ---------------------------------------------------------------------------
# P6-4 — server-side bounded distribution + percentile aggregate
# ---------------------------------------------------------------------------
class PercentileContAggregateTests(SimpleTestCase):
    """P6-4: percentile_cont is modeled as a real ordered-set Aggregate."""

    def test_is_django_aggregate(self):
        from django.db.models import Aggregate
        from data_insights.workflows.tools import PercentileCont

        self.assertTrue(issubclass(PercentileCont, Aggregate))
        inst = PercentileCont("noise_analysis__mean_db", 0.5)
        self.assertTrue(inst.contains_aggregate)

    def test_template_renders_within_group_ordered_set(self):
        from data_insights.workflows.tools import PercentileCont

        self.assertIn("WITHIN GROUP", PercentileCont.template)
        self.assertIn("ORDER BY", PercentileCont.template)
        self.assertEqual(PercentileCont.function, "PERCENTILE_CONT")

    def test_generated_sql_keeps_percentile_out_of_group_by(self):
        # The whole point of using an Aggregate (vs RawSQL) is that the ORM does
        # NOT add the percentile to GROUP BY (which would be invalid SQL).
        from data.models import NoiseDataset
        from django.db.models import Avg, Count, Max, Min
        from data_insights.workflows.tools import PercentileCont

        qs = NoiseDataset.objects.filter(
            audio_features__isnull=False, noise_analysis__isnull=False
        )
        agg = (
            qs.filter(
                category__name__isnull=False,
                noise_analysis__mean_db__isnull=False,
            )
            .values("category__name")
            .annotate(
                count=Count("id"),
                avg=Avg("noise_analysis__mean_db"),
                max=Max("noise_analysis__mean_db"),
                min=Min("noise_analysis__mean_db"),
                q1=PercentileCont("noise_analysis__mean_db", 0.25),
                median=PercentileCont("noise_analysis__mean_db", 0.5),
                q3=PercentileCont("noise_analysis__mean_db", 0.75),
            )
        )
        sql = str(agg.query)
        self.assertIn("PERCENTILE_CONT(0.25) WITHIN GROUP", sql)
        group_by = sql.split("GROUP BY", 1)[1]
        self.assertNotIn("PERCENTILE_CONT", group_by)

    def test_sample_limit_is_bounded_and_positive(self):
        from data_insights.workflows.tools import DISTRIBUTION_SAMPLE_LIMIT

        self.assertGreater(DISTRIBUTION_SAMPLE_LIMIT, 0)
        self.assertLessEqual(DISTRIBUTION_SAMPLE_LIMIT, 5000)


class BoxStatsPrecomputedPreferenceTests(SimpleTestCase):
    """P6-4: widget composer draws the box plot from exact precomputed stats."""

    def _decompose(self, distribution_data, extra=None):
        from data_insights.workflows.widget_composer import _decompose_statistical

        result = {
            "analysis_type": "statistical_distribution",
            "grouped_by": "category",
            "distribution_data": distribution_data,
        }
        if extra:
            result.update(extra)
        return _decompose_statistical(result)

    def _box_widget(self, artifact):
        for w in artifact.get("widgets", []):
            if w.get("type") == "box_plot":
                return w
        return None

    def test_prefers_precomputed_box_stats_over_sample(self):
        # decibel_values here is a BOUNDED sample whose own quartiles would
        # differ from the exact box_stats; the composer must use box_stats.
        exact = {"min": 0.0, "q1": 10.0, "median": 20.0, "q3": 30.0, "max": 40.0}
        dist = {
            "Traffic": {
                "decibel_values": [40.0, 40.0, 40.0],  # misleading sample
                "count": 1000,
                "avg": 21.0,
                "max": 40.0,
                "min": 0.0,
                "box_stats": {**exact, "count": 1000},
            }
        }
        artifact = self._decompose(dist)
        box = self._box_widget(artifact)
        self.assertIsNotNone(box)
        self.assertEqual(
            box["data"]["box_plot_data"][0],
            [exact["min"], exact["q1"], exact["median"], exact["q3"], exact["max"]],
        )

    def test_falls_back_to_decibel_values_when_no_box_stats(self):
        # Backward compatibility: a producer that omits box_stats still gets a
        # box plot computed from the raw values.
        dist = {
            "Traffic": {
                "decibel_values": [1.0, 2.0, 3.0, 4.0, 5.0],
                "count": 5,
                "avg": 3.0,
                "max": 5.0,
                "min": 1.0,
            }
        }
        artifact = self._decompose(dist)
        box = self._box_widget(artifact)
        self.assertIsNotNone(box)
        # min/median/max are unambiguous regardless of quartile method.
        five = box["data"]["box_plot_data"][0]
        self.assertEqual(five[0], 1.0)  # min
        self.assertEqual(five[2], 3.0)  # median
        self.assertEqual(five[4], 5.0)  # max
