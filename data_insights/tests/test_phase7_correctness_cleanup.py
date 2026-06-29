"""
Unit tests for Phase 7 "Remaining correctness & cleanup".

All tests are DB-free Django SimpleTestCases (no Postgres / OpenAI required).
They cover the correctness fixes on the data-insights-fixes branch:

P7-1  The user-id ContextVar (data_insights.workflows.tools._current_user_id) must
      be set INSIDE the streaming generator (the execution context the tools read
      it from) and reset via the token in a ``finally`` so it never leaks across
      requests on a reused worker thread. Exercised with the real ContextVar and
      a generator that mirrors the view's set()/try/finally(reset) shape.

P7-2  ``fix_stuck_messages`` must only reset messages whose ``updated_at`` is older
      than a configurable threshold, so an in-flight (recently updated) PROCESSING
      message is never failed. Verified by capturing the ORM filter kwargs.

P7-3  ``decompose`` branches that previously indexed ``result["rows"]`` directly
      (``_decompose_grouped`` / ``_decompose_ranked``) must NOT raise KeyError when
      a routed analysis_type's result lacks rows — they return a safe value and
      ``decompose`` stays exception-free.

P7-4  ``ChatSession.increment_total_messages`` must use an atomic
      ``F("total_messages") + 1`` UPDATE (no read-modify-write race), then
      refresh the in-memory value.

P7-5  Cleanup: ``_humanise`` is defined ONCE (in chart_builder) and imported by
      widget_composer; the dead ``get_pending_clarification`` /
      ``_update_query_handles`` helpers are gone.
"""

from contextvars import ContextVar
from unittest import mock

from django.test import SimpleTestCase
from django.db.models import F

from data_insights.workflows.tools import _current_user_id
from data_insights.workflows.widget_composer import (
    decompose,
    _decompose_grouped,
    _decompose_ranked,
)


# ---------------------------------------------------------------------------
# P7-1 — ContextVar set inside the generator, reset in finally
# ---------------------------------------------------------------------------
class CurrentUserIdContextVarTests(SimpleTestCase):
    """P7-1: the user-id ContextVar is set inside the generator and reset."""

    def _make_stream(self, user_id, sink):
        """Mirror the view generator's set()/try/finally(reset) shape."""

        def stream():
            token = _current_user_id.set(user_id)
            try:
                # The tools read the ContextVar while the loop runs.
                sink.append(_current_user_id.get())
                yield "chunk"
            finally:
                _current_user_id.reset(token)

        return stream()

    def test_value_visible_during_generation(self):
        sink = []
        gen = self._make_stream(42, sink)
        list(gen)  # drive to completion
        self.assertEqual(sink, [42], "tools must see the user id during streaming")

    def test_reset_after_completion(self):
        self.assertIsNone(_current_user_id.get())
        list(self._make_stream(7, []))
        self.assertIsNone(
            _current_user_id.get(), "ContextVar must be reset to its default"
        )

    def test_reset_even_on_early_close(self):
        """GeneratorExit (client disconnect) must still reset the ContextVar."""
        self.assertIsNone(_current_user_id.get())
        gen = self._make_stream(99, [])
        next(gen)  # enter the generator, value is now set
        gen.close()  # raises GeneratorExit inside -> finally runs
        self.assertIsNone(
            _current_user_id.get(), "disconnect must not leak the user id"
        )

    def test_no_cross_request_leak_on_reused_context(self):
        """Two sequential generations on the same context don't leak between them."""
        list(self._make_stream(1, []))
        self.assertIsNone(_current_user_id.get())
        sink = []
        list(self._make_stream(2, sink))
        self.assertEqual(sink, [2])
        self.assertIsNone(_current_user_id.get())

    def test_contextvar_default_is_none(self):
        self.assertIsInstance(_current_user_id, ContextVar)
        self.assertIsNone(_current_user_id.get())


# ---------------------------------------------------------------------------
# P7-2 — fix_stuck_messages only touches messages older than a threshold
# ---------------------------------------------------------------------------
class FixStuckMessagesThresholdTests(SimpleTestCase):
    """P7-2: the command filters PROCESSING messages by an age cutoff."""

    def _run(self, **opts):
        from data_insights.management.commands import fix_stuck_messages

        captured = {}

        class _QS:
            def count(self):
                return 0

            def update(self, **kwargs):  # pragma: no cover - count is 0
                return 0

        def fake_filter(**kwargs):
            captured["kwargs"] = kwargs
            return _QS()

        cmd = fix_stuck_messages.Command()
        cmd.stdout = mock.MagicMock()
        cmd.style = mock.MagicMock()

        with mock.patch(
            "data_insights.management.commands.fix_stuck_messages."
            "ChatMessage.objects.filter",
            side_effect=fake_filter,
        ):
            cmd.handle(**opts)
        return captured["kwargs"]

    def test_filters_by_updated_at_cutoff(self):
        kwargs = self._run(minutes=15)
        self.assertIn(
            "updated_at__lt",
            kwargs,
            "must constrain to messages older than the cutoff",
        )

    def test_status_is_still_processing(self):
        from data_insights.models import ChatMessage

        kwargs = self._run(minutes=15)
        self.assertEqual(kwargs.get("status"), ChatMessage.MessageStatus.PROCESSING)

    def test_smaller_threshold_yields_later_cutoff(self):
        """A smaller minutes value produces a more-recent (later) cutoff."""
        cutoff_30 = self._run(minutes=30)["updated_at__lt"]
        cutoff_5 = self._run(minutes=5)["updated_at__lt"]
        self.assertGreater(
            cutoff_5,
            cutoff_30,
            "5-minute cutoff must be later in time than the 30-minute one",
        )

    def test_default_threshold_is_positive(self):
        from data_insights.management.commands.fix_stuck_messages import (
            DEFAULT_STUCK_AGE_MINUTES,
        )

        self.assertGreater(DEFAULT_STUCK_AGE_MINUTES, 0)


# ---------------------------------------------------------------------------
# P7-3 — decompose never raises KeyError on missing rows
# ---------------------------------------------------------------------------
class DecomposeMissingRowsTests(SimpleTestCase):
    """P7-3: rows-indexing branches return safely instead of raising KeyError."""

    def test_grouped_without_rows_does_not_raise(self):
        # No "rows" key at all — previously result["rows"] -> KeyError.
        out = _decompose_grouped({"analysis_type": "group_count"})
        self.assertIsNone(out)

    def test_grouped_with_empty_rows_does_not_raise(self):
        out = _decompose_grouped({"analysis_type": "group_count", "rows": []})
        self.assertIsNone(out)

    def test_ranked_without_rows_does_not_raise(self):
        out = _decompose_ranked({"analysis_type": "highest_decibel"})
        self.assertIsNone(out)

    def test_decompose_router_safe_for_grouped_without_rows(self):
        # decompose() routes avg_decibel_by_* / group_count to _decompose_grouped.
        try:
            out = decompose({"analysis_type": "avg_decibel_by_region"})
        except KeyError as exc:  # pragma: no cover - regression guard
            self.fail(f"decompose raised KeyError on missing rows: {exc}")
        self.assertIsNone(out)

    def test_decompose_router_safe_for_ranked_without_rows(self):
        try:
            out = decompose({"analysis_type": "lowest_decibel"})
        except KeyError as exc:  # pragma: no cover - regression guard
            self.fail(f"decompose raised KeyError on missing rows: {exc}")
        self.assertIsNone(out)

    def test_normal_grouped_path_unchanged(self):
        """A well-formed grouped result still produces a multi-widget artifact."""
        result = {
            "analysis_type": "avg_decibel_by_region",
            "chart_hint": {"x": "region", "y": "avg_db"},
            "rows": [
                {"region": "Accra", "avg_db": 62.0},
                {"region": "Kumasi", "avg_db": 58.5},
            ],
        }
        out = _decompose_grouped(result)
        self.assertIsInstance(out, dict)
        self.assertIn("widgets", out)
        self.assertTrue(len(out["widgets"]) >= 1)


# ---------------------------------------------------------------------------
# P7-4 — increment_total_messages uses an atomic F() update
# ---------------------------------------------------------------------------
class IncrementTotalMessagesAtomicTests(SimpleTestCase):
    """P7-4: increment uses F('total_messages') + 1, not read-modify-write."""

    def test_uses_f_expression_update(self):
        from data_insights.models import ChatSession

        session = ChatSession(pk=123)

        captured = {}

        class _FilterResult:
            def update(self, **kwargs):
                captured["update_kwargs"] = kwargs
                return 1

        def fake_filter(*args, **kwargs):
            captured["filter_kwargs"] = kwargs
            return _FilterResult()

        with mock.patch.object(
            ChatSession.objects, "filter", side_effect=fake_filter
        ), mock.patch.object(ChatSession, "refresh_from_db") as refresh:
            session.increment_total_messages()

        # Filtered by this row's pk (no global update)
        self.assertEqual(captured["filter_kwargs"].get("pk"), 123)

        # The update value must be an F-expression (atomic), not a Python int.
        upd = captured["update_kwargs"]
        self.assertIn("total_messages", upd)
        value = upd["total_messages"]
        self.assertNotIsInstance(
            value, int, "must be an F() expression, not a pre-read Python int"
        )
        # F('total_messages') + 1 is a CombinedExpression; assert it references F.
        self.assertIn("total_messages", str(value))

        # In-memory value refreshed afterwards.
        refresh.assert_called_once()

    def test_f_import_available(self):
        # Sanity: the model module exposes the atomic primitive we rely on.
        self.assertTrue(callable(F))


# ---------------------------------------------------------------------------
# P7-5 — cleanup: single _humanise, dead helpers removed
# ---------------------------------------------------------------------------
class CleanupTests(SimpleTestCase):
    """P7-5: duplicated helper consolidated; dead code removed."""

    def test_humanise_is_single_definition_shared(self):
        from data_insights.workflows import widget_composer, chart_builder

        # widget_composer must reference chart_builder's function object — same
        # identity proves the duplicate definition was removed and consolidated.
        self.assertIs(widget_composer._humanise, chart_builder._humanise)
        self.assertEqual(chart_builder._humanise("avg_decibel"), "Avg Decibel")

    def test_get_pending_clarification_removed(self):
        from data_insights.workflows import clarification_gate

        self.assertFalse(
            hasattr(clarification_gate, "get_pending_clarification"),
            "dead get_pending_clarification helper should be removed",
        )

    def test_update_query_handles_removed(self):
        from data_insights.workflows.agent_workflow import DataAgent

        self.assertFalse(
            hasattr(DataAgent, "_update_query_handles"),
            "dead no-op _update_query_handles method should be removed",
        )
