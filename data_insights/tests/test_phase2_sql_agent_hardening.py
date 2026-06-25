"""
Unit tests for Phase 2 "NL->SQL agent hardening (read-only execution,
SELECT-only, DoS guard)".

All tests are DB-free Django SimpleTestCases (no Postgres / OpenAI required).
They cover the surgical, security-critical changes made on the
data-insights-fixes branch in ``data_insights/workflows/sql_agent.py`` and the
related ``datacollection/settings.py`` wiring:

P2-1  Read-only engine: ``create_readonly_engine`` builds a SQLAlchemy engine
      whose every pooled connection runs ``SET default_transaction_read_only =
      on`` and ``SET statement_timeout = <ms>`` (writes/DDL rejected by Postgres
      and runaway queries killed server-side). Dedicated read-only credentials
      are used when configured, otherwise the app creds are used with a loud
      warning. The Django ORM / LangGraph checkpointer connections are NOT
      touched (separate connections; not exercised here).

P2-2  SELECT-only + DoS guard in ``_validate_sql_query``: only a single
      top-level SELECT / WITH (CTE) statement is allowed; comments are stripped
      before validation; multi-statement input, writes/DDL, COPY, file access
      and time-based (pg_sleep / sleep / benchmark / waitfor) payloads are
      rejected. Legitimate analytics (plain SELECT and CTE) must still pass and
      get a bounded LIMIT.
"""

import os

from django.conf import settings
from django.test import SimpleTestCase

from data_insights.workflows.sql_agent import (
    TextToSQLAgent,
    create_readonly_engine,
    _resolve_readonly_db_credentials,
    _get_statement_timeout_seconds,
    DEFAULT_SQL_STATEMENT_TIMEOUT_SECONDS,
)


class _FakeCursor:
    """Records SQL passed to ``execute`` so we can assert on it."""

    def __init__(self, sink):
        self._sink = sink

    def execute(self, sql):
        self._sink.append(sql)

    def close(self):
        pass


class _FakeDBAPIConnection:
    def __init__(self, sink):
        self._sink = sink

    def cursor(self):
        return _FakeCursor(self._sink)


def _invoke_readonly_listener(engine, dbapi_connection):
    """Run only OUR connect listener (``_set_readonly_session``).

    The pool also carries dialect-level connect listeners that require a real
    psycopg2 connection; we select ours by name so the test stays DB-free.
    """
    ours = [
        fn
        for fn in engine.pool.dispatch.connect
        if getattr(fn, "__name__", "") == "_set_readonly_session"
    ]
    assert len(ours) == 1, f"expected exactly one read-only listener, got {len(ours)}"
    ours[0](dbapi_connection, object())


class ReadOnlyEngineTests(SimpleTestCase):
    """P2-1: the SQL-agent engine is forced read-only at the session level."""

    def test_engine_builds_without_connecting(self):
        """Engine construction must not require a live DB (lazy connect)."""
        engine = create_readonly_engine("u", "p", "nonexistent-host", 5432, "db")
        self.assertEqual(engine.url.username, "u")
        self.assertEqual(engine.url.host, "nonexistent-host")
        self.assertEqual(engine.url.database, "db")

    def test_connect_listener_sets_readonly_and_timeout(self):
        """On every pooled connection: read-only + statement_timeout are SET."""
        engine = create_readonly_engine("u", "p", "h", 5432, "db")
        sink = []
        _invoke_readonly_listener(engine, _FakeDBAPIConnection(sink))

        joined = " | ".join(sink)
        self.assertIn("SET default_transaction_read_only = on", joined)
        self.assertRegex(joined, r"SET statement_timeout = \d+")

    def test_statement_timeout_is_in_milliseconds(self):
        """statement_timeout must be the configured seconds expressed in ms."""
        engine = create_readonly_engine("u", "p", "h", 5432, "db")
        sink = []
        _invoke_readonly_listener(engine, _FakeDBAPIConnection(sink))

        timeout_stmt = next(s for s in sink if "statement_timeout" in s)
        ms = int(timeout_stmt.rsplit("=", 1)[1].strip())
        expected = _get_statement_timeout_seconds() * 1000
        self.assertEqual(ms, expected)
        self.assertGreater(ms, 0)


class ReadOnlyCredentialResolutionTests(SimpleTestCase):
    """P2-1: dedicated read-only creds preferred; safe fallback otherwise."""

    def test_falls_back_to_app_credentials_when_no_ro_creds(self):
        # Ensure no env RO creds leak in from the surrounding shell.
        with self.settings():
            for k in (
                "AI_INSIGHT_DB_READONLY_USER",
                "AI_INSIGHT_DB_READONLY_PASSWORD",
            ):
                os.environ.pop(k, None)
            user, pw = _resolve_readonly_db_credentials("appuser", "apppw")
        self.assertEqual((user, pw), ("appuser", "apppw"))

    def test_prefers_dedicated_readonly_env_credentials(self):
        os.environ["AI_INSIGHT_DB_READONLY_USER"] = "ro_user"
        os.environ["AI_INSIGHT_DB_READONLY_PASSWORD"] = "ro_pw"
        try:
            user, pw = _resolve_readonly_db_credentials("appuser", "apppw")
        finally:
            os.environ.pop("AI_INSIGHT_DB_READONLY_USER", None)
            os.environ.pop("AI_INSIGHT_DB_READONLY_PASSWORD", None)
        self.assertEqual((user, pw), ("ro_user", "ro_pw"))


class StatementTimeoutSettingsTests(SimpleTestCase):
    """P2-1: statement timeout is configurable and sane."""

    def test_default_timeout_is_positive(self):
        self.assertGreater(_get_statement_timeout_seconds(), 0)

    def test_default_constant_is_sane(self):
        self.assertGreaterEqual(DEFAULT_SQL_STATEMENT_TIMEOUT_SECONDS, 1)

    def test_settings_expose_statement_timeout(self):
        db_cfg = settings.AI_INSIGHT["DATABASE"]
        self.assertIn("STATEMENT_TIMEOUT_SECONDS", db_cfg)
        self.assertGreater(int(db_cfg["STATEMENT_TIMEOUT_SECONDS"]), 0)

    def test_settings_expose_readonly_credential_keys(self):
        db_cfg = settings.AI_INSIGHT["DATABASE"]
        self.assertIn("READONLY_USER", db_cfg)
        self.assertIn("READONLY_PASSWORD", db_cfg)


class SqlValidationAllowsAnalyticsTests(SimpleTestCase):
    """P2-2: legitimate SELECT / CTE analytics must keep working."""

    def _validate(self, query, table_names=None):
        return TextToSQLAgent._validate_sql_query(query, table_names)

    def test_plain_select_passes_and_gets_limit(self):
        out = self._validate("SELECT id FROM responses")
        self.assertTrue(out.upper().startswith("SELECT"))
        self.assertIn("LIMIT", out.upper())

    def test_cte_with_query_passes(self):
        out = self._validate("WITH t AS (SELECT id FROM responses) SELECT * FROM t")
        self.assertTrue(out.upper().startswith("WITH"))
        self.assertIn("LIMIT", out.upper())

    def test_block_comment_is_stripped_not_rejected(self):
        """A legitimate SELECT with an inline comment must still pass."""
        out = self._validate("SELECT /* note */ id FROM responses")
        self.assertTrue(out.upper().startswith("SELECT"))

    def test_existing_limit_under_cap_is_preserved(self):
        out = self._validate("SELECT id FROM responses LIMIT 10")
        self.assertRegex(out, r"LIMIT\s+10")


class SqlValidationRejectsWritesAndDosTests(SimpleTestCase):
    """P2-2: writes/DDL, multi-statements, COPY and DoS payloads are rejected."""

    def _assert_rejected(self, query, table_names=None):
        with self.assertRaises(ValueError):
            TextToSQLAgent._validate_sql_query(query, table_names)

    def test_reject_drop(self):
        self._assert_rejected("DROP TABLE responses")

    def test_reject_delete(self):
        self._assert_rejected("DELETE FROM responses")

    def test_reject_update(self):
        self._assert_rejected("UPDATE responses SET x = 1")

    def test_reject_insert(self):
        self._assert_rejected("INSERT INTO responses VALUES (1)")

    def test_reject_create_table_as_select(self):
        self._assert_rejected("CREATE TABLE x AS SELECT 1")

    def test_reject_multiple_statements(self):
        self._assert_rejected("SELECT * FROM responses; DROP TABLE responses")

    def test_reject_comment_hidden_write(self):
        # A write hidden after a line comment must not slip through after
        # comment stripping (the trailing DROP becomes visible/flagged).
        self._assert_rejected("SELECT 1 -- harmless\nDROP TABLE responses")

    def test_reject_pg_sleep_dos(self):
        self._assert_rejected("SELECT pg_sleep(10)")

    def test_reject_copy(self):
        self._assert_rejected("COPY responses TO STDOUT")

    def test_reject_empty_query(self):
        self._assert_rejected("")

    def test_reject_unauthorized_table(self):
        # When a table allow-list is supplied, references outside it are denied.
        self._assert_rejected("SELECT * FROM secret_table", table_names=["responses"])
