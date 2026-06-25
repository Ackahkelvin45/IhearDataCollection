"""
Unit tests for Phase 3 "Sensitive-data minimization (PII & raw storage paths)".

All tests are DB-free Django SimpleTestCases. The schema-masking tests use an
in-memory SQLite engine (no Postgres / OpenAI required) so they exercise the
real SQLDatabaseWrapper code path without any external service.

They cover the surgical, privacy-critical changes made on the
data-insights-fixes branch:

P3-1  Schema-level masking: the ``audio`` column (a raw filesystem path /
      object-storage key, e.g. DO Spaces) is listed in
      ``SENSITIVE_COLUMN_NAMES`` and is stripped by SQLDatabaseWrapper from BOTH
      the CREATE TABLE schema AND the sample-row preview injected into
      ``{table_info}`` — so its raw value never reaches the LLM. Non-sensitive
      columns and their sample values are preserved (org-wide aggregate
      analytics is unaffected).

P3-2  PII reachability: ``_get_default_allowed_tables`` only discovers the
      ``data`` and ``core`` apps, so the ``authentication`` app (the only owner
      of direct contact PII: email / phone_number / first_name / last_name) is
      never exposed to the read-only SQL agent. The allowed data tables that
      carry an ``audio`` storage path are still kept (excluding them would break
      the product's org-wide aggregate analytics).

P3-3  Tool output: the related-field allow-list (``VALID_RELATED_FIELDS``)
      exposes the collector ``username`` (an acceptable display handle) only and
      no longer references ``collector__first_name`` / ``collector__last_name``
      (full legal name = direct PII). The ``top_collectors`` leaderboard is
      preserved by username.

P3-4  Defense-in-depth prompt rule: the SQL_SYSTEM_TEMPLATE instructs the agent
      never to SELECT/expose the ``audio`` column or direct contact details, on
      top of the schema-level masking in P3-1.
"""

import sqlalchemy as sa
from django.test import SimpleTestCase

from data_insights.workflows.sql_agent import (
    SENSITIVE_COLUMN_NAMES,
    SQLDatabaseWrapper,
)


# A raw storage path / object-storage key value of the kind the `audio`
# FileField column holds; must never appear in any LLM-facing output.
_RAW_AUDIO_PATH = "files/2026/secret_object_key_abc123.wav"


def _build_sqlite_db(sample_rows=3):
    """Build an in-memory SQLite SQLDatabaseWrapper with one data table that
    has an `audio` storage-path column plus benign analytics columns."""
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "CREATE TABLE data_noisedataset ("
                "id INTEGER PRIMARY KEY, "
                "name TEXT, "
                "audio TEXT, "
                "recording_device TEXT)"
            )
        )
        conn.execute(
            sa.text(
                "INSERT INTO data_noisedataset "
                "(id, name, audio, recording_device) "
                "VALUES (1, :name, :audio, :device)"
            ),
            {"name": "noise-001", "audio": _RAW_AUDIO_PATH, "device": "iPhone 16"},
        )
    return SQLDatabaseWrapper(
        engine=engine,
        include_tables=["data_noisedataset"],
        sample_rows_in_table_info=sample_rows,
        lazy_table_reflection=True,
    )


class SensitiveColumnConstantTests(SimpleTestCase):
    """P3-1: the storage-path column is registered as sensitive."""

    def test_audio_is_sensitive(self):
        self.assertIn("audio", SENSITIVE_COLUMN_NAMES)

    def test_sensitive_set_is_frozen(self):
        # Module-level allow/deny constants must be immutable so they cannot be
        # mutated at runtime to re-expose a sensitive column.
        self.assertIsInstance(SENSITIVE_COLUMN_NAMES, frozenset)


class SchemaMaskingTests(SimpleTestCase):
    """P3-1: `audio` is stripped from BOTH the schema and the sample rows."""

    def test_audio_column_absent_from_table_info(self):
        info = _build_sqlite_db().get_table_info(["data_noisedataset"])
        self.assertNotIn("audio", info)

    def test_raw_storage_path_not_leaked(self):
        info = _build_sqlite_db().get_table_info(["data_noisedataset"])
        self.assertNotIn(_RAW_AUDIO_PATH, info)
        self.assertNotIn("secret_object_key", info)

    def test_non_sensitive_columns_preserved(self):
        """Masking must be surgical: benign analytics columns remain in the
        CREATE TABLE schema so legitimate queries still work."""
        info = _build_sqlite_db().get_table_info(["data_noisedataset"])
        self.assertIn("recording_device", info)
        self.assertIn("name", info)
        self.assertIn("id", info)

    def test_non_sensitive_sample_values_preserved(self):
        """The sample-row preview must still show benign values (proves we
        masked only the column, not the whole preview)."""
        info = _build_sqlite_db().get_table_info(["data_noisedataset"])
        self.assertIn("iPhone 16", info)
        self.assertIn("noise-001", info)

    def test_sample_rows_helper_excludes_sensitive_column(self):
        """Directly exercise the sample-row builder: the raw path must never be
        in the preview even though a row exists in the table."""
        db = _build_sqlite_db()
        # Force reflection so the table object is available.
        db.get_table_info(["data_noisedataset"])
        table = db._metadata.tables["data_noisedataset"]
        sample = db._get_sample_rows(table)
        self.assertIn("iPhone 16", sample)  # a row was returned
        self.assertNotIn(_RAW_AUDIO_PATH, sample)
        self.assertNotIn("secret_object_key", sample)


class AllowedTablesPiiReachabilityTests(SimpleTestCase):
    """P3-2: the auth app (the only direct-PII source) is never SQL-reachable."""

    def _allowed(self):
        from data_insights.workflows.tools import _get_default_allowed_tables

        return _get_default_allowed_tables()

    def test_no_authentication_tables_exposed(self):
        leaked = [t for t in self._allowed() if t.startswith("authentication_")]
        self.assertEqual(leaked, [], f"auth/PII tables exposed to SQL agent: {leaked}")

    def test_customuser_table_not_exposed(self):
        self.assertNotIn("authentication_customuser", self._allowed())

    def test_userotp_table_not_exposed(self):
        self.assertNotIn("authentication_userotp", self._allowed())

    def test_data_tables_with_audio_path_still_allowed(self):
        """Org-wide aggregate analytics must be preserved: the data tables are
        kept even though they carry the (now schema-masked) `audio` column."""
        allowed = self._allowed()
        self.assertIn("data_noisedataset", allowed)

    def test_only_data_and_core_apps_discovered(self):
        """Every discovered table must belong to the `data` or `core` app."""
        for table in self._allowed():
            self.assertTrue(
                table.startswith("data_") or table.startswith("core_"),
                f"unexpected non-data/core table discovered: {table}",
            )


class ValidRelatedFieldsTests(SimpleTestCase):
    """P3-3: tool field allow-list exposes username only, never legal name."""

    def _fields(self):
        from data_insights.workflows.tools import VALID_RELATED_FIELDS

        return VALID_RELATED_FIELDS

    def test_username_allowed(self):
        self.assertIn("collector__username", self._fields())

    def test_first_name_not_allowed(self):
        self.assertNotIn("collector__first_name", self._fields())

    def test_last_name_not_allowed(self):
        self.assertNotIn("collector__last_name", self._fields())

    def test_no_direct_pii_fields_in_allowlist(self):
        """No allowed reference may name a direct-PII attribute.

        Matching is on whole ``__``-separated segments so legitimate names that
        merely *contain* a marker substring (e.g. ``microphone_type__name``
        contains "phone") are not flagged.
        """
        pii_markers = {"first_name", "last_name", "email", "phone", "phone_number"}
        for field in self._fields():
            segments = set(field.split("__"))
            leaked = segments & pii_markers
            self.assertEqual(
                leaked,
                set(),
                f"PII field reference leaked into allow-list: {field}",
            )


class PromptHardeningTests(SimpleTestCase):
    """P3-4: defense-in-depth prompt rule on top of schema masking."""

    def _template(self):
        from data_insights.workflows.prompt import SQL_SYSTEM_TEMPLATE

        return SQL_SYSTEM_TEMPLATE

    def test_prompt_forbids_audio_column(self):
        t = self._template().lower()
        self.assertIn("audio", t)
        self.assertIn("sensitive", t)

    def test_prompt_suggests_presence_check(self):
        # Steers the agent to `audio IS NOT NULL` instead of selecting the path.
        self.assertIn("IS NOT NULL", self._template())

    def test_prompt_forbids_contact_details(self):
        t = self._template().lower()
        self.assertIn("email", t)
        self.assertIn("phone", t)
