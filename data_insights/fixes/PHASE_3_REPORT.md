# Phase 3 — Sensitive-data minimization (PII & raw storage paths)

# Phase 3 — Sensitive-data minimization (PII & raw storage paths)

**Branch:** `data-insights-fixes` · **App under review:** `data_insights/` · **Status:** Acceptance met (validation), tests pass, 0 regressions.

## 1. Goal recap
Stop **direct PII** (email, phone, physical address, precise GPS) and **raw storage locations** (filesystem paths, object-storage keys, DO Spaces URLs/signed URLs) from being returned to the chat / LLM / tool output — **without** breaking the intentional org-wide (cross-collector) aggregate analytics or adding per-user row scoping.

### Investigation findings (pre-edit)
- **Only direct-PII source is the `authentication` app**: `CustomUser.email` / `phone_number` / `first_name` / `last_name` (`authentication_customuser`) and `authentication_userotp`. The `data` and `core` models contain **no** email/phone/address/GPS columns (verified: no `latitude`/`longitude`/`location`/`address` fields exist; `recording_device` is a free-text device-model CharField, not a precise identifier).
- **Only raw storage path is the `audio` FileField** (`upload_to="files/"`) on `data_noisedataset`, `data_recording`, and `data_cleanspeechdataset` — it holds the raw storage path / object-storage key.
- Dataset FKs (`collector_id` / `contributor_id`) are opaque integers, not PII.
- The SQL agent's table discovery (`_get_default_allowed_tables`) already scoped to `data`/`core` only, so the auth/PII tables were already SQL-unreachable — but the `audio` path was still flowing through the schema/sample-rows, and tool output still exposed full legal names + raw audio paths.

## 2. Issues fixed (final status)

| ID | Issue | Fix | Status |
|----|-------|-----|--------|
| P3-1 | Raw `audio` storage path/object key reached the LLM via the `{table_info}` schema (CREATE TABLE) and sample-row preview | Added `SENSITIVE_COLUMN_NAMES = frozenset({"audio"})` in `sql_agent.py`; `SQLDatabaseWrapper.get_table_info()` now strips sensitive columns from the CREATE TABLE; `_get_sample_rows()` now selects explicit columns (not `*`) so the path is never pulled into the preview | **Fixed** |
| P3-2 | Confirm direct-PII (`authentication`) tables are not SQL-reachable; keep audio-bearing data tables for analytics | Documented + test-locked: discovery stays `data`/`core`-only; audio tables retained (excluding them would break aggregate analytics) since the `audio` column is masked at the schema layer | **Verified / locked** |
| P3-3 | Tool output leaked full legal names: `top_collectors` joined `first_name`+`last_name`; `VALID_RELATED_FIELDS` allow-listed `collector__first_name` / `collector__last_name` | Removed both PII fields from `VALID_RELATED_FIELDS`; `_top_collectors_monthly` now selects/returns `collector_id` + `username` only (leaderboard preserved by username, an acceptable display handle) | **Fixed** |
| P3-4 | `NoiseDetailTool` returned the raw `audio` path | Replaced with `"has_audio": bool(getattr(dataset, "audio", None))` — a presence signal only | **Fixed** |
| P3-5 | No defense-in-depth instruction to the agent | Added a `SQL_SYSTEM_TEMPLATE` rule: never SELECT/expose `audio` (use `audio IS NOT NULL` presence check), never return direct contact details; group/display collectors by id/username only | **Fixed** |
| Critique must_fix (1) | PII-marker test could false-flag benign names containing a marker substring (e.g. `microphone_type__name` contains "phone") | Test `test_no_direct_pii_fields_in_allowlist` matches on whole `__`-separated segments, not substrings | **Resolved** |

## 3. Files modified
- `data_insights/workflows/sql_agent.py` — added `SENSITIVE_COLUMN_NAMES`; strip sensitive cols from schema in `get_table_info()`; explicit-column (non-`*`) select in `_get_sample_rows()` with empty-guard.
- `data_insights/workflows/tools.py` — removed `collector__first_name`/`collector__last_name` from `VALID_RELATED_FIELDS`; `_top_collectors_monthly` returns username-only; `NoiseDetailTool` returns `has_audio` instead of the path; explanatory comments on the intentional-aggregate-analytics decision and PII reachability in `_get_default_allowed_tables`.
- `data_insights/workflows/prompt.py` — added the sensitive-column / contact-detail prohibition rule to `SQL_SYSTEM_TEMPLATE`.
- `data_insights/tests/test_phase3_pii_minimization.py` (new) — 19 DB-free `SimpleTestCase` tests (in-memory SQLite for the real `SQLDatabaseWrapper` code path; no Postgres/OpenAI needed) covering schema masking, sample-row masking, allow-list PII reachability, related-field allow-list, and the prompt rule.

## 4. Key decisions & tradeoffs
- **Schema-layer masking as primary control, prompt rule as secondary.** The `audio` column is removed before the agent ever sees the schema, so even a prompt-injected/jailbroken agent cannot `SELECT audio` — the column is absent from its world model. The prompt rule is defense-in-depth, not the sole guard.
- **Kept the audio-bearing data tables in the SQL allow-list** rather than excluding them. Excluding would have broken org-wide aggregate analytics (the product's core feature, explicitly in-scope to preserve). Masking the single sensitive column achieves minimization without scope loss.
- **Presence signal over omission for audio.** `NoiseDetailTool` returns `has_audio` (boolean) so legitimate "does this dataset have audio?" questions still work, without revealing the path.
- **Username preserved in `top_collectors`.** Per the product decision, collector usernames/display handles in existing leaderboards are acceptable; only the full legal name (first+last) was removed.
- **No per-user row scoping added** and **no restriction on which rows are aggregated** — honored the hard rule.
- **Phase 2 read-only SQL engine untouched**; no secrets rotated; no history rewritten; no commits made (left for the human).

## 5. Risks introduced
- **Hardcoded sensitive-column set.** `SENSITIVE_COLUMN_NAMES` matches by bare column name (`"audio"`) across all tables. If a future, legitimately non-sensitive column were ever named `audio`, it would be masked. Low likelihood; acceptable given the conservative direction (over-mask vs. leak).
- **`_get_sample_rows` empty-guard.** If a table's only column were sensitive, the sample preview returns `""`. Not possible for current data tables (all have benign columns), but worth noting.
- **Prompt rule is advisory.** It can be ignored/jailbroken — but the schema masking (the real control) does not depend on it, so a bypass does not re-expose the path.
- **Test isolation tradeoff.** Schema-masking tests use SQLite, not Postgres. The masking logic is engine-agnostic (operates on SQLAlchemy table metadata), so this is a faithful exercise, but it does not catch Postgres-specific reflection quirks.

## 6. Remaining concerns
- Sensitive-column matching is name-based and global, not `(table, column)`-scoped. Fine today; could become brittle as the schema grows.
- The masking only covers the **NL→SQL agent path** and the audited tool outputs. PII/storage exposure through other surfaces (REST serializers in `data/`, admin, file-download/streaming endpoints) was **not** in this phase's scope and was not changed.
- `recording_device` (free-text) and `recording_date` are returned by search/detail tools. Judged non-PII (device model + date), but if device strings ever carried serials/identifiers that assumption would need revisiting.

## 7. Recommended follow-up work
1. **Promote the sensitive-column policy to config/security settings** (e.g. an `AI_INSIGHT["SECURITY"]["SENSITIVE_COLUMNS"]` allow/deny, optionally `table.column`-qualified) so it is centrally auditable and extendable without code edits.
2. **Add a runtime/post-generation guard** that rejects or rewrites any agent-generated SQL referencing a sensitive column — a third layer beyond schema masking + prompt.
3. **Audit the non-agent surfaces** (DRF serializers, admin, audio download/streaming) for raw-path/PII exposure in a separate phase.
4. **Add an integration test against Postgres** (the production engine) mirroring the SQLite schema-masking tests, ideally in CI.
5. **Install `pytest`/`pytest-django` in the venv** — Phase 3 tests had to be run via `manage.py test` because `pytest` is absent from `./venv`.

---
*Verification: `./venv/bin/python manage.py test data_insights.tests.test_phase3_pii_minimization` → Ran 19 tests, OK (PostgreSQL). Prior stages: critique `needs_revision` (1 must_fix, now resolved); testing `check_passed: true`, no regressions; validation `acceptance_met: true`.*

## Backlog (next phases)

- P1: Add a runtime/post-generation SQL guard that rejects or rewrites agent-generated SQL referencing any sensitive column (third defense layer beyond schema masking + prompt).
- P1: Audit non-agent exposure surfaces (DRF serializers in data/, Django admin, audio download/streaming endpoints) for raw storage paths and PII — out of scope this phase.
- P2: Move SENSITIVE_COLUMN_NAMES into central security config (e.g. AI_INSIGHT['SECURITY']['SENSITIVE_COLUMNS']), optionally table.column-qualified, so it is auditable and extendable without code changes.
- P2: Add a Postgres-backed integration test mirroring the SQLite schema-masking tests and wire it into CI.
- P3: Install pytest/pytest-django into ./venv so the suite runs under pytest (currently only via manage.py test).
- P3: Re-evaluate recording_device free-text exposure if device strings ever begin carrying serials/unique identifiers.
