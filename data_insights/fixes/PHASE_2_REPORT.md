# Phase 2 — NL→SQL agent hardening (read-only execution, SELECT-only, DoS guard)

# Phase 2 — NL→SQL Agent Hardening (read-only execution, SELECT-only, DoS guard) — Phase Report

Branch: `data-insights-fixes` (working tree, uncommitted — the human commits). App under review: `data_insights/`. Repo root: `/Users/kelvinackah/Desktop/projects/freelance/datacollection`.

Scope guardrails honored: only the **SQL-agent SQLAlchemy engine** was made read-only. The Django ORM connection and the LangGraph `PostgresSaver` checkpointer use **separate** connections and were left writable (`DB_URI` in `data_insights/views.py:90` is still used by the checkpointer at line 109 and was intentionally not changed). Org-wide READ analytics remains intentional — the engine may `SELECT` across all collectors; only writes are blocked. No secrets rotated, no history rewritten, no commit/push/pre-commit run.

Prior-stage gates: Critique = approve (0 must_fix); Testing check_passed = true (0 regressions); Validation acceptance_met = true. A dedicated DB-free test module was added (`data_insights/tests/test_phase2_sql_agent_hardening.py`, 24 tests) — all pass locally, and the Phase 1 suite (19 tests) still passes with no regression.

## 1. Issues fixed (with final status)

| # | Category | Issue | Fix | Status |
|---|----------|-------|-----|--------|
| P2-1 | Read-only execution | The NL→SQL agent built a plain read-WRITE engine via `create_engine(...)`; any bypass of the regex/SELECT guard could `INSERT`/`UPDATE`/`DELETE`/run DDL, and runaway/cartesian queries could hang the DB. | New `create_readonly_engine(...)` attaches a SQLAlchemy `"connect"` event listener that runs `SET default_transaction_read_only = on` and `SET statement_timeout = <ms>` on **every** pooled connection (covers query execution, schema reflection, and sample-row reads). Postgres now rejects writes/DDL server-side and kills long queries independently of the application-layer guard. `TextToSQLAgent.__init__` now uses it. | FIXED |
| P2-2 | SELECT-only enforcement | The top-level statement-type check was commented out, and `"UNKNOWN"`-typed statements were silently skipped — a bypass. | Re-enabled SELECT/WITH enforcement on the top-level statement; statements whose first meaningful token is not `SELECT`/`WITH` are rejected. `"UNKNOWN"` statements are no longer skipped; only whitespace-only fragments are. Parenthesis recursion does NOT apply top-level SELECT-only (so `IN (1,2,3)` and scalar subqueries still pass), while writes inside parens are still caught by the `UNSAFE_KEYWORDS` scan. | FIXED |
| P2-3 | DoS / multi-statement | Comment-hidden payloads (`-- …`, `/* … */`), stacked statements (`SELECT …; DROP …`), and time-based DoS functions were not reliably blocked. | Comments are stripped via `sqlparse.format(..., strip_comments=True)` **before** validation (replaces the old commented-out `--` regex rule); multi-statement input is rejected (`len(meaningful) > 1`); added regex patterns for `pg_sleep*`, `pg_terminate_backend`/`pg_cancel_backend`, statement-initial `COPY`, and `lo_import`/`lo_export`. `statement_timeout` (default 15s) is the server-side backstop. | FIXED |
| P2-4 | Dead-code / consistency | `extract_sql` still preferred a `CREATE TABLE … AS …;` DDL extraction path that, under a read-only SELECT-only engine, could only ever produce a guaranteed failure. | Removed the `CREATE TABLE AS` extraction branch; the WITH/SELECT extraction paths remain for normal analytics. | FIXED |
| P2-5 | Pagination re-execution path | `ChatSessionView` re-executed agent-generated SQL in the `query_kind == "sql"` branch using a plain `create_engine(DB_URI)` — a second, unhardened write-capable execution path (and a known connection-leak hotspot per the deep-eval). | Switched that path to `create_readonly_engine(...)` sourced from `DB_CONFIG`, so re-executed SQL inherits the same read-only + statement_timeout enforcement. | FIXED |
| P2-6 | Least-privilege credentials | No support for a dedicated read-only DB role; the agent always used full app creds. | Added optional `AI_INSIGHT["DATABASE"]["READONLY_USER"/"READONLY_PASSWORD"]` (env `AI_INSIGHT_DB_READONLY_USER` / `AI_INSIGHT_DB_READONLY_PASSWORD`). When unset, falls back to app creds with a **loud `logger.warning`**; the engine is still forced read-only at the session level regardless. The exact least-privilege role-creation SQL is documented in the `_resolve_readonly_db_credentials` docstring. | FIXED |

## 2. Files modified

- `data_insights/workflows/sql_agent.py` — new module logger; `DEFAULT_SQL_STATEMENT_TIMEOUT_SECONDS = 15`; new helpers `_resolve_readonly_db_credentials` (with documented role-creation SQL), `_get_statement_timeout_seconds`, and `create_readonly_engine` (read-only + statement_timeout `connect` listener); `TextToSQLAgent.__init__` now builds its engine via `create_readonly_engine`; `_validate_sql_query` hardened (comment stripping before validation, multi-statement rejection, re-enabled top-level SELECT/WITH check via `top_level=True`, no longer skips `UNKNOWN` statements, added pg_sleep/terminate/COPY/lo_import-export DoS patterns); `extract_sql` drops the `CREATE TABLE AS` DDL path. (Also a couple of cosmetic line-wrap reflows.)
- `data_insights/views.py` — import `create_readonly_engine`; the `query_kind == "sql"` SQL re-execution path now uses `create_readonly_engine(...)` from `DB_CONFIG` instead of `create_engine(DB_URI)`. `DB_URI` retained — still used by the (writable) LangGraph checkpointer connection.
- `datacollection/settings.py` — added `READONLY_USER`, `READONLY_PASSWORD`, and `STATEMENT_TIMEOUT_SECONDS` (env `AI_INSIGHT_SQL_TIMEOUT_SECONDS`, default 15) to `AI_INSIGHT["DATABASE"]` in **both** the `USE_SQLITE` and Docker branches, with explanatory comments.
- `data_insights/tests/test_phase2_sql_agent_hardening.py` — NEW (untracked). 24 DB-free `SimpleTestCase`s: read-only/timeout listener behavior, ms conversion, credential resolution + warning fallback, settings exposure, and the full SELECT-only/DoS matrix (plain SELECT + CTE pass and get a bounded LIMIT; DROP/DELETE/UPDATE/INSERT/CTAS/multi-statement/comment-hidden write/pg_sleep/COPY/empty/unauthorized-table all rejected).

## 3. Key decisions & tradeoffs

- **Defense-in-depth, layered.** The regex/keyword guard is explicitly demoted (in code comments) to a "catch obvious abuse early" layer; the authoritative defenses are now (a) `default_transaction_read_only = on`, (b) the SELECT/WITH-only statement check, and (c) `statement_timeout`. This is deliberate — regex cannot reliably parse SQL, so correctness no longer depends on it.
- **Session-level read-only instead of a DB role.** Because role creation cannot be guaranteed from application code, read-only is enforced per-connection on the engine. Optional dedicated read-only creds are supported and strongly recommended (with documented role SQL); the loud warning on fallback makes the weaker posture visible in logs rather than silent.
- **Top-level-only SELECT enforcement.** SELECT-only is applied to the top-level statement, NOT to recursively-validated parenthesis contents, so legitimate value lists (`IN (...)`) and scalar subqueries are not broken. Writes nested in parentheses are still caught by the `UNSAFE_KEYWORDS` scan.
- **Comments stripped before validation** (not merely regex-matched) so comment-based obfuscation cannot hide a write/DoS from any downstream check.
- **Config-driven with backward-compatible defaults** (`STATEMENT_TIMEOUT_SECONDS=15`, read-only creds optional). The 15s default is intentionally generous so legitimate aggregations are unaffected; tunable via env without code edits.
- **Pagination path hardened too**, since it independently re-executes agent SQL — leaving it on a plain engine would have been a SELECT-only/read-only bypass.
- **Both settings branches edited** (SQLite + Docker) to preserve the existing duplicated structure rather than refactoring config — lower risk for a surgical phase, consistent with Phase 1.
- **Checkpointer/ORM untouched.** `DB_URI` and `create_engine` were retained exactly where the writable checkpointer needs them.

## 4. Risks introduced

- **`statement_timeout` could clip a genuinely long legitimate aggregation** if one exceeds 15s. Mitigated by env tunability (`AI_INSIGHT_SQL_TIMEOUT_SECONDS`); default chosen to be generous. Low risk.
- **Stricter validation may reject a small number of previously-accepted edge queries** (e.g. anything that parsed as `UNKNOWN`, or multi-statement inputs). This is the intended security behavior; risk is that an unusual-but-legitimate query shape is now blocked. Plain SELECT/CTE analytics are re-tested and pass. Low risk.
- **Fallback to app (write) credentials** when no read-only role is configured means the *credential* is still write-capable even though the *session* is forced read-only — a single missing `SET` would matter. Mitigated by the `connect`-event enforcement on every pooled connection plus the loud warning. Residual, by design until a read-only role is provisioned.
- **Statement timeout is interpolated into SQL via f-string** (`SET statement_timeout = {timeout_ms}`). The value is an `int` derived from settings/env and coerced via `int(...)`/`max(1, ...)`, so it is not user-controlled — no injection vector. Noted for reviewer awareness.

## 5. Remaining concerns

- **No live-Postgres integration test** asserts that an actual write is rejected end-to-end; the 24 tests are DB-free (they assert the listener issues the correct `SET` statements and that the validator/credential logic behaves correctly). The server-side enforcement itself is standard Postgres behavior but is not exercised against a real DB in CI.
- **Connection-pool lifecycle** of the pagination-path engine: `create_readonly_engine` is now used there, but per the deep-eval (`DEEP_EVALUATION_REPORT.md:392`) that path historically created an engine per request and never disposed it. This phase made the engine read-only but did not change its lifecycle — the potential leak under pagination load persists.
- **No dedicated read-only DB role exists yet** in any environment; production currently runs on the warning-logged fallback path until ops provisions `ai_insight_ro` and sets the env vars.

## 6. Recommended follow-up work

1. Provision the least-privilege `ai_insight_ro` Postgres role (SQL is documented in `sql_agent.py`'s `_resolve_readonly_db_credentials` docstring) and set `AI_INSIGHT_DB_READONLY_USER` / `AI_INSIGHT_DB_READONLY_PASSWORD` in each environment to eliminate the write-credential fallback.
2. Add a marked live-Postgres integration test (opt-in) that confirms a write is actually rejected and that `statement_timeout` fires, complementing the DB-free unit tests.
3. Fix the per-request engine creation / non-disposal in the `query_kind == "sql"` pagination path (cache or `engine.dispose()`), per `DEEP_EVALUATION_REPORT.md:392`.
4. Consider extracting the duplicated `AI_INSIGHT["DATABASE"]`/`AGENT` config (SQLite vs Docker branches) into a single source to avoid future drift across the two blocks.
5. Add structured metrics/alerting on `statement_timeout` cancellations so an overly-tight limit (or an abusive query pattern) surfaces operationally.

## Backlog (next phases)

- P1: Provision least-privilege ai_insight_ro Postgres role (SQL documented in sql_agent.py) and set AI_INSIGHT_DB_READONLY_USER/PASSWORD per environment to remove the write-credential fallback.
- P2: Add opt-in live-Postgres integration test asserting writes/DDL are rejected and statement_timeout cancels a long query, complementing the DB-free unit tests.
- P2: Fix per-request engine creation/non-disposal in the views.py SQL pagination path (cache or dispose) to stop the connection leak under load.
- P3: Add metrics/alerting on statement_timeout cancellations to detect overly-tight limits or abusive query patterns.
- P3: Deduplicate the AI_INSIGHT DATABASE/AGENT config across the USE_SQLITE and Docker branches in settings.py to prevent drift.
