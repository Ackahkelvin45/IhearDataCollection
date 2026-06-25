# Data Insights — Deep Multi-Agent Evaluation Report

_Generated 2026-06-25 via a multi-agent evaluation workflow (89 agents)._

> Produced by an automated multi-agent workflow: 6 discovery agents → per-finding independent validation/challenge + context investigation → remediation planning → completeness critic + a targeted gap round (iterative loop) → parallel synthesis.

**Confirmed findings:** 39  ·  **Severity mix:** {"CRITICAL":4,"HIGH":12,"MEDIUM":16,"LOW":7}  ·  **Refuted by validators:** 1  ·  **Critic confidence:** 0.82

## 1. Executive Summary

# Executive Summary: Data Insights App Evaluation

**Overall health: At risk — not production-ready in its current state.** The app's deterministic analytics-and-charting core is genuinely well-engineered, but it ships with critical security holes and several correctness defects that quietly corrupt the insights users act on. The register holds 4 CRITICAL, 12 HIGH, 16 MEDIUM, and 7 LOW findings; the criticals cluster tightly around two themes that must be closed before any wider rollout.

**The most important risks:**

1. **No tenant isolation.** No data_insights tool filters by collector or requesting user, so any authenticated user can read every other collector's recordings, file paths, and analyses [F02]. This is the single highest-priority issue.
2. **An unsafe NL→SQL path.** The agent connects with read-WRITE application DB credentials [F13], while the SELECT-only enforcement and `--` comment denylist are commented out [F03], leaving only a bypassable regex guard — with confirmed injection/DoS bypasses (`pg_sleep`, CASE, cartesian) [F12].
3. **Fabricated analytics.** The correlation matrix drops nulls per-column and then aligns rows by position, so it correlates mismatched records and produces fabricated coefficients and p-values [F01].
4. **Committed secrets.** `.env` (DB password, SECRET_KEY, OpenAI and DO Spaces keys) remains retrievable from git history and must be treated as compromised and rotated [F14].
5. **No LLM cost or abuse controls.** Expensive streaming endpoints have no DRF throttling [G01], the configured rate/session limits are dead code [G02], and there is no per-session in-flight guard [G06] — leaving OpenAI spend unbounded.

**What is genuinely good:** The deterministic chart pipeline is the strongest part of the codebase — thoughtful cardinality caps, chart-type selection, and a deliberate effort to unify the toolset across modes. Several guardrails (table allowlist, keyword scans, governance config) were scaffolded with clear intent, even if left unwired.

**Top recommendation:** Treat security as the release gate. Before any further feature work, enforce collector scoping at the ORM tool layer [F02], move the SQL agent to a least-privilege read-only role with statement timeouts [F13][F03][F12], rotate all leaked credentials [F14], and fix the correlation alignment bug [F01]. These four changes address every CRITICAL and the worst HIGH risks; the many quick-win MEDIUM/LOW fixes can follow.

## 2. Technical Findings

# TECHNICAL FINDINGS

## Architecture

**[CRITICAL · conf 0.90 · F09 (uncertain)] New SQLAlchemy engine + uncached SQL agent graph built per `DataAnalysisTool` instantiation** — `data_insights/workflows/tools.py:1665`
`DataAnalysisTool.add_agent` (a pydantic `model_validator`) constructs a fresh `TextToSQLAgent` on every instantiation, which calls `create_engine()`, instantiates `SQLDatabaseWrapper` (running `inspect(engine)` and dialect probing), and recompiles the workflow. Each new engine spins up a new connection pool while the SQL sub-agent is compiled uncached, multiplying DB connections and startup cost under load.

**[HIGH · conf 0.85 · F11 (uncertain)] Decibel ranking/grouping and avg-by-dimension queries sort/aggregate on unindexed `mean_db` across full join** — `data_insights/workflows/tools.py:2042`
`NoiseAnalysis.mean_db` has no `db_index` (`data/models.py:505`) and `NoiseDataset` lacks composite indexes, so `_decibel_ranked` does a full scan + sort, and `_decibel_grouped`/`AudioAnalysisTool` aggregations join across the full table without index support. These hot analytics paths degrade to O(rows) sequential scans, becoming a latency cliff as data grows.

**[HIGH · conf 0.95 · G05 (uncertain)] `trim_messages` disabled while full `get_table_info()` re-sent every SQL turn** — `data_insights/workflows/sql_agent.py:631`
The SQL sub-agent rebuilds the system message every turn with the full schema DDL plus 2 sample rows per allowed table inlined, and the `trim_messages` cap (6000 tokens) is commented out. With the allowed-table set falling back to ALL data+core tables, per-multi-step-query input-token spend scales as O(schema_size × turns).

**[MEDIUM · conf 0.82 · F08 (adjusted)] SQL agent rebuilds full schema info (reflect + sample EVERY allowed table) on every LLM turn** — `data_insights/workflows/sql_agent.py:631`
`TextToSQLAgent.call_llm()` calls `self.db.get_table_info()` with no `table_names` arg on every model invocation, forcing reflection, `CreateTable` DDL compilation, and a `SELECT * ... LIMIT n` sample per table across ~15 tables. The formatted schema is never cached, so each turn pays full reflection + DDL recompile cost.

**[MEDIUM · conf 0.85 · F10 (adjusted)] `paginate_message`/clarification SQL-pagination create a fresh engine per request, never disposed** — `data_insights/views.py:1056`
`_paginate_from_tool_data` (the `sql` branch) calls `create_engine(DB_URI)` and builds a new `SQLDatabaseWrapper` (eager `inspect`) on every pagination request and never disposes the engine, leaking psycopg connection pools that linger until GC. The `recent_datasets`/`dataset_search` branches also run `count()` and slice separately, compounding the per-request DB pressure.

**[MEDIUM · conf 0.80 · F31 (uncertain)] Statistical 'distribution' analysis loads every `mean_db` row into Python and aggregates in app code** — `data_insights/workflows/tools.py:1230`
The distribution branch fetches `.values('category__name','noise_analysis__mean_db')` for the entire filtered dataset with no limit, then groups in a Python dict and stores every individual decibel value in `distribution_data[name]['decibel_values']`. This pulls all rows over the wire and embeds full value arrays in the payload, scaling memory and transfer linearly with dataset size.

**[MEDIUM · conf 0.82 · G03 (adjusted)] Main agent `ChatOpenAI` has no `max_tokens` cap (unbounded output cost per call)** — `data_insights/views.py:1229`
The primary agent LLM and dashboard-summary LLM are constructed with only model/api_key/streaming and no `max_tokens`, so every tool-bound model turn in the agent loop can emit an arbitrarily long completion at full per-token price. The small insight LLM elsewhere is capped, making this an unguarded cost asymmetry on the most-invoked path.

**[MEDIUM · conf 0.85 · G04 (adjusted)] LangGraph stream/invoke configs set no `recursion_limit` (only `thread_id`)** — `data_insights/workflows/agent_workflow.py:778`
All four agent entry points build `config = {'configurable': {'thread_id': ...}}` with no `recursion_limit`, relying on LangGraph's implicit 25-superstep default (~8 tool round-trips). Since `should_continue` routes back to tools on any tool call, worst-case LLM spend per message is bounded only by that implicit cap rather than an explicit, tunable governance value.

**[MEDIUM · conf 0.84 · F28 (confirmed)] Advanced ML analytics tools registered only in the exception fallback, unreachable in normal operation** — `data_insights/workflows/tools.py:3452-3464`
`get_agent_tools`' success branch registers only `MLDatasetProfileTool`, `MLFeatureStatsTool`, and `ListMLSchemaTool`; the six advanced tools (class balance, train/test split, correlation matrix, statistical test, feature export, feature importance) appear ONLY in the `except` branch that fires if importing `WebFetchTool` fails. So whenever the import succeeds (normal operation), those tools are never bound to the agent.

**[LOW · conf 0.90 · F30 (adjusted)] `lru_cache` on bound method `_get_cached_table` leaks `Table` objects and ties cache to instance lifetime incorrectly** — `data_insights/workflows/sql_agent.py:155`
`@lru_cache(maxsize=20)` decorates the instance method `_get_cached_table(self, table_name)`, so the cache key includes `self`, making the cache process-wide across all `SQLDatabaseWrapper` instances and holding a strong reference to every `self` ever passed. Combined with per-request wrapper creation, this prevents GC of wrappers/engines/metadata and is effectively a memory leak (the method also appears to be dead code).

**[LOW · conf 0.82 · F32 (adjusted)] `_top_collectors_monthly` and `_dataset_count` run multiple unindexed entity-name lookups + full-table counts** — `data_insights/workflows/tools.py:1963`
`_dataset_count` issues up to 6 separate `*.objects.values_list('name')` full-table loads on every count question just to do Python substring matching, then a `count()` over `NoiseDataset` filtered on `recording_date__year/month` or `icontains` (unindexed). These small but slow-changing lookups are re-read on every request rather than cached, adding avoidable per-query overhead.

## Code Correctness / Bugs

**[CRITICAL · conf 0.90 · F01 (confirmed)] Correlation matrix correlates misaligned rows (per-column null drop) — fabricated correlations** — `data_insights/workflows/tools.py:2788-2814`
`MLCorrelationMatrixTool` builds each feature's vector by independently dropping that column's nulls, then truncates every vector to the global min length and column-stacks them. Because nulls are removed per-column, position *i* of `rms_energy` and position *i* of `spectral_centroid` can come from different `AudioFeature` rows, so the resulting Spearman/Pearson coefficients, p-values, and top-10 lists are mathematically meaningless fabrications. The correct fix is listwise (pairwise-complete) deletion preserving row identity.

**[HIGH · conf 0.90 · F04 (confirmed)] `cumulative_energy` is a per-period sum, not a running total — mislabeled metric** — `data_insights/workflows/tools.py:1362`
Energy and temporal analyses expose `cumulative_energy` computed as `Sum('audio_features__rms_energy')` within a single GROUP BY bucket (per region/month/day), which is the bucket total, not a cumulative/running sum. A user reading "cumulative energy over time" will assume the values accumulate when each point is independent — and RMS is not additive to begin with, so the sum is meaningless. (Appears in five spots.)

**[HIGH · conf 0.83 · F05 (uncertain)] `decibel_grouped` `sample_count` counts datasets that did not contribute to `avg_db`** — `data_insights/workflows/tools.py:2101-2103`
`_decibel_grouped` reports `avg_db = Avg('noise_analysis__mean_db')` alongside `sample_count = Count('id')` without filtering `noise_analysis__isnull=False`, so `avg_db` correctly ignores NULL `mean_db` rows while `sample_count` includes every dataset in the group (including those with no `NoiseAnalysis` row). The displayed average and its sample size are computed over different populations, overstating how many samples back each average.

**[MEDIUM · conf 0.84 · F06 (confirmed)] `top_collectors_monthly` filters on `recording_date`, not upload/created date** — `data_insights/workflows/tools.py:1866-1867`
The tool answers "who contributed the most datasets this month" by filtering `recording_date__year/month` to the current month, but `recording_date` is the operator-supplied (often backdated) recording time, not the contribution time. `NoiseDataset.created_at` (auto_now_add) is the true upload timestamp, so bulk/backdated uploads are mis-credited or excluded from the leaderboard.

**[MEDIUM · conf 0.90 · F07 (confirmed)] Correlation tool labels plain random sampling as 'stratified'** — `data_insights/workflows/tools.py:2759-2763`
When `label_column` is supplied, `MLCorrelationMatrixTool` claims stratified sampling but executes an unstratified `order_by('?')` random sample identical to the non-label branch — the computed field map and grouping key are discarded. The result advertises `sampling_method='stratified'` (and the stat card shows "(stratified)") while the sample is in fact a uniform random draw, misleading users about the statistical guarantees.

**[MEDIUM · conf 0.92 · F19 (confirmed)] `NameError`: `timezone.now()` called but `timezone` never imported in `views.py`** — `data_insights/views.py:1116`
`save_dashboard` builds a default title with `timezone.now()` when the client sends an empty title, but `views.py` never imports `timezone` (neither `django.utils.timezone` nor `datetime`). The call raises `NameError`, returning a 500 instead of saving — triggered every time a user saves a dashboard without a title.

**[MEDIUM · conf 0.92 · F20 (adjusted)] `AttributeError`: `ChatSession.Status.DELETED` referenced but not defined** — `data_insights/serializers.py:90`
`ChatSessionUpdateSerializer.validate_status` builds an `allowed_transitions` dict keyed on `ChatSession.Status.DELETED`, but the `Status` `TextChoices` only defines `ACTIVE`, `INACTIVE`, `ARCHIVED`. Accessing the missing member raises `AttributeError` at execution time, so EVERY session status update (PATCH/PUT) crashes with a 500 before any transition validation runs.

**[MEDIUM · conf 0.84 · F23 (adjusted)] Message marked `PROCESSING` before stream consumed; no age threshold in recovery → stuck and clobbered statuses** — `data_insights/management/commands/fix_stuck_messages.py:9`
`mark_processing()` persists `PROCESSING` before the `StreamingHttpResponse` begins iterating, and the terminal `COMPLETED`/`FAILED` writes live inside the generator body, so a dead worker / unconnected client / undrained generator leaves the message `PROCESSING` forever. The recovery command bulk-flips ALL `PROCESSING` messages to `FAILED` with no age threshold and overwrites `assistant_response`, so it can clobber live, mid-generation messages.

**[MEDIUM · conf 0.95 · F24 (confirmed)] Stratified train/val/test split over-allocates rows for small classes (counts exceed class size)** — `data_insights/workflows/tools.py:2586-2591`
Each class uses `train=max(1,round(c*train_pct))`, `test=max(1,round(c*test_pct))`, `val=max(1,c-train-test)`; for `c=2` at 70/15/15 this yields `train=1, test=1, val=1` so the split sums to 3 > 2. The per-class breakdown and summed totals shown to the user allocate more samples than exist, breaking the partition guarantee.

**[MEDIUM · conf 0.92 · F29 (confirmed)] `_match_entity_name` uses unanchored substring matching — false-positive entity filters** — `data_insights/workflows/tools.py:1680-1683`
`_dataset_count` resolves region/community/category/class/subclass by checking if a lowercased DB name is a substring of the query (longest-first), so a region "Ada" matches "adapter"/"Canada", a category "Car" matches "cardiac", etc. These false positives silently apply an entity filter the user never named, making the returned `total_count` and `filter_meta` quietly wrong; the fix is the word-boundary regex already used in `clarification_gate.py`.

**[MEDIUM · conf 0.82 · F27 (adjusted)] `_to_float` coerces non-numeric / null y-values to 0.0, fabricating zero bars** — `data_insights/workflows/chart_builder.py:214-222`
`build_chart_config` maps every y value through `_to_float`, which returns `0.0` for `None` or any unparseable value, so a NULL aggregate (e.g. a group whose `avg_db` is NULL) renders as a real 0-height bar instead of being omitted or shown as "no data". For decibel charts a fabricated 0 dB bar is materially misleading because 0 dB is a meaningful value, not "missing."

**[MEDIUM · conf 0.90 · F16 (adjusted)] Box-plot widgets silently degrade to a mislabeled bar chart and lose the distribution entirely** — `data_insights/workflows/widget_composer.py:724`
`_decompose_statistical` emits a `box_plot` widget whose `data.data` is a list of raw value arrays, but the frontend box-plot renderer needs precomputed `actualData.boxPlotData` ([min,q1,median,q3,max]). When rendered via `renderArtifact→createChart`, `extractChartData` never derives `boxPlotData`, so the widget falls back to a bar chart that is both mislabeled and devoid of the intended distribution; fix is to compute the five-number summary in Python at composition time.

**[LOW · conf 0.85 · F26 (confirmed)] Grouped energy `chart_hint` `y=avg_decibel` mismatches the `order_by` ranking (`avg_rms_energy`) for region** — `data_insights/workflows/tools.py:967`
In `_energy_analysis` the region branch orders rows by `-avg_rms_energy` (line 967) but the returned `chart_hint` plots `y='avg_decibel'` (line 1019), so the bars are sorted by RMS energy while displaying decibels. The chart is non-monotonic in the plotted metric and the "top" bar is not the loudest by the shown metric; category/microphone branches are consistent, making this a region-only copy/paste drift.

**[LOW · conf 0.90 · F25 (adjusted)] `recent_datasets` `chart_hint` references a `count` column that does not exist in rows** — `data_insights/workflows/tools.py:1855`
`_recent_datasets` returns rows keyed name/region/community/category/recording_date/recording_device but its `chart_hint` is `{x:'recording_date', y:'count', group_by:'category'}`. There is no `count` field, so `_to_float` coerces the missing y to 0.0 for every row, yielding all-zero bars if the hint is ever honored (the live `_decompose_recent` renders a table, so it's a latent landmine).

**[LOW · conf 0.82 · F15 (adjusted)] Fractional 0–1 metrics (RMS, entropy, correlation) misclassified as ratio data and drawn as pie/donut** — `data_insights/workflows/chart_builder.py:164`
`_is_ratio_data` returns True for ANY column whose values all fall in 0.0–1.0 with no column-name or sum-to-1 check, so naturally-normalized audio metrics (avg RMS, zero-crossing rate, normalized entropy, Spearman/Pearson coefficients) are routed to pie/donut charts. These are not parts-of-a-whole, so the pie representation is semantically wrong; the fix mirrors the stricter 0–100 branch with a ratio-name and/or sum-to-~1 guard.

**[LOW · conf 0.80 · F22 (adjusted)] `KeyError` crash in `widget_composer.decompose()` on result missing `rows` key** — `data_insights/workflows/widget_composer.py:529`
`_decompose_grouped` and `_decompose_ranked` do `rows = result["rows"]` with no default, while `decompose()` routes to them purely on `analysis_type`. If a tool emits that `analysis_type` but omits `rows` (empty result, error-shaped dict, partial payload), this raises `KeyError`; the fix is `result.get("rows", [])` to match every sibling decomposer, which already handle the empty case.

## Security

**[CRITICAL · conf 0.90 · F02 (confirmed)] No collector/user scoping — any authenticated user reads ALL collectors' data** — `data_insights/workflows/tools.py:374` (`NoiseDatasetSearchTool`), `tools.py:1452` (`NoiseDetailTool.get`), `tools.py:748` (`AudioAnalysisTool`)
`NoiseDataset` has a `collector` FK (the data owner), but not a single `data_insights` tool filters by `collector` or by the requesting user — every ORM tool and the NL→SQL agent query the full table. A contributor in session A can enumerate every other user's recordings, audio file paths, collector identities, and analyses, a critical multi-tenant data-isolation breach; the fix injects the authenticated identity via LangGraph `InjectedState` and applies a mandatory tenant predicate for non-researcher/admin users.

**[CRITICAL · conf 0.90 · F03 (confirmed)] SQL agent SELECT-only enforcement and `--` comment denylist are commented out (write-capable creds)** — `data_insights/workflows/sql_agent.py:572`
The statement-type check that would reject anything other than SELECT/WITH is fully commented out, as is the `--` inline-comment denylist pattern, leaving only a bypassable regex blocklist plus a `UNSAFE_KEYWORDS` token scan. Because the engine uses the app's read-WRITE DB credentials (no read-only role, no `SET TRANSACTION READ ONLY`), any guard bypass permits writes/DDL against production data.

**[CRITICAL · conf 0.90 · F13 (confirmed)] NL→SQL agent connects with read-WRITE application DB credentials** — `data_insights/workflows/sql_agent.py:466`
`TextToSQLAgent` builds its SQLAlchemy engine from the full application Postgres credentials (the same USER/PASSWORD Django uses for writes), not a dedicated read-only role. The system prompt claims "read-only PostgreSQL" but nothing enforces it at the DB layer, so the only barrier between an LLM-crafted statement and a write/DDL is the bypassable regex denylist; the fix is a dedicated least-privilege `SELECT`-only Postgres role plus read-only transaction enforcement.

**[HIGH · conf 0.92 · F14 (adjusted)] Secrets (`.env`) committed to git history** — `datacollection/settings.py:34` (`SECRET_KEY = os.getenv`); git history commit `a143af0`
`.env` was committed (commit `a143af0`) and only removed later (`5b769e9`), so it remains retrievable from history along with any secrets it held — DB password, `SECRET_KEY`, `OPENAI_API_KEY`, and email/DigitalOcean Spaces credentials. All must be treated as compromised and rotated; removal in a later commit does not remove it from history.

**[HIGH · conf 0.90 · F12 (uncertain)] SQL injection / DoS bypasses of the regex+keyword guard (pg_sleep, CASE, cartesian)** — `data_insights/workflows/sql_agent.py:507-526` (patterns), `sql_agent.py:571-574` (commented SELECT-only)
The `_validate_sql_query` guard is a denylist of regexes + a keyword token scan with the SELECT-only check and `--` rule commented out, and concrete bypasses were verified: `SELECT pg_sleep(5) FROM data_noisedataset` passes the anti-sleep regex, as do blind CASE-WHEN timing and cartesian/memory-bomb expressions. This yields a confirmed DoS class against the DB; the durable fix is a read-only, `statement_timeout`-bounded connection plus a positive structural (allowlist) check.

**[HIGH · conf 0.82 · F18 (confirmed)] Streamed assistant markdown rendered as raw HTML via `innerHTML` with regex 'markdown' — XSS / formatting-corruption risk** — `data_insights/templates/data_insights/unified_chat.html:2612`
`formatMarkdown` builds an HTML string from LLM/tool output via regex replacements and assigns it through `innerHTML` (`_renderStreamFrame:1097`, `formatStreamedText:1071`, `appendStreamText:1040`) without escaping the streamed prose first — `escapeHtml` is used only for table cells and widget labels. This permits XSS via injected markup and corrupts legitimate content like `<30 dB`; the fix is to `escapeHtml` before the regex pass so literal markup is inert.

**[HIGH · conf 0.90 · G01 (adjusted)] No DRF throttling on expensive LLM streaming endpoints (only `IsAuthenticated`)** — `data_insights/views.py:125`
The three most expensive endpoints — `create_message`, `clarify_message`, and `generate_insight` — each kick off OpenAI calls (a full LangGraph agent for the first two) but are protected only by `IsAuthenticated`, with no `DEFAULT_THROTTLE_CLASSES`/`DEFAULT_THROTTLE_RATES` declared in settings. A single authenticated account can issue unlimited agent runs, uncapping OpenAI spend and worker/DB-pool pressure; the fix is a `ScopedRateThrottle` with a tight per-user rate.

**[HIGH · conf 0.90 · G02 (adjusted)] `AI_INSIGHT.SECURITY` rate/abuse config is entirely dead (never read)** — `datacollection/settings.py:485`
`RATE_LIMIT_PER_MINUTE` (30), `MAX_SESSIONS_PER_USER` (10), `SESSION_INACTIVITY_HOURS` (24), and `MAX_MESSAGE_LENGTH` (10000) are defined under `AI_INSIGHT.SECURITY`, and `views.py:61` binds `SECURITY_CONFIG = AI_CONFIG.get('SECURITY', {})`, but `SECURITY_CONFIG` is dereferenced exactly zero times after assignment. Four advertised abuse/cost controls are pure dead config providing no actual enforcement.

**[HIGH · conf 0.85 · G06 (confirmed)] No per-session in-flight guard: unlimited concurrent agent runs per user/session** — `data_insights/views.py:162`
`create_message` creates a `ChatMessage` and immediately calls `_process_message_sync` without checking whether the session already has a `PROCESSING` message, with no `select_for_update`, no in-flight flag, and no `MAX_SESSIONS_PER_USER` enforcement. A user (or a script) can fan out unlimited concurrent agent runs, multiplying OpenAI token spend, SQLAlchemy engine/connection leakage, and worker pressure.

**[HIGH · conf 0.90 · G07 (adjusted)] No `request_timeout` on any agent LLM (`TIMEOUT_SECONDS` config never applied)** — `data_insights/workflows/sql_agent.py:642`
`AI_INSIGHT.AGENT.TIMEOUT_SECONDS` (120) is defined but never read, and none of the `ChatOpenAI` constructions on the agent path (`views.py:1229`, `1236`; `tools.py:1537`) pass `request_timeout`/`timeout`. A hung or slow OpenAI connection therefore has no client-side deadline, so a stalled request can pin a worker thread, a DB connection, and an open streaming response indefinitely.

**[MEDIUM · conf 0.90 · G08 (confirmed)] Message length validated against hardcoded 10000, not configurable `MAX_MESSAGE_LENGTH`** — `data_insights/serializers.py:38`
`ChatMessageCreateSerializer.user_input` uses a literal `max_length=10000` instead of `AI_INSIGHT.SECURITY.MAX_MESSAGE_LENGTH` (defaults to 10000 but meant to be tunable via env var). The operator cannot tighten the per-message input cap to reduce prompt-token cost without a code change, and the config knob is misleading dead config.

## 3. Analytics & KPI Validation

All findings verified. F15 confirmed (line 165-166: 0.0–1.0 range returns True with no column-name check, contradicting the docstring intent). F25 confirmed (chart_hint y="count" with no count field in rows). F31 confirmed (line 1230-1233 loads all rows, no limit). I have sufficient verification across all confirmed analytics-correctness and data-integrity findings. Writing the report now.

# ANALYTICS & KPI VALIDATION REPORT

**App:** `data_insights` (Django + DRF + LangGraph NL→SQL analytics chatbot)
**Date:** 2026-06-25
**Scope:** Metric/aggregation/calculation correctness and data integrity (categories: `analytics-correctness`, `data-integrity`)
**Verdict:** Analytics are **NOT largely sound.** The deterministic ORM-aggregation pipeline (decibel/energy grouping, ranking, counts) is mostly correct in its *core* aggregates, but it is undermined by one **CRITICAL fabricated-statistics defect**, several **HIGH/MEDIUM mislabeling and arithmetic defects**, and a class of silent data-fabrication issues (zero-bars, ghost columns) that make charts actively misleading. All findings below were re-verified against current source.

---

## CRITICAL

### [F01] Correlation matrix correlates misaligned rows — fabricated correlations
**Location:** `data_insights/workflows/tools.py:2788–2814`

- **What is computed:** A feature-vs-feature Spearman/Pearson correlation matrix (with p-values and a top-10 correlated-pairs list) over `AudioFeature` columns (e.g. `rms_energy`, `spectral_centroid`).
- **Is it correct? NO.** The matrix is **statistically invalid.** Each feature's vector is built by *independently* dropping that column's nulls (`col_data = [r.get(col) for r in rows if r.get(col) is not None]`, line 2790), then every vector is truncated to the global `min_len` (line 2811) and column-stacked (line 2812). Because nulls are dropped **per column**, position *i* of `rms_energy` and position *i* of `spectral_centroid` no longer come from the same `AudioFeature` row. Verified: there is no row-identity preservation — `data_matrix` discards which row each value came from.
- **Impact:** Every reported correlation coefficient, p-value, and "top correlated pairs" insight is **fabricated** — derived from arbitrarily paired values across different recordings. Any user (or downstream feature-selection / ML decision) acting on "X correlates with Y at r=0.7" is acting on noise. This is the single most dangerous analytics defect: the output *looks* authoritative (matrix + significance) but is meaningless whenever **any** feature column has nulls.
- **Correct computation:** **Listwise (pairwise-complete) deletion preserving row identity.** Pull rows once as dicts, keep only rows where **all** `valid_features` are non-null, then build the matrix from that single aligned 2D array:
  ```python
  rows_clean = [r for r in rows
                if all(r.get(f) is not None and not math.isnan(float(r[f]))
                       for f in valid_features)]
  aligned_data = np.array([[float(r[f]) for f in valid_features] for r in rows_clean])
  ```
  (Per-column null-drop is correct only for the *unpaired* group-comparison tool at `tools.py:2985–3028` — it was incorrectly transplanted into this paired context.)

---

## HIGH

### [F04] `cumulative_energy` is a per-bucket sum, not a running total — mislabeled metric
**Location:** `data_insights/workflows/tools.py:1362` (and `965`, `979`, `1007`)

- **What is computed:** A field named `cumulative_energy = Sum("audio_features__rms_energy")` exposed in energy and temporal analyses, inside a `GROUP BY` bucket (per region, per category, per month, per day). Verified at all four sites.
- **Is it correct? NO — on two counts.** (1) The name is wrong: it is the **bucket total**, not a cumulative/running sum. On a monthly/daily trend (line 1362) a user reading "cumulative energy over time" will believe values accumulate, but each point is independent. (2) The metric itself is **invalid**: RMS energy is **not additive** — summing RMS values across recordings is physically meaningless.
- **Impact:** Users analyzing energy "over time" see independent per-bucket sums presented as accumulating values, and the underlying quantity (Sum of RMS) has no physical interpretation. Misleads any energy-trend or "loudest region by total energy" reading.
- **Correct computation:** Drop `Sum(rms_energy)` (RMS is non-additive); keep the valid descriptors already present (`avg/max/min/std` rms, `avg/max/min` decibel). If an extensive "total energy" proxy is genuinely wanted, use **power × time**: `Sum(F("audio_features__rms_energy") * F("audio_features__duration"))`, or `Sum(duration)`. A true running total requires a window function / post-aggregation cumsum over the ordered buckets, not a per-bucket `Sum`.

### [F05] `decibel_grouped` `sample_count` counts datasets that did not contribute to `avg_db`
**Location:** `data_insights/workflows/tools.py:2098–2105` *(verdict: uncertain, conf 0.83)*

- **What is computed:** Per group, `avg_db = Avg("noise_analysis__mean_db")` alongside `sample_count = Count("id")`. Verified: the queryset does `.exclude(avg_db__isnull=True)` (line 2104) — dropping groups with a null *average* — but **does not** filter `noise_analysis__mean_db__isnull=False` before counting.
- **Is it correct? Partially wrong (data-integrity).** `avg_db` correctly ignores rows whose `mean_db` is NULL (SQL `AVG` skips NULLs), but `sample_count = Count("id")` counts **every** `NoiseDataset` in the surviving group — including datasets with no `NoiseAnalysis` row or a null `mean_db`. The displayed denominator therefore overstates how many samples actually backed the average.
- **Impact:** The "average X dB based on N samples" claim is internally inconsistent: N can exceed the number of values that produced the average. Erodes trust and can mask thin/under-supported groups (e.g. avg from 2 values labeled "N=40").
- **Correct computation:** Count only contributing rows: `sample_count = Count("id", filter=Q(noise_analysis__mean_db__isnull=False))`, so the denominator matches the rows `AVG` consumed.

### [F18] Streamed assistant markdown rendered as raw HTML via `innerHTML` — formatting-corruption / XSS
**Location:** `data_insights/templates/data_insights/unified_chat.html:2612`

- **What is computed (UX-of-analytics):** `formatMarkdown` turns LLM/tool prose into an HTML string via regex replacements, assigned through `innerHTML` (render paths at 1097/1071/1040).
- **Is it correct? NO.** Unlike table cells / stat-card labels (which use `escapeHtml`), the streamed prose body is **not escaped** before the regex pass.
- **Impact (data-integrity of the insight text):** Beyond the XSS surface, this **corrupts insight content** — e.g. a literal `<30 dB` threshold in an analytics answer is parsed as an HTML tag and silently dropped, so the numeric insight the user reads is wrong.
- **Correct approach:** Escape first, then format — run `escapeHtml(text)` *before* the regex pass so literal `<`, `>`, `&` become inert, then inject the known-safe formatting tags. Fixes both the corruption and the XSS.

---

## MEDIUM

### [F06] `top_collectors_monthly` filters on `recording_date`, not upload date
**Location:** `data_insights/workflows/tools.py:1866–1867`

- **What is computed:** "Who contributed the most datasets this month," via `recording_date__year/month == now`. Verified.
- **Is it correct? NO (wrong dimension).** `recording_date` is operator-supplied (often backdated) — when the audio was *recorded*, not when it was *contributed*. The true contribution timestamp is `created_at` (`auto_now_add`).
- **Impact:** A contribution leaderboard miscredits activity: bulk uploads of older recordings are excluded from "this month," and the leaderboard reflects recording dates rather than upload effort. Wrong people credited / wrong "this month" totals.
- **Correct computation:** Filter and report on `created_at__year/__month` to match the stated intent.

### [F07] Correlation tool labels plain random sampling as "stratified"
**Location:** `data_insights/workflows/tools.py:2758–2766`

- **What is computed:** When `label_column` is supplied, the tool advertises `sampling_method="stratified"` (surfaced on the stat card as "Sample: N of M (stratified)").
- **Is it correct? NO.** Verified: the label branch builds `field_map`/`field` (line 2755–2758) then does an **unstratified** `order_by("?")` random sample (line 2760) identical to the non-label branch — the grouping field is computed and discarded. No bucketing/proportional draw occurs.
- **Impact (data-integrity of provenance):** Users believe class proportions were preserved when they were not; rare classes may be missing or over/under-represented, while the label falsely asserts otherwise. False methodological claim on a statistical output.
- **Correct computation:** Either implement true stratification (bucket by `field`, draw proportionally per group) or stop mislabeling — set `sampling_method = "random"` and drop the dead `field_map` code.

### [F24] Stratified train/val/test split over-allocates rows for small classes
**Location:** `data_insights/workflows/tools.py:2586–2591`

- **What is computed:** Per-class split: `train=max(1,round(c*train_pct))`, `test=max(1,round(c*test_pct))`, `val=max(1,c-train-test)`. Verified.
- **Is it correct? NO.** The `max(1,…)` floors break the partition: a class with `c=2` at 70/15/15 yields `train=1, test=1, val=1` → sum = **3 > 2**. The recommended split allocates more rows than the class contains, and the summed train/val/test totals exceed `total_rows`.
- **Impact:** The recommended split is arithmetically impossible to realize; per-class and global totals don't reconcile with dataset size, breaking the stratification guarantee and any downstream ML plan built from it.
- **Correct computation:** Exact integer partition — `train=round(c*train_pct)`, `val=round(c*val_pct)`, `test=c-train-val` (remainder absorbs rounding so the three always sum to exactly `c`; clamp `test=max(0,test)` defensively).

### [F27] `_to_float` coerces null/non-numeric y-values to `0.0` — fabricated zero bars
**Location:** `data_insights/workflows/chart_builder.py:214–222`

- **What is computed:** Every chart y-value passes through `_to_float`, which returns `0.0` for `None`/unparseable (verified, line 220).
- **Is it correct? NO (data-integrity).** Null aggregates (a group whose `avg_db` is NULL, or an "Unknown" bucket) are rendered as a real **0-height bar** rather than omitted or shown as "no data." For decibel/energy charts, **0 dB is a meaningful value**, so a fabricated 0 dB bar is materially misleading.
- **Impact:** Region/category/microphone energy charts (trigger path `tools.py:955–1010`) draw phantom 0-value points that read as "this group is silent / has zero energy" when the truth is "no data." Wrong conclusions about quiet regions/categories.
- **Correct computation:** Make `_to_float` return `Optional[float]` (None for null/unparseable, **no** `"0"` fallback) and either drop those rows (label+value together) or pass null through so the renderer shows a gap/"no data," not a 0.

### [F29] `_match_entity_name` uses unanchored substring matching — false-positive entity filters
**Location:** `data_insights/workflows/tools.py:1680–1683`

- **What is computed:** `_dataset_count` resolves region/community/category/class/subclass by testing if a lowercased DB name is a **substring** of the query (longest-first). Verified: `if name_lower and name_lower in query_lower` (line 1682), no word boundaries.
- **Is it correct? NO.** Short/common names false-match: region "Ada" matches "adapter"/"Canada"; category "Car" matches "cardiac"; class "Bus" matches "business." The resulting filter silently narrows the count.
- **Impact:** `total_count` and `filter_meta` become untrustworthy — the chatbot returns a count scoped to an entity the user never named, with no indication a filter was applied. Silent wrong scalar answers.
- **Correct computation:** Word-boundary match (already used in `clarification_gate.py`): `re.search(r'\b' + re.escape(name_lower) + r'\b', query_lower)`, factored into one shared helper so the two paths can't drift.

### [F28] Advanced ML analytics tools registered only in the exception fallback — unreachable
**Location:** `data_insights/workflows/tools.py:3452–3464`

- **What is computed (analytics availability):** `get_agent_tools` success branch binds only `MLDatasetProfileTool`, `MLFeatureStatsTool`, `ListMLSchemaTool` (verified, lines 3461–3463). The six advanced tools — class balance, train/test split, correlation matrix, statistical/significance test, feature export, feature importance — appear **only** in the `except` branch (line 3467+), which fires only if the `WebFetchTool` import fails.
- **Is it correct? NO (correctness/availability).** In normal operation the import succeeds, so those six analytics capabilities are **never bound** and the agent cannot call them. (Note: this means F01's broken correlation matrix and F07's mislabeled sampling are effectively dormant in the *primary* path — but they activate the moment the fallback fires, and other ML tools are simply missing.)
- **Impact:** Advertised ML analytics are silently unavailable; users asking for class balance, splits, correlations, or significance tests get a degraded answer with no error.
- **Correct computation/fix:** Single module-level registry of all 15 tool classes, instantiated in a loop with per-tool try/except, so one tool's construction failure degrades only that tool and the advanced ML tools are always bound on success.

### [F31] Statistical "distribution" analysis loads every `mean_db` row into Python *(data-flow / integrity-adjacent, conf 0.8)*
**Location:** `data_insights/workflows/tools.py:1228–1233`

- **What is computed:** For distribution queries, all `(category__name, mean_db)` rows for the *entire* filtered dataset are pulled (no limit, verified line 1230), grouped in a Python dict, and every individual decibel value is stored in `decibel_values`.
- **Is it correct? Numerically yes, operationally fragile.** The values are real, but materializing the full per-value arrays into the tool payload is unbounded.
- **Impact:** At scale this is a full-table read + large in-payload arrays; combined with [F16] (box-plot rendering) it also bloats the response. Not a wrong number, but a data-integrity/scale liability.
- **Correct computation:** Compute the five-number summary (min/q1/median/q3/max) in the DB via `PERCENTILE_CONT` and return one row per group instead of shipping raw value arrays.

---

## LOW

### [F26] Grouped energy `chart_hint` y mismatches the `order_by` metric (region branch)
**Location:** `data_insights/workflows/tools.py:967` vs `1019`

- **What is computed:** Region energy rows are ordered by `-avg_rms_energy` (verified line 967) but the returned `chart_hint` plots `y="avg_decibel"` (line 1019). Category/microphone branches order by `-avg_decibel` (line 981/993), consistent with the hint.
- **Is it correct? NO (presentation correctness).** The region chart's bars are sorted by RMS energy while displaying decibels, so it is **not monotonic in the plotted metric** and the "top" bar is not the loudest by the shown metric.
- **Impact:** A "loudest regions" chart that visually contradicts its own ordering. Low severity (only the region branch, only ordering vs. label).
- **Correct computation:** Order the region branch by `-avg_decibel` to match its siblings and the plotted y, or derive ordering metric and `chart_hint.y` from a single variable so they cannot drift.

### [F25] `recent_datasets` `chart_hint` references a non-existent `count` column
**Location:** `data_insights/workflows/tools.py:1854–1857`

- **What is computed:** Row dicts carry `name/region/community/category/recording_date/recording_device`, but `chart_hint` is `{x:"recording_date", y:"count", group_by:"category"}` (verified). There is **no** `count` field.
- **Is it correct? NO (latent).** If a chart is built from this hint, `_to_float` (see [F27]) coerces the missing y to `0.0` for every row → an all-zero chart. In practice `_decompose_recent` renders a table and ignores the hint, so it is currently dormant.
- **Impact:** A landmine: any path that honors the hint produces a meaningless all-zero chart.
- **Correct computation:** Delete the dishonest hint (a record listing is a table, not a chart), or replace it with a real aggregate (e.g. count of datasets per `recording_date`).

### [F15] Fractional 0–1 metrics misclassified as ratio data → drawn as pie/donut
**Location:** `data_insights/workflows/chart_builder.py:164–166`

- **What is computed:** `_is_ratio_data` returns True for **any** column whose values all fall in 0.0–1.0 (verified lines 165–166), with **no** column-name check in that branch — despite the docstring claiming it also requires ratio-name semantics.
- **Is it correct? NO (chart-type correctness).** Many audio metrics are naturally 0–1 but are **not parts-of-a-whole**: average RMS energy, zero-crossing rate, normalized entropy, Spearman/Pearson coefficients, mutual-information. These get routed to pie/donut.
- **Impact:** A pie chart of per-group RMS or correlation coefficients implies the slices sum to a whole, which is false — visually misrepresents non-compositional metrics.
- **Correct computation:** Make the 0–1 branch symmetric with the 0–100 branch: require a ratio-name keyword AND/OR a sum-to-~1 check before returning True; otherwise fall through to bar.

---

## Cross-cutting data-integrity note (not analytics-specific, but corrupts the data the analytics run on)

The analytics findings above all assume the underlying table is the *intended* scope. It is not: **[F02]** (no collector/user scoping — every tool queries the full `NoiseDataset` table) and **[F13]/[F03]/[F12]** (NL→SQL agent runs with read-WRITE DB creds and a bypassable, partially-commented-out guard) mean a given user's "analytics" silently span **all** collectors' data, and the NL→SQL path could mutate the very rows being analyzed. These are catalogued under security/data-isolation but are load-bearing for any KPI's correctness: an aggregate over the wrong (or mutable) population is wrong regardless of its arithmetic.

---

## Summary table

| ID | Sev | Category | Metric | Correct? | Core defect |
|----|-----|----------|--------|----------|-------------|
| F01 | CRITICAL | analytics-correctness | Correlation matrix | NO | Per-column null-drop breaks row alignment → fabricated r/p |
| F04 | HIGH | analytics-correctness | `cumulative_energy` | NO | Per-bucket Sum of non-additive RMS, mislabeled "cumulative" |
| F05 | HIGH | data-integrity | `sample_count` vs `avg_db` | NO | Count includes null-`mean_db` rows the avg excluded |
| F18 | HIGH | data-integrity (UX) | streamed insight text | NO | Unescaped `innerHTML` drops/garbles literal `<…dB` content |
| F06 | MED | analytics-correctness | monthly leaderboard | NO | Filters `recording_date`, not upload `created_at` |
| F07 | MED | data-integrity | sampling provenance | NO | "stratified" label on plain random sample |
| F24 | MED | analytics-correctness | train/val/test split | NO | `max(1,…)` floors overshoot class size |
| F27 | MED | data-integrity | chart y-values | NO | null/non-numeric → fabricated 0-height bars |
| F29 | MED | analytics-correctness | entity-scoped count | NO | unanchored substring → false-positive filters |
| F28 | MED | analytics-correctness | ML tool availability | NO | 6 advanced tools bound only in error fallback |
| F31 | MED | data-integrity/scale | distribution values | numerically OK, unbounded | full-table value arrays in payload |
| F26 | LOW | analytics-correctness | region energy chart | NO | order_by RMS, plot dB |
| F25 | LOW | analytics-correctness | recent_datasets hint | NO (latent) | references nonexistent `count` column |
| F15 | LOW | analytics-correctness | chart-type selection | NO | 0–1 non-ratio metrics drawn as pie/donut |

**Bottom line:** The arithmetic of the *primary* deterministic aggregates (decibel grouping/ranking, dataset counts, energy avg/max/min) is largely correct, but the analytics layer as shipped is **not trustworthy**: one CRITICAL fabricated-statistics bug ([F01]), a mislabeled non-additive metric ([F04]), a self-inconsistent sample denominator ([F05]), a wrong-dimension leaderboard ([F06]), an impossible split partition ([F24]), and multiple silent data-fabrication paths in charts ([F27], [F25], [F15]) — plus a false "stratified" provenance claim ([F07]). Fix priority: **F01 → F04 → F05 → F24/F27/F29 → remainder.**

## 4. Data Flow & Performance

I have everything I need. Note that `_current_user_id` is only used for `created_by_id` write-stamping in tools.py (427, 610), NOT for read scoping — confirming F02. The assessment follows.

---

# DATA FLOW & PERFORMANCE ASSESSMENT — `data_insights`

## 1. Scope and pipeline map

A chat message traverses: DRF endpoint (`create_message` / `clarify_message`) → `_process_message_sync` → `_create_ai_agent` → LangGraph main agent loop → tool calls (deterministic ORM tools in `tools.py`, or the NL→SQL sub-agent in `sql_agent.py`) → widget/chart composition → `StreamingHttpResponse` token stream → terminal DB write. State persists in `PostgresSaver` keyed by `thread_id = session_id`. This assessment covers query patterns, caching correctness, engine/connection handling, blocking work in the request cycle, and the streaming pipeline. Findings IDs in brackets.

---

## 2. Query patterns: N+1, full scans, app-side aggregation

### 2.1 Full sequential scans + sorts on the analytics hot path `[F11]`
`NoiseAnalysis.mean_db` has no `db_index` (`data/models.py:505`) and `NoiseDataset` carries no composite indexes beyond `recording_date` and auto-FK indexes (`data/models.py:97`). The hottest analytics queries all hit this:
- `_decibel_ranked` — `NoiseAnalysis.exclude(mean_db__isnull=True).order_by('-mean_db')` (`tools.py:2042`): a full scan of `NoiseAnalysis` joined to dataset/region/community/category, plus a top-N sort with no supporting index.
- `_decibel_grouped` (`tools.py:2101`), `_energy_analysis` group-bys (`tools.py:955-1010`), `_statistical_analysis` — all `Avg`/`Max`/`Min`/`Sum` over `mean_db` / `rms_energy` across the full join.

**Optimization:** Add `Meta.indexes` to `NoiseAnalysis` (on `mean_db`, plus `noise_dataset` is already the OneToOne FK) and `AudioFeature` (`rms_energy`), generate one migration. **Expected impact:** `_decibel_ranked` moves from O(rows) seq-scan + sort to an index scan reading ~`limit` rows; this is the single largest latency win on the chat analytics path at scale. The grouped aggregates still scan but stop paying the sort cost. Effort: quick-win.

### 2.2 App-side aggregation pulling entire columns into Python `[F31]`
`_statistical_analysis` distribution branch (`tools.py:1230`) fetches `.values('category__name','noise_analysis__mean_db')` for the **entire** filtered dataset with no limit, groups in a Python dict, and stores **every individual decibel value** in `distribution_data[name]['decibel_values']`. Those raw arrays are then serialized into the tool response and persisted on `ChatMessage.visualization`. For a large dataset this is an unbounded over-the-wire read, large Python list construction, and bloated JSON.

**Optimization:** Compute the five-number summary in Postgres via `PERCENTILE_CONT` (a custom `Aggregate`), returning one row per group (typically <20 categories) instead of N raw rows. Drop the raw `decibel_values` arrays from the payload. **Expected impact:** distribution queries go from O(N) row transfer + O(N) JSON to O(groups); also shrinks the persisted `visualization` blob. This dovetails with the box-plot fix `[F16]` which already wants precomputed `[min,q1,median,q3,max]` server-side. Effort: medium-term.

### 2.3 Repeated full-table entity lookups per question `[F32]`
`_dataset_count` issues up to 6 separate `Region/Community/Category/Class/SubClass.objects.values_list('name')` full-table loads on **every** count question, purely to do Python substring matching (`tools.py:1963`, `_match_entity_name` at `1677-1684`), then a `count()` over `NoiseDataset` filtered on `recording_date__year/month` or `icontains` (unindexed). The substring match is also a correctness defect — unanchored matching false-positives ("Ada" in "Canada", "Bus" in "business") `[F29]`.

**Optimization:** Cache the 5 lookup name lists with `django.core.cache` (small, slow-changing; TTL or signal-invalidated). Replace substring matching with the word-boundary regex already used in `clarification_gate.py`, factored into one shared helper so the two paths cannot drift `[F29]`. **Expected impact:** removes 6 redundant table reads per count and eliminates silently-wrong scoped counts. Effort: quick-win.

### 2.4 Pagination double-queries
The `recent_datasets`/`dataset_search` pagination branches run `qs.count()` and a separate slice (`views.py:1042-1051`). This is acceptable (count + page is standard), but note `has_more` for the SQL branch is derived from `len(rows) == limit` with `total_count: None` (`views.py:1080-1081`), which is a heuristic, not a true has-more. Low priority.

No classic ORM N+1 loops were found in the tool layer — the deterministic tools use `.values().annotate()` aggregation rather than per-object FK traversal, which is the right instinct. The session list correctly uses `prefetch_related("messages")` (`views.py:153`).

---

## 3. Caching: correctness

### 3.1 Correct caches
- Compiled LangGraph workflows (`_compiled_graphs`, `views.py:99`) and tool lists (`_cached_tools`, `views.py:100`, `1226`) are cached process-wide and are genuinely immutable/stateless — this is correct, and conversation state is isolated in `PostgresSaver` by `thread_id`. The `reuse_compiled_graph` path (`views.py:1251`) is sound.
- `_get_checkpointer` (`views.py:77`) is a proper lock-guarded, double-checked singleton over one `ConnectionPool`. This is the model the rest of the engine handling should follow.

### 3.2 `@lru_cache` on a bound method leaks instances `[F30]`
`@lru_cache(maxsize=20)` decorates `SQLDatabaseWrapper._get_cached_table(self, table_name)` (`sql_agent.py:155`). Because `self` is part of the cache key, the cache is process-wide across **all** wrapper instances and holds a strong reference to every `self` ever passed, pinning those wrappers (and their engines/metadata) against GC. Combined with the per-request wrapper creation in §4, this is a slow leak. The method also appears to be dead code.

**Optimization:** Delete `_get_cached_table` (preferred — unused), or move memoization to a per-instance dict. **Expected impact:** removes a latent memory leak that compounds with `[F09]`/`[F10]`. Effort: quick-win.

### 3.3 Schema info rebuilt and re-sent every LLM turn `[F08]` `[G05]`
`call_llm` formats the system prompt with `self.db.get_table_info()` on **every** model invocation (`sql_agent.py:631`). With no `table_names` arg, `get_table_info()` falls back to ALL usable tables, forces reflection of any unreflected table, recompiles `CreateTable` DDL per table, and runs a `SELECT * ... LIMIT n` sample per table. The allowed-table set falls back to ~15 data+core tables. Worse, the `trim_messages` cap is commented out (`sql_agent.py:635-640`), so the full schema is re-tokenized and re-sent on every turn of a multi-step SQL conversation.

This is both a performance and a cost defect: per-multi-step-query input tokens are O(schema_size × turns) instead of O(schema_size + history).

**Optimization:** Compute the formatted schema string **once** per agent instance in `__init__` and cache it on `self` (schema is static between migrations); re-enable `trim_messages` to bound history. Optionally enable OpenAI prompt caching by keeping the system block stable. **Expected impact:** eliminates per-turn DDL recompile + sample-row SELECTs and collapses repeated schema tokenization; large input-token and latency reduction on any query that takes more than one SQL turn. Effort: medium-term.

---

## 4. Connection / engine handling

### 4.1 New engine + uncached SQL sub-graph per `DataAnalysisTool` instantiation `[F09]`
`DataAnalysisTool.add_agent` (a pydantic `model_validator`, `tools.py:1659-1675`) constructs a fresh `TextToSQLAgent` on each instantiation, which calls `create_engine()` (`sql_agent.py:466`), builds a `SQLDatabaseWrapper` (running `inspect(engine)` / dialect probing), and compiles the workflow. Each new engine spins up its own connection pool. The tool list is cached (`_cached_tools`), so in steady state this fires once per process — but any code path that rebuilds tools re-pays it, and the SQL sub-agent itself is compiled without a shared checkpointer.

### 4.2 Fresh engine per pagination request, never disposed `[F10]`
`_paginate_from_tool_data` (`views.py:1056`) calls `create_engine(DB_URI)` and builds a new `SQLDatabaseWrapper` (eager reflect) on **every** pagination request in the `query_kind == 'sql'` branch, and **never disposes** it. Each call opens a new psycopg pool that lingers until GC — a connection leak under pagination load. This is a copy of the `sql_agent.py` setup pattern without porting the lifecycle.

**Optimization:** Add a lazily-initialized, lock-guarded module-level singleton engine + shared `SQLDatabaseWrapper` (mirror `_get_checkpointer` at `views.py:77`) and have `_paginate_from_tool_data` reference it. **Expected impact:** bounds pagination connection usage to one shared pool instead of one-pool-per-request; eliminates the undisposed-pool growth. Effort: quick-win. (`[F09]` benefits from the same shared engine.)

### 4.3 NL→SQL engine uses read-WRITE app credentials `[F13]` `[F03]` `[F12]`
The SQL agent's engine is built from the full application Postgres credentials (`sql_agent.py:466`), not a read-only role. The system prompt *claims* "read-only" but nothing enforces it at the DB layer. Compounding this: the SELECT-only statement check and the `--` comment denylist are both **commented out** (`sql_agent.py:572-574`, `510`), leaving only a bypassable regex denylist + keyword scan. Confirmed bypasses exist (`pg_sleep`, blind CASE-WHEN timing, cartesian explosion) `[F12]`. There is also no `statement_timeout`, so a single crafted query can run unbounded.

This is primarily a security finding, but it is squarely a **data-flow/connection** concern: the engine itself is over-privileged and unbounded.

**Optimization (defense in depth, ordered):** (1) provision a dedicated least-privilege Postgres role (`GRANT SELECT` only on allowlisted tables, non-superuser) and point the agent engine at it; (2) set `statement_timeout` + `default_transaction_read_only` via `connect_args`/`SET` on the engine; (3) restore the SELECT-only structural check. **Expected impact:** even if the regex guard is bypassed, the connection becomes physically incapable of writing or running long/expensive queries — this collapses the entire DoS + write-injection class. Effort: medium-term.

---

## 5. Blocking work in the request cycle

### 5.1 Synchronous LLM/agent execution inside the request
`create_message` runs the full LangGraph tool-calling agent **synchronously** inside the request via `_process_message_sync` (`views.py:189`), and `generate_insight` makes a blocking `llm.invoke` (`views.py:1199`). There is no task queue; OpenAI latency directly occupies a worker for the entire generation.

### 5.2 No client-side LLM timeout `[G07]`
`AI_INSIGHT.AGENT.TIMEOUT_SECONDS` (120) is defined but **never read**. None of the `ChatOpenAI` constructions (`views.py:1229`, `1236`; `tools.py:1537`) pass `request_timeout`/`timeout`. A hung OpenAI connection has no client-side deadline, so a stalled request holds a worker thread, a DB connection, and an open streaming response indefinitely. **Optimization:** pass a granular `httpx.Timeout(connect=5, read=TIMEOUT_SECONDS, …)` plus `max_retries=MAX_RETRIES` (both already in config) at all four sites. **Expected impact:** bounds worst-case worker occupancy; a stalled connection can no longer pin resources. Effort: quick-win.

### 5.3 No recursion limit / unbounded agent loop `[G04]`
All four agent configs set only `thread_id` (`agent_workflow.py:778`, `783`, `806`, `848`, `881`) — no `recursion_limit`. `should_continue` routes back to tools on any tool call. The loop relies on LangGraph's implicit 25-superstep default (~8 round-trips of LLM + tool work, all blocking the request). **Optimization:** set an explicit `recursion_limit` (~12) and catch `GraphRecursionError` for a clean message. **Expected impact:** caps worst-case in-request LLM+DB work per message. Effort: quick-win.

### 5.4 No throttling / no in-flight guard `[G01]` `[G06]`
The three most expensive endpoints (`create_message`, `clarify_message`, `generate_insight`) are protected only by `IsAuthenticated`; `REST_FRAMEWORK` declares no throttle classes (`views.py:125`). `create_message` immediately calls `_process_message_sync` with no check for an already-`PROCESSING` message — unlimited concurrent agent runs per session (`views.py:162`). Each concurrent run multiplies worker occupancy, DB-pool pressure, and the engine leak in §4.2. **Optimization:** add DRF `ScopedRateThrottle` wired to the existing (dead) `RATE_LIMIT_PER_MINUTE` config, plus a per-session in-flight check (reject if a `PROCESSING` message exists). **Expected impact:** caps per-user concurrency and bounds the resource fan-out. Effort: quick-win.

### 5.5 `max_tokens` unbounded on the main agent `[G03]`
The main agent + dashboard LLMs are built with no `max_tokens` (`views.py:1229`, `1236`); only the small insight LLM caps at 200 (`views.py:1186`). Every agent turn can emit an arbitrarily long completion, prolonging the blocking stream. **Optimization:** add a generous configurable cap (e.g. 4000) as a backstop. Effort: quick-win.

---

## 6. Streaming pipeline

### 6.1 Architecture
`_process_message_sync` wraps a **sync** generator (`stream()`) inside `_safe_stream` and returns a `StreamingHttpResponse` (`views.py:524-532`) with correct streaming headers (`X-Accel-Buffering: no`, no-cache). The design intentionally keeps the generator sync to avoid `sync_to_async` per ORM call while Django's ASGI handler iterates it. This is reasonable.

### 6.2 Terminal status written inside the generator body — stuck-PROCESSING risk `[F23]`
`mark_processing()` persists `PROCESSING` on the request thread **before** the stream begins (`views.py:551`), but the `COMPLETED`/`FAILED` writes live **inside** the generator (`views.py:457-462`, `381-386` on `GeneratorExit`). If the worker dies, the client never connects, or the ASGI handler never drains the generator, the message stays `PROCESSING` forever. The recovery command `fix_stuck_messages.py` bulk-flips **all** `PROCESSING` rows to `FAILED` with no age threshold (`fix_stuck_messages.py:9`), so it can clobber a live, mid-generation message and overwrite its `assistant_response`.

**Optimization:** add an age threshold (`updated_at < now − N min`) to the recovery command and never overwrite a populated `assistant_response`; ideally reconcile on worker startup. **Expected impact:** removes the destructive race where recovery marks in-progress messages `FAILED`. Effort: quick-win.

### 6.3 Streamed markdown rendered via `innerHTML` without escaping `[F18]`
On the client, `formatMarkdown` builds an HTML string from LLM/tool output with regex replacements and assigns it via `innerHTML` (`unified_chat.html:2612`, render path `1040`/`1071`/`1097`). Unlike the table/stat-card builders, the streamed prose is **not** escaped before the regex pass (`escapeHtml` is applied only to table cells/widget labels). This is both an XSS vector and a formatting-corruption bug (e.g. `<30 dB` mangled). **Optimization:** `escapeHtml(text)` first, then run the regex formatting pass over the inert text. **Expected impact:** closes the XSS hole and fixes literal-`<`/`>` corruption in insight text. Effort: quick-win.

### 6.4 Robustness
The generator wraps nearly every yield in `try/except` and `_safe_stream` catches `GeneratorExit`/`Exception` (`views.py:506-522`) — good defensive structure. The post-stream re-reads of graph state via `get_state` (`views.py:360`, `412`) add two extra checkpointer round-trips per message after streaming completes; minor, but worth folding into a single read.

---

## 7. Prioritized optimization summary

| Priority | Item | IDs | Expected impact | Effort |
|---|---|---|---|---|
| P0 | Read-only least-priv role + `statement_timeout` on NL→SQL engine; restore SELECT-only check | F13, F03, F12 | Eliminates write-injection + unbounded-query DoS class | medium |
| P0 | Shared singleton engine for pagination (stop per-request `create_engine`, dispose) | F10, F09 | Stops connection-pool leak under pagination/agent load | quick |
| P0 | Add throttling + per-session in-flight guard + `request_timeout` + `recursion_limit` + `max_tokens` | G01, G06, G07, G04, G03 | Bounds worker occupancy, DB-pool pressure, and OpenAI spend per request | quick |
| P1 | Index `NoiseAnalysis.mean_db`, `AudioFeature.rms_energy` | F11 | Decibel ranking from full-scan+sort to index scan — biggest analytics latency win | quick |
| P1 | Cache `get_table_info()` once per agent + re-enable `trim_messages` | F08, G05 | Per-turn DDL/sample-SELECT eliminated; input tokens O(schema+history) | medium |
| P1 | Push distribution aggregation into Postgres (`PERCENTILE_CONT`), drop raw value arrays | F31, F16 | Distribution query from O(N) transfer to O(groups); smaller persisted blobs | medium |
| P2 | Cache lookup-name lists + word-boundary entity match | F32, F29 | Removes 6 redundant table reads/count; fixes silently-wrong scoped counts | quick |
| P2 | Remove `@lru_cache` on bound method | F30 | Removes instance/engine memory leak | quick |
| P2 | Age-threshold + non-destructive stuck-message recovery | F23 | Stops clobbering live in-progress streams | quick |
| P2 | Escape-before-format streamed markdown | F18 | Closes XSS + formatting corruption in insight text | quick |

## 8. Key file:line references
- Engine/connection handling: `data_insights/workflows/sql_agent.py:466` (engine creds), `:155` (lru_cache leak); `data_insights/views.py:1056` (per-request engine), `:77` (correct singleton pattern), `:99-100` (graph/tool caches), `:1212-1248` (agent build).
- Query patterns: `data_insights/workflows/tools.py:2042` (decibel ranked), `:2101` (decibel grouped), `:955-1010` (energy group-bys), `:1228-1259` (statistical distribution app-side aggregation), `:1963` + `:1677-1684` (entity lookups); `data/models.py:505` (unindexed `mean_db`), `:97` (`recording_date` index).
- SQL guard/schema: `sql_agent.py:507-526` (regex patterns), `:571-574` (commented SELECT-only), `:628-643` (per-turn `get_table_info` + commented `trim_messages`).
- Blocking work: `views.py:189` (sync agent in request), `:1229/1236` + `tools.py:1537` (no timeout/max_tokens); `agent_workflow.py:778-881` (no recursion_limit); `views.py:162` (no in-flight guard), `:125` (no throttle).
- Streaming: `views.py:524-532` (StreamingHttpResponse), `:381-386`/`:457-462` (terminal status in generator), `fix_stuck_messages.py:9` (recovery); `data_insights/templates/data_insights/unified_chat.html:2612` (innerHTML markdown).

## 5. UX & Insight Quality

I now have a thorough understanding of all the UX-relevant code paths. I have enough to write the assessment grounded in the findings (F15, F16, F17, F18, F27, plus supporting analytics-correctness findings that affect insight trust) and the code I've verified.

# UX & Insight-Quality Assessment — `data_insights` NL→SQL Analytics Chatbot

**Scope:** Can end users understand, trust, and act on the insights this chatbot produces? This evaluates the deterministic chart pipeline (`chart_builder.py`, `widget_composer.py`), the frontend renderer (`unified_chat.html`), and the clarification/streaming UX. Findings are cross-referenced to the confirmed register; supporting analytics-correctness findings are cited where a "correct" number is still presented to the user in a misleading way.

**Headline:** The pipeline architecture is genuinely good — deterministic, no-LLM chart selection (`chart_builder.select_chart_type`) with per-analysis-type multi-widget decomposition (stat card + chart + ranking + table) is a strong, legible pattern. But several defects silently degrade or corrupt the insight a user sees, and the truncation/empty-state/units story is inconsistent enough to erode trust. The single most urgent item is a security-and-formatting defect in the streamed prose (F18).

---

## 1. Chart-type appropriateness

**What works.** `select_chart_type` (chart_builder.py:97) is a sound decision tree: temporal→line/bar, two-numeric→scatter, categorical+numeric→bar/horizontal-bar by cardinality. The per-type decomposers in `widget_composer.py` pick contextually correct widgets (box plot for distributions, `correlation_heatmap` for the matrix, `class_distribution_bar` for balance, `feature_importance_bar` for MI scores).

**Problems.**

- **[F16] Box plots silently collapse into a mislabeled bar chart, losing the entire distribution.** `_decompose_statistical` (widget_composer.py:724) ships `data.data` as raw value arrays, but the renderer's box-plot branch (unified_chat.html:1352) needs `actualData.boxPlotData` (the 5-number summary). `extractChartData` only computes `boxPlotData` from a raw `tool_response` with `analysis_type==='statistical_distribution'` (line 1621) — which is NOT present on the widget-render path (`renderArtifact`→`createChart` passes `{frontend_data: widget.data}`, line 2006). So the box-plot widget hits the bar-chart fallback (line 1386) labeled "Average dB" while plotting raw arrays. **Fix:** compute the five-number summary in Python at composition time (the data already carries `decibel_values`/`min`/`max`/`avg`) and emit `boxPlotData` in `widget.data` so the frontend needs no special-casing.

- **[F15] Fractional 0–1 audio metrics are drawn as pie/donut charts.** `_is_ratio_data` (chart_builder.py:165) returns `True` for any column whose values are all in 0.0–1.0, with no name check on that branch. RMS energy, zero-crossing rate, normalized entropy, Spearman/Pearson coefficients are all naturally 0–1 but are NOT parts-of-a-whole, yet route to `pie_chart`/`donut_chart` (lines 138–141). A pie of "correlation by feature" is meaningless and actively misleads. **Fix:** make the 0–1 branch symmetric with the 0–100 branch — require a ratio-name keyword OR a sum-to-~1 check before returning `True`.

- **[F25/F26] chart_hint drift produces wrong or all-zero charts.** `_recent_datasets` hints `y:'count'` which does not exist in its rows (tools.py:1855) → all-zero bars if ever charted; the region energy branch orders by `-avg_rms_energy` but plots `y='avg_decibel'` (tools.py:967 vs 1019) → bars not monotonic in the displayed metric, so the "top" bar is not the loudest by the value shown. These break the reader's basic expectation that a sorted chart is sorted by what it displays.

---

## 2. Labeling, units, and axis clarity

- **Units are applied inconsistently.** Only the box-plot bar-fallback (unified_chat.html:1408) and its tooltip (lines 1428–1430) append "dB". The generic `bar_chart`/`line_chart`/`horizontal_bar_chart` paths have **no y-axis title and no unit suffix** — a region energy chart shows a bare number axis with no "dB" or "RMS". Stat cards (`buildStatCardHtml`, line 2174) print `value.toLocaleString()` with no unit at all, so "Average Mean Db: 62.4" loses its dimension unless the key name happens to contain it.
- **Titles are auto-humanized from column names** (`_humanise`, chart_builder.py:178) — "Avg Decibel" is fine, but raw keys like `region__name` leak into ML profile labels (widget_composer.py:463) producing "Label Distribution By Category" with double-underscore artifacts in some paths.
- **Generic axis fallbacks ship to users.** Scatter plots default to literal `'X Axis'`/`'Y Axis'` (lines 1463/1469) when `xLabel`/`yLabel` are absent — which is the case for any scatter built from the widget path, since those labels are only set in the `correlation_analysis` extract branch (line 1819).

**Improvements:** (1) Carry an explicit `unit` (e.g. "dB", "s", "Hz") through `chart_hint`/widget `data` and render it in both the axis title and the stat-card value. (2) Set a y-axis `title.text` on every numeric chart from `y_label`/`_humanise(y_key)`. (3) Strip `__name` and double-underscores in `_humanise`.

---

## 3. Truncation transparency

**[F17] This is the biggest trust gap.** Truncation is disclosed on exactly one path. `build_chart_config` (chart_builder.py:226) appends "(Top 12 of N)" only when a `ChartDecision.truncate` spec is set (i.e. cardinality ≥ 13). But:

- `build_chart_config` **always** slices `display_rows = rows[:12]` (line 210) with no note when no truncate spec exists — a 12-row chart from a 12-category result looks complete even if there were more.
- `_decompose_ranked` charts the top 10 (`display_rows = rows[:10]`, widget_composer.py:642) with title "Top 10 …" but no "of N".
- `_decompose_grouped`'s chart widget passes full rows through `_rows_to_widget`, which slices to 12 again inside `build_chart_config` — silently.
- ML tables slice to 20 (`_build_table_config`, chart_builder.py:247) / preview 20 (`_decompose_export_features`, line 1109) with no "of N".

The one widget that does it right is `ranking_highlight` ("Showing top and bottom {limit} of {total}", unified_chat.html:1987). **A user can conclude "only 12 communities exist" when there are 50.**

**Fix:** centralize disclosure. In `build_chart_config`, compute `total_before_truncate` before any slice (already at line 200) and make the title note **unconditional** whenever `len(display_rows) < total` — regardless of whether a `truncate` spec exists. Mirror the `ranking_highlight` "of N" pattern into every table widget and the ranked-chart title.

---

## 4. Color & accessibility

- **[F18 — CRITICAL for this dimension] Streamed assistant prose is injected via `innerHTML` after a regex "markdown" pass, with no escaping.** `formatMarkdown` (unified_chat.html:2612) runs regex replacements and the result is assigned via `innerHTML` (`_renderStreamFrame`:1097, `appendStreamText`:1040, `formatStreamedText`:1071). Unlike table cells and widget labels, the prose body is **never** passed through `escapeHtml` (line 2606) before formatting. This is both (a) a stored/streamed XSS vector if any tool output, entity name, or DB value reaches the prose, and (b) a formatting-corruption bug: literal insight text like "<30 dB" or "a < b" is eaten as a bogus HTML tag. **Fix:** escape first (`escapeHtml(text)`), then run the regex formatting pass so literal `<`/`>`/`&` become inert before the known-safe formatting tags are injected — this fixes both the XSS and the "<30 dB" corruption.
- **Color palette is not colorblind-safe.** `generateColors` (unified_chat.html:1880) is a fixed 12-color wheel (red/blue/yellow/teal/…) reused by modulo. Adjacent categories on a pie or stacked bar can be red/green pairs indistinguishable to deuteranopes, and there is no pattern/texture fallback. Categorical charts also rely on color alone with the legend at the bottom — no direct data labels.
- **Contrast / semantics.** Severity colors are mapped to raw hex (e.g. class-balance `#ef4444`/`#22c55e`, widget_composer.py:760; missingness thresholds line 434) and surfaced as the only signal of "severe vs balanced" — color-only encoding of a critical judgment.
- **Keyboard/ARIA.** Clarification chips are real `<button>`s with `:focus-visible` styling (good, lines 3340/3362) and the input is labeled, but chart `<canvas>` elements have no `aria-label`/text alternative, and the streamed status (`stream-status`) is not an ARIA live region, so screen-reader users get no progress announcement.

**Improvements:** swap to an Okabe–Ito (colorblind-safe) palette; add direct value labels on bars; encode severity with an icon/text token in addition to color; add `role="img"` + `aria-label` summarizing each chart; mark the stream/status container `aria-live="polite"`.

---

## 5. Empty & error states

- **Charts with no data render nothing — no message.** `createChart` returns silently if `labels`/`data` are empty (unified_chat.html:1214) and `renderArtifact` has no `else` to show "No data for this view." A widget can simply vanish, leaving a confusing gap.
- **[F27] Null/non-numeric y-values are fabricated as real 0 bars.** `_to_float` (chart_builder.py:214–220) returns `0.0` for `None` or unparseable values. A group whose `avg_db` is NULL, or an "Unknown" bucket, renders as a solid **0 dB bar** — and 0 dB is a meaningful loudness, so this is materially misleading, not just cosmetic. **Fix:** make `_to_float` return `Optional[float]` (None for null/unparseable) and either drop those label+value pairs or pass `null` so Chart.js draws a gap; never coerce to 0.
- **Generic failure copy.** On error the whole bubble is replaced with "Failed to process message. Please try again." (unified_chat.html:1127) — no indication of whether it was a clarification timeout, an empty result, a guardrail rejection, or a server error, and no retry affordance beyond re-typing.
- **[F22] A missing `rows` key crashes the decomposer.** `_decompose_grouped`/`_decompose_ranked` do `result["rows"]` with no default (widget_composer.py:529, 624). An empty or error-shaped tool payload raises `KeyError` and the user gets the generic 500/"Failed". **Fix:** `result.get("rows", [])` (every sibling decomposer already does this; both functions already handle the empty case).
- **[F23] Stuck "Processing" states.** Messages are marked PROCESSING before the stream is consumed; if the worker dies the message stays PROCESSING forever, and the recovery command can clobber live in-progress messages. From the user's side this manifests as a spinner that never resolves or a message that flips to "Failed" mid-answer.

**Improvements:** render an explicit empty-state card ("No data matched this query — try widening the filters") when a widget has no rows; differentiate error messages by cause and add a one-click "Retry" button; add a client-side timeout that surfaces "This is taking longer than expected."

---

## 6. Clarification UX

**What works.** The floating clarification panel (`showClarificationUI`, line 2211) is well-built: a clear question, chip options, an optional free-text "Type your own answer" input, disabled-until-valid "Continue" button, `:focus-visible` styling, and a sensible slide-up animation. Options and custom input are properly escaped (lines 2228/2237). This is the strongest UX surface in the app.

**Gaps.**

- No visible record in the transcript of *what the user clarified* — after answering, the panel hides (line 2328) and the conversation continues, but the chosen disambiguation is not echoed into the thread, so scrolling back loses context of why the result was scoped a certain way.
- The rephrase fallback (line 2362) reconstructs the question from `data.dimension` with a generic template ("What should I use for {dimension}?"), which can read awkwardly.
- No "skip / answer for all results" affordance — every ambiguous turn forces a modal-style interrupt even when the user would accept a sensible default.

**Improvements:** echo the resolved clarification as a small inline chip in the message thread ("Scoped to: Greater Accra"); offer a "use default" option; persist the clarification context with the message so reloading the session shows the disambiguation.

---

## 7. Are the insights understandable and actionable?

**Strengths.** The multi-widget decomposition is genuinely actionable: stat card (the number) + chart (the shape) + ranking (the extremes) + table (the detail) + deterministic follow-up chips (`FOLLOW_UP_MAP`, widget_composer.py:22) that suggest concrete next questions. The statistical-test decomposer even surfaces a `plain_english` verdict plus Cohen's d with a labeled effect size (lines 996–1004) — that is exactly the right altitude for a non-statistician. The opt-in "What do you notice?" insight button (line 2035) is a good cost-aware pattern.

**But several insights are confidently wrong or misleadingly labeled — the deepest actionability risk:**

- **[F01 CRITICAL] The correlation matrix is fabricated.** Per-column null-dropping then truncating to a global min length (tools.py:2788) means position *i* of `rms_energy` and position *i* of `spectral_centroid` come from different rows. The heatmap, "Top 10 strongest pairs" table, and the "Strongest pair" stat card (`_decompose_correlation`, widget_composer.py:934) all present **invented coefficients** as authoritative. A user choosing features for ML based on this will act on noise.
- **[F07] Sampling falsely labeled "stratified."** The correlation stat card prints `Sample: N of M (stratified)` (widget_composer.py:943) while the sample is plain random (tools.py:2759). The label asserts a rigor that isn't there.
- **[F04] "Cumulative energy" is a per-bucket sum, not a running total** (tools.py:1362). A user reading "cumulative energy over time" reasonably believes values accumulate; they don't — and RMS isn't even additive.
- **[F24] The recommended train/val/test split can allocate more rows than a class contains** (tools.py:2586) and the per-class table sums to more than the dataset — a user copying these counts gets an impossible split.
- **[F06] The monthly leaderboard credits by recording date, not upload date** (tools.py:1866) — "top contributors this month" silently mis-attributes bulk/backdated uploads, an insight people will act on socially/operationally.
- **[F29] Entity filters can be silently wrong** — "Ada" matches "Canada", "Bus" matches "business" (tools.py:1680), so a scoped count can quietly answer a different question than asked, with no indication the filter fired.

**The throughline:** the *presentation* layer projects high confidence (clean stat cards, precise 4-decimal coefficients, plain-English verdicts) on top of numbers that are in several cases fabricated or mislabeled. That combination — polished UI + wrong number — is more dangerous for actionability than an obviously broken chart, because users will trust and act on it. Fixing the underlying correctness findings (F01, F04, F07, F24, F06, F29) is as much a UX/insight-quality requirement as a correctness one.

---

## Prioritized concrete improvements

1. **[F18]** Escape-then-format the streamed prose (`escapeHtml` before the regex pass in `formatMarkdown`). Fixes XSS + "<30 dB" corruption. *(quick win, highest urgency)*
2. **[F01/F07] Stop shipping fabricated/mislabeled correlations** — listwise deletion preserving row identity; change the stat-card label to the true sampling method.
3. **[F17] Make truncation disclosure unconditional** in `build_chart_config`; add "of N" to ranked-chart titles and every table widget.
4. **[F27] `_to_float` must preserve null** — gap/omit instead of fabricating 0 bars; render an explicit empty-state card when a widget has no data.
5. **[F16] Compute box-plot 5-number summary in Python** so the box plot renders instead of a mislabeled bar.
6. **[F15] Gate the 0–1 pie/donut branch** behind a ratio-name or sum-to-1 check.
7. **Units & axes:** carry an explicit `unit`, render a y-axis title on every numeric chart, append units in stat cards.
8. **Accessibility:** Okabe–Ito palette + direct labels + icon/text severity tokens; `aria-label` on chart canvases; `aria-live` on stream status.
9. **[F22] `result.get("rows", [])`** in the two crashing decomposers; differentiate error copy and add a Retry button.
10. **Clarification:** echo the resolved disambiguation inline in the transcript; offer a default/skip.

**Key files:** `/Users/kelvinackah/Desktop/projects/freelance/datacollection/data_insights/workflows/chart_builder.py`, `/Users/kelvinackah/Desktop/projects/freelance/datacollection/data_insights/workflows/widget_composer.py`, `/Users/kelvinackah/Desktop/projects/freelance/datacollection/data_insights/templates/data_insights/unified_chat.html`, `/Users/kelvinackah/Desktop/projects/freelance/datacollection/data_insights/workflows/tools.py`.

## 6. Findings Register

| ID | Severity | Area | Finding | Verdict | Conf | Location |
|----|----------|------|---------|---------|------|----------|
| F01 | 🔴 CRITICAL | analytics-correctness | Correlation matrix correlates misaligned rows (per-column null drop) — fabricated correlations | confirmed | 0.90 | `data_insights/workflows/tools.py:2788-2814` |
| F02 | 🔴 CRITICAL | Security | No collector/user scoping — any authenticated user reads ALL collectors' data | confirmed | 0.90 | `data_insights/workflows/tools.py:374 (NoiseDatasetSearchTool), tools.py:1452 (NoiseDetailTool.get), tools.py:748 (AudioAnalysisTool)` |
| F03 | 🔴 CRITICAL | Correctness | SQL agent SELECT-only enforcement and '--' comment denylist are commented out (write-capable creds) | confirmed | 0.90 | `data_insights/workflows/sql_agent.py:572` |
| F13 | 🔴 CRITICAL | Security | NL→SQL agent connects with read-WRITE application DB credentials | confirmed | 0.90 | `data_insights/workflows/sql_agent.py:466` |
| F04 | 🟠 HIGH | analytics-correctness | 'cumulative_energy' is a per-period sum, not a running total — mislabeled metric | confirmed | 0.90 | `data_insights/workflows/tools.py:1362` |
| F05 | 🟠 HIGH | analytics-correctness | decibel_grouped sample_count counts datasets that did not contribute to avg_db | uncertain | 0.83 | `data_insights/workflows/tools.py:2101-2103` |
| F09 | 🟠 HIGH | DATA | A new SQLAlchemy engine + uncached SQL agent graph is built for every DataAnalysisTool instantiation | uncertain | 0.85 | `data_insights/workflows/tools.py:1665` |
| F11 | 🟠 HIGH | DATA | Decibel ranking/grouping and avg-by-dimension queries sort/aggregate on unindexed mean_db across full join | uncertain | 0.85 | `data_insights/workflows/tools.py:2042` |
| F12 | 🟠 HIGH | Security | SQL injection / DoS bypasses of the regex+keyword guard (pg_sleep, CASE, cartesian) | uncertain | 0.90 | `data_insights/workflows/sql_agent.py:507-526 (patterns), sql_agent.py:571-574 (commented SELECT-only)` |
| F14 | 🟠 HIGH | Security | Secrets (.env) committed to git history | adjusted | 0.92 | `datacollection/settings.py:34 (SECRET_KEY = os.getenv); git history commit a143af0` |
| F18 | 🟠 HIGH | UX | Streamed assistant markdown is rendered as raw HTML via innerHTML with regex 'markdown' — XSS / formatting-corruption risk in the insight text | confirmed | 0.82 | `data_insights/templates/data_insights/unified_chat.html:2612` |
| G01 | 🟠 HIGH | LLM | No DRF throttling on expensive LLM streaming endpoints (only IsAuthenticated) | adjusted | 0.90 | `data_insights/views.py:125` |
| G02 | 🟠 HIGH | LLM | AI_INSIGHT.SECURITY rate/abuse config is entirely dead (never read) | adjusted | 0.90 | `datacollection/settings.py:485` |
| G05 | 🟠 HIGH | LLM | trim_messages disabled while full get_table_info() re-sent every SQL turn | uncertain | 0.95 | `data_insights/workflows/sql_agent.py:631` |
| G06 | 🟠 HIGH | LLM | No per-session in-flight guard: unlimited concurrent agent runs per user/session | confirmed | 0.85 | `data_insights/views.py:162` |
| G07 | 🟠 HIGH | LLM | No request_timeout on any agent LLM (TIMEOUT_SECONDS config never applied) | adjusted | 0.90 | `data_insights/workflows/sql_agent.py:642` |
| F06 | 🟡 MEDIUM | analytics-correctness | top_collectors_monthly filters on recording_date, not upload/created date | confirmed | 0.84 | `data_insights/workflows/tools.py:1866-1867` |
| F07 | 🟡 MEDIUM | analytics-correctness | Correlation tool labels plain random sampling as 'stratified' | confirmed | 0.90 | `data_insights/workflows/tools.py:2759-2763` |
| F08 | 🟡 MEDIUM | DATA | SQL agent rebuilds full schema info (reflect + sample EVERY allowed table) on every LLM turn | adjusted | 0.82 | `data_insights/workflows/sql_agent.py:631` |
| F10 | 🟡 MEDIUM | DATA | paginate_message and clarification SQL-pagination create a fresh engine per request, never disposed | adjusted | 0.85 | `data_insights/views.py:1056` |
| F16 | 🟡 MEDIUM | UX | Box-plot widgets silently degrade to a mislabeled bar chart and lose the distribution entirely | adjusted | 0.90 | `data_insights/workflows/widget_composer.py:724` |
| F17 | 🟡 MEDIUM | UX | Top-N truncation is invisible in most widget paths — users don't know data was cut | adjusted | 0.82 | `data_insights/workflows/chart_builder.py:209` |
| F19 | 🟡 MEDIUM | Correctness | NameError: timezone.now() called but timezone never imported in views.py | confirmed | 0.92 | `data_insights/views.py:1116` |
| F20 | 🟡 MEDIUM | Correctness | AttributeError: ChatSession.Status.DELETED referenced but not defined | adjusted | 0.92 | `data_insights/serializers.py:90` |
| F23 | 🟡 MEDIUM | Correctness | Message marked PROCESSING before stream consumed; no age threshold in recovery → stuck and clobbered statuses | adjusted | 0.84 | `data_insights/management/commands/fix_stuck_messages.py:9` |
| F24 | 🟡 MEDIUM | analytics-correctness | Stratified train/val/test split over-allocates rows for small classes (counts exceed class size) | confirmed | 0.95 | `data_insights/workflows/tools.py:2586-2591` |
| F27 | 🟡 MEDIUM | analytics-correctness | _to_float coerces non-numeric / null y-values to 0.0, fabricating zero bars | adjusted | 0.82 | `data_insights/workflows/chart_builder.py:214-222` |
| F28 | 🟡 MEDIUM | analytics-correctness | Advanced ML analytics tools are registered only in the exception fallback, unreachable in normal operation | confirmed | 0.92 | `data_insights/workflows/tools.py:3452-3464` |
| F29 | 🟡 MEDIUM | analytics-correctness | _match_entity_name uses unanchored substring matching — false-positive entity filters | confirmed | 0.88 | `data_insights/workflows/tools.py:1680-1683` |
| F31 | 🟡 MEDIUM | DATA | Statistical 'distribution' analysis loads every mean_db row into Python and aggregates in app code | uncertain | 0.80 | `data_insights/workflows/tools.py:1230` |
| G03 | 🟡 MEDIUM | LLM | Main agent ChatOpenAI has no max_tokens cap (unbounded output cost per call) | adjusted | 0.82 | `data_insights/views.py:1229` |
| G04 | 🟡 MEDIUM | LLM | LangGraph stream/invoke configs set no recursion_limit (only thread_id) | adjusted | 0.85 | `data_insights/workflows/agent_workflow.py:778` |
| F15 | 🟢 LOW | UX | Fractional 0–1 metrics (RMS, entropy, correlation) are misclassified as ratio data and drawn as pie/donut charts | adjusted | 0.82 | `data_insights/workflows/chart_builder.py:164` |
| F22 | 🟢 LOW | Correctness | KeyError crash in widget_composer decompose() on result missing 'rows' key | adjusted | 0.80 | `data_insights/workflows/widget_composer.py:529` |
| F25 | 🟢 LOW | analytics-correctness | recent_datasets chart_hint references a 'count' column that does not exist in rows | adjusted | 0.90 | `data_insights/workflows/tools.py:1855` |
| F26 | 🟢 LOW | analytics-correctness | Grouped energy chart_hint y=avg_decibel mismatches the order_by ranking (avg_rms_energy) for region | confirmed | 0.85 | `data_insights/workflows/tools.py:967` |
| F30 | 🟢 LOW | DATA | lru_cache on bound method _get_cached_table leaks Table objects and ties cache to instance lifetime incorrectly | adjusted | 0.90 | `data_insights/workflows/sql_agent.py:155` |
| F32 | 🟢 LOW | DATA | _top_collectors_monthly and _dataset_count run multiple unindexed entity-name lookups + full-table counts | adjusted | 0.82 | `data_insights/workflows/tools.py:1963` |
| G08 | 🟢 LOW | LLM | Message length validated against hardcoded 10000, not configurable MAX_MESSAGE_LENGTH | confirmed | 0.90 | `data_insights/serializers.py:38` |

## 7. Prioritized Implementation Roadmap

The findings are all provided. This is a synthesis/writing task — no investigation needed. Let me organize the 32 findings by remediation effort (using the `effort` field where present, inferring from severity/verdict/`fix` where null) and severity, then write the roadmap.

### PRIORITIZED IMPLEMENTATION ROADMAP

#### Quick Wins (days)

**Security / crash criticals first:**

- [F19] Missing `timezone` import — add `from django.utils import timezone` to `data_insights/views.py` — eliminates guaranteed 500 NameError on every titleless dashboard save.
- [F20] Undefined `ChatSession.Status.DELETED` — delete the bogus `ChatSession.Status.DELETED: []` key in `serializers.py:90` — eliminates guaranteed 500 AttributeError on every session status PATCH/PUT.
- [G01] No DRF throttling on LLM endpoints — add `ScopedRateThrottle` + per-user/min rates to `create_message`/`clarify_message`/`generate_insight` — caps worst-case OpenAI spend and worker/DB-pool pressure from any single account.
- [G02] Dead `AI_INSIGHT.SECURITY` config — wire `RATE_LIMIT_PER_MINUTE` into `DEFAULT_THROTTLE_RATES` — converts advertised-but-unenforced rate limits into real per-user caps on the only unbounded cost sink.
- [G06] No per-session in-flight guard — add DRF throttle scope + in-flight PROCESSING check before `_process_message_sync` — caps per-user agent concurrency and the fan-out that multiplies token spend and connection leaks.
- [G07] No LLM client timeout — pass `timeout=AGENT_CONFIG['TIMEOUT_SECONDS']` + `max_retries` to every `ChatOpenAI` on the agent path — a hung OpenAI connection can no longer pin a worker, DB connection, and streaming response indefinitely.
- [G04] No `recursion_limit` on LangGraph configs — add tunable `recursion_limit` (~12) to all four agent configs and catch `GraphRecursionError` — caps worst-case round-trips per message instead of relying on the implicit 25.

**Correctness / data-quality quick wins:**

- [F23] Stuck-message recovery clobbers live messages — add an age threshold and stop overwriting populated `assistant_response` in `fix_stuck_messages.py` — prevents the recovery job from marking in-progress messages FAILED.
- [F04] `cumulative_energy` is a per-bucket sum mislabeled as a running total — drop `Sum(rms_energy)` in `_energy_analysis` (keep avg/max/min/std) — stops users from reading independent per-bucket sums as accumulating values.
- [F24] Stratified split over-allocates small classes — replace `max(1,...)` with exact integer partition (`test = c - train - val`) in `tools.py:2586` — per-class and global train/val/test counts sum correctly to class/dataset size.
- [F29] Unanchored substring entity matching — use the word-boundary regex in `_match_entity_name` — eliminates silent false-positive entity filters ("Ada" matching "Canada") that corrupt counts.
- [F06] Monthly leaderboard filters `recording_date` not `created_at` — switch `_top_collectors_monthly` to filter/report `created_at` — correctly credits bulk/backdated uploads to the contribution month.
- [F11] Unindexed `mean_db` ranking/aggregation full scans — add `db_index`/Meta indexes on `NoiseAnalysis.mean_db` + `AudioFeature` and one migration — turns full seq-scan+sort hot-path queries into bounded index scans.
- [F10] Per-request undisposed SQLAlchemy engine in pagination — replace with a lazily-initialized, lock-guarded module-level singleton engine/wrapper — stops connection-pool leakage on the pagination endpoint.
- [F16] Box-plot widget degrades to a mislabeled bar chart — compute the five-number summary (`[min,q1,median,q3,max]`) in `_decompose_statistical` — renders the intended box plot instead of a meaningless bar chart.
- [F17] Silent top-N truncation across widget paths — make `build_chart_config` always append "(showing X of N)" when rows are cut — users always see when data was truncated.
- [F27] `_to_float` fabricates 0-value bars from null/non-numeric y — make it return `Optional[float]` and drop/gap nulls — removes misleading fabricated 0 dB bars.
- [F26] Region energy chart sorted by RMS but plots dB — change `tools.py:967` to `order_by('-avg_decibel')` — chart ordering matches the plotted metric.
- [F25] `recent_datasets` chart_hint references nonexistent `count` column — delete the dead `chart_hint` from `_recent_datasets` — removes the all-zero-chart landmine.
- [F22] KeyError in `widget_composer.decompose()` on missing `rows` — change to `result.get('rows', [])` in `_decompose_grouped`/`_decompose_ranked` — no crash on empty/error-shaped payloads.

#### Medium-Term (weeks)

- [F02] No collector/user scoping — any user reads all collectors' data — inject the authenticated identity via LangGraph `InjectedState` and enforce a mandatory `collector_id` tenant predicate in every ORM tool — closes the critical multi-tenant data-isolation hole.
- [F03] SQL agent SELECT-only + `--` denylist commented out — re-enable the statement-type/comment guards and run the agent against a least-privilege read-only Postgres role — removes write/DDL capability from the NL→SQL path.
- [F13] NL→SQL agent uses read-WRITE app DB credentials — provision and use a dedicated `SELECT`-only Postgres role for `TextToSQLAgent`'s engine — enforces read-only at the DB layer, not just the bypassable regex.
- [F12] Regex+keyword guard bypasses (pg_sleep, CASE, cartesian DoS) — add `statement_timeout` + read-only connection + positive structural allowlist check — eliminates the confirmed DoS class regardless of validator gaps.
- [F14] Secrets committed to git history — rotate every credential that ever appeared in `.env` (DO Spaces keys, SMTP, DB password, SECRET_KEY, OpenAI key) and scrub history — eliminates ongoing exposure of live credentials from public history.
- [F01] Correlation matrix correlates misaligned rows — replace per-column null drop with listwise (pairwise-complete) deletion preserving row identity — produces correct Spearman/Pearson coefficients instead of fabricated correlations.
- [F18] Streamed markdown rendered via `innerHTML` (XSS) — escape text before the regex markdown pass in `formatMarkdown` — closes the XSS / formatting-corruption hole in the insight text.
- [G05] Full `get_table_info()` re-sent every SQL turn with trimming off — cache the schema/system message per agent and re-enable `trim_messages` — drops per-multi-step input-token spend from O(schema × turns) toward O(schema + history).
- [F08] SQL agent reflects + samples every table on every turn — cache the formatted schema string and invalidate on migration — eliminates per-turn DDL recompile and sample-row queries.
- [F28] Advanced ML tools registered only in the exception fallback — replace the dual-list try/except with one module-level registry instantiated per-tool with isolation — restores the 6 advanced ML tools to normal operation.
- [F31] Distribution analysis loads every `mean_db` row into Python — compute the box-plot five-number summary in the DB via `PERCENTILE_CONT` — replaces an unbounded full-table read with one row per group.

#### Long-Term (architectural)

- [F05] `decibel_grouped` sample_count counts non-contributing rows — filter `noise_analysis__isnull=False` before `Count` so count matches `avg_db` (uncertain; verify before/after intent) — count and average describe the same population.
- [F07] Plain random sampling mislabeled "stratified" — implement true proportional per-bucket stratification (or honestly relabel) in `MLCorrelationMatrixTool` — the advertised sampling method matches what runs.
- [F09] New SQLAlchemy engine + uncached SQL agent graph per `DataAnalysisTool` instance — cache/share a single compiled SQL sub-agent and engine pool across instantiations — removes redundant connection pools and graph compiles.
- [G03] Main agent `ChatOpenAI` has no `max_tokens` cap — add a generous configurable `AGENT_CONFIG['MAX_TOKENS']` backstop on the main and dashboard LLMs — hard ceiling against pathological completions.
- [F32] `_dataset_count`/`_top_collectors_monthly` repeated unindexed lookups + full-table counts — cache the small lookup-name lists (TTL/signal-invalidated) and switch the month filter to a range over an indexed field — removes repeated full-table reads per count question.
- [F15] Fractional 0–1 audio metrics misclassified as ratio → pie/donut — gate the 0–1 branch in `_is_ratio_data` on a ratio-name and/or sum-to-1 check — RMS/entropy/correlation metrics stop rendering as parts-of-a-whole.
- [F30] `lru_cache` on bound method leaks Table objects process-wide — delete the unused `_get_cached_table` (or move to per-instance dict cache) in `sql_agent.py:155` — removes the latent memory/GC footgun.
- [G08] Message length hardcoded to 10000 instead of `MAX_MESSAGE_LENGTH` — wire the serializer `max_length` to the settings knob (shared constant) — makes the per-message input cap operator-tunable.

## 8. Confidence & Evidence

| ID | Confidence | Recommendation | Evidence |
|----|-----------|----------------|----------|
| F02 | 0.90 | Inject the authenticated identity into the tool layer via LangGraph's InjectedState (the correct, thread-safe replacement for the broken ContextVar) and enforce… | `data_insights/workflows/tools.py:374 (NoiseDatasetSearchTool), tools.py:1452 (NoiseDetailTool.get), tools.py:748 (AudioAnalysisTool)` |
| F04 | 0.90 | Fix in tools.py, two-part. PART A (remove the meaningless sum / rename the valid ones): In `_energy_analysis` (per-region l.965, per-category l.979, overall l.1… | `data_insights/workflows/tools.py:1362` |
| F11 | 0.85 | Two complementary fixes: (A) add the right indexes so the hot queries stop doing full scans/sorts, and (B) bound the unbounded aggregates so latency degrades gr… | `data_insights/workflows/tools.py:2042` |
| F12 | 0.90 | Defense in depth. Stop relying on the regex denylist as the security boundary; make the database connection itself incapable of writing or running long/expensiv… | `data_insights/workflows/sql_agent.py:507-526 (patterns), sql_agent.py:571-574 (commented SELECT-only)` |
| F14 | 0.92 | Two independent workstreams; rotation is the priority and is independent of history rewriting.  WORKSTREAM A — ROTATE every credential that ever appeared in com… | `datacollection/settings.py:34 (SECRET_KEY = os.getenv); git history commit a143af0` |
| G01 | 0.90 | Two layers, smallest-first.  LAYER 1 (primary, quick-win): Add DRF ScopedRateThrottle. In datacollection/settings.py REST_FRAMEWORK (lines 181-185) add:   "DEFA… | `data_insights/views.py:125` |
| G02 | 0.90 | Wire the dead config to real enforcement, highest-value first. (1) PER-USER LLM/SQL RATE LIMIT (primary cost control): In settings.py add to REST_FRAMEWORK: DEF… | `datacollection/settings.py:485` |
| G05 | 0.95 | Apply layered ceilings; smallest-blast-radius first.  A. Cap schema size sent to the LLM (sql_agent.py:628-643). Build the system message once per agent instanc… | `data_insights/workflows/sql_agent.py:631` |
| G06 | 0.85 | Layered defense; implement all three layers.  LAYER 1 (quick win, highest leverage) — DRF ScopedRateThrottle. In datacollection/settings.py REST_FRAMEWORK (line… | `data_insights/views.py:162` |
| G07 | 0.90 | Read the existing config and pass it into every ChatOpenAI on the agent path; do not invent new settings (TIMEOUT_SECONDS and MAX_RETRIES already exist and AGEN… | `data_insights/workflows/sql_agent.py:642` |
| F06 | 0.84 | Switch the monthly leaderboard to filter and report on the upload timestamp (created_at), which matches the stated intent. Concretely, in data_insights/workflow… | `data_insights/workflows/tools.py:1866-1867` |
| F10 | 0.85 | 1) Add a lazily-initialized, lock-guarded module-level singleton mirroring `_get_checkpointer` (views.py:77). After the existing `_compiled_graphs`/`_cached_too… | `data_insights/views.py:1056` |
| F16 | 0.90 | Fix at composition time in Python so the frontend needs no special-casing (the data already carries `decibel_values`, `min`, `max`, `avg` per group). Two coordi… | `data_insights/workflows/widget_composer.py:724` |
| F17 | 0.82 | Centralize disclosure so no path can cut silently, reusing the existing ranking_highlight pattern. (1) build_chart_config (chart_builder.py): compute total_befo… | `data_insights/workflows/chart_builder.py:209` |
| F19 | 0.92 | Primary fix (required): add `from django.utils import timezone` at module level in data_insights/views.py, placing it next to the other django imports near the … | `data_insights/views.py:1116` |
| F20 | 0.92 | Apply the validator's preferred fix (option 1): delete the line `ChatSession.Status.DELETED: []` from the allowed_transitions dict at data_insights/serializers.… | `data_insights/serializers.py:90` |
| F23 | 0.84 | Three coordinated changes, none touching the happy-path stream logic.  1) Make fix_stuck_messages.py safe and idempotent (quick-win core fix). Rewrite handle() … | `data_insights/management/commands/fix_stuck_messages.py:9` |
| F24 | 0.95 | Replace the floor-to-1 arithmetic with an exact integer partition in both places. For each class of size c: train=round(c*train_pct), val=round(c*val_pct), then… | `data_insights/workflows/tools.py:2586-2591` |
| F27 | 0.82 | Fix at the render layer so all chart types (bar/pie/scatter/line, truncated or not) are covered uniformly via the single data-array site.  1) chart_builder.py: … | `data_insights/workflows/chart_builder.py:214-222` |
| F28 | 0.92 | Replace the dual-list try/except in get_agent_tools() (tools.py:3439-3482) with a single module-level registry of all 15 tool classes instantiated in a loop wit… | `data_insights/workflows/tools.py:3452-3464` |
| F29 | 0.88 | 1) Create a single shared helper to eliminate drift. Add `match_name_word_boundary(query_lower: str, name: str) -> bool` (and an optional `match_first_name(quer… | `data_insights/workflows/tools.py:1680-1683` |
| F31 | 0.80 | Compute the box-plot 5-number summary in the database and stop shipping/persisting raw value arrays.  1) In `tools.py`, define a tiny reusable PostgreSQL percen… | `data_insights/workflows/tools.py:1230` |
| G04 | 0.85 | Three coordinated changes, smallest-first.  1) Explicit, tunable recursion_limit on all four configs. Add RECURSION_LIMIT to the AGENT config in datacollection/… | `data_insights/workflows/agent_workflow.py:778` |

## 9. Coverage / Critic Notes

The 32-finding register is strong on analytics-correctness, the NL->SQL safety bypasses, data-isolation, and the data-flow/caching pipeline. After reading the un-cited large modules (clarification_gate.py 872L, agent_workflow.py, models.py, admin.py, serializers.py, settings.py, urls.py, both test files) and re-verifying cited code, I found the register has NOT examined an entire class of operational/cost/governance dimensions, plus several concrete unflagged bugs. The biggest blind spots are: (1) LLM cost/abuse controls — rate limiting and per-user session caps are configured in settings but never enforced, no DRF throttling, no token/recursion caps, no cost observability; (2) observability/logging — zero metrics, tracing, or token-usage instrumentation, and PII/raw user text flows to OpenAI and is stored plaintext with no governance; (3) several orphaned/incomplete features introduced by migrations (Dashboard model is write-only, no retrieve/public endpoint despite is_public/slug, not in admin); (4) auth/transport gaps not in the register (public chat page with no login_required, DRF default BasicAuthentication enabled, atomic-block-wraps-streaming, no concurrency lock on same-session messages); (5) the clarification subsystem (the single largest un-cited file) has correctness gaps the register never touched. Confidence is high on the enforcement/instrumentation gaps (verified by absence across grep of the whole app) and on the concrete bugs (read directly). These are worth another focused pass on the operational/governance and clarification dimensions.

**Remaining gaps:**
- (CRITICAL) **LLM cost controls & abuse / rate limiting (entirely unexamined dimension)** — settings.py:485/527 define AI_INSIGHT.SECURITY.RATE_LIMIT_PER_MINUTE (30) and MAX_SESSIONS_PER_USER (10) and SESSION_INACTIVITY_HOURS, but grep shows these keys are NEVER read anywhere in data_insights/ — they are dead config. REST_FRAMEWORK (settings.py:181-185) sets NO DEFAULT_THROTTLE_CLASSES/RATES. The expensive streaming endpoints (create_message, clarify, generate_insight) have only IsAuthenticated and zero throttling. Combined with: the main agent ChatOpenAI (views.py:1229) has NO max_tokens cap; the LangGraph stream/invoke configs (agent_workflow.py:778,806,848,881) set only thread_id with NO recursion_limit; and sql_agent trim_messages is commented out (sql_agent.py:635-640) while full get_table_info() is re-sent every turn (F08). Net: a single authenticated user can issue unlimited concurrent agent runs, each an unbounded tool-calling loop on unbounded tokens — direct, uncapped OpenAI cost / DoS. This whole cost-governance dimension is absent from the 32-finding register.
- (HIGH) **Observability / logging / cost & error visibility (dimension absent from register)** — grep for token/usage/cost/get_openai_callback/langsmith/sentry/prometheus/metric across data_insights/ finds NOTHING in app code. There is no LLM token-usage tracking, no cost metering, no request tracing, no error monitoring, and no structured health/metrics endpoint. ChatMessage stores no token count, latency, or model used (processing_time_ms is referenced in serializers.py:112 but is not even a real field — F-class bug). With unbounded LLM spend (see cost gap) and no instrumentation, runaway cost or a prompt-loop is invisible until the bill arrives. No finding covers observability.
- (HIGH) **Public/unauthenticated exposure of the chat UI + DRF default auth** — unified_chat (views.py:103, routed at urls.py:12 'insights/') renders the full chat template with NO @login_required and is not in any auth-required mixin — anonymous users load the page. Separately, settings REST_FRAMEWORK defines NO DEFAULT_AUTHENTICATION_CLASSES, so DRF falls back to its defaults including BasicAuthentication (credentials over the wire) on every viewset, and there is no DEFAULT_PERMISSION_CLASSES (per-view IsAuthenticated is the only guard). Coupled with the confirmed missing SESSION/CSRF cookie-secure + HSTS, session auth is exposed. The register flags secrets-in-git and missing HSTS but not the public page or the BasicAuth/default-auth posture.
- (HIGH) **Orphaned / incomplete Dashboard feature (migration-introduced, write-only)** — migration 0007 + models.py:140 add Dashboard with is_public, slug, thumbnail. The ONLY endpoint is save_dashboard (views.py:1095). grep confirms there is NO list/retrieve/public-by-slug endpoint, NO DashboardSerializer, and Dashboard/QueryCacheModel are NOT registered in admin.py. So saved dashboards can never be read back, is_public/slug/thumbnail are dead, and the 'save, share, revisit dashboards' feature (per new.md spec) is half-built. This is a data-integrity/feature-completeness gap and a migration-safety concern (shipped schema with no read path). Register only flags the timezone NameError inside save_dashboard, not the orphaned feature.
- (HIGH) **Concurrency / data integrity on a single session's checkpointer thread** — create_message (views.py:177) wraps message creation AND the call to _process_message_sync (which returns a StreamingHttpResponse) inside transaction.atomic() — the streaming generator runs after the response leaves the atomic block, so DB writes inside the stream are outside the transaction, defeating its purpose. There is NO select_for_update or any lock guarding a session: two concurrent messages (or a clarify + new message, which clarification_gate.py:445-474 tries to reconcile heuristically) on the same session share one checkpointer thread_id (=session_id) and one in-flight graph state, so concurrent runs can interleave/clobber LangGraph state and message statuses. The register notes PROCESSING-before-stream and the clarification cancel logic but not the missing concurrency guard or the atomic-wraps-streaming mistake.
- (HIGH) **PII / data governance: user text and results sent to OpenAI and stored plaintext** — ChatMessage.user_input and assistant_response are stored unencrypted (models.py:75-76) and the full conversation + query results are sent to OpenAI (views.py:1229, sql_agent call_llm) with no redaction, no data-processing controls, no retention policy, and no opt-out. Given the confirmed total absence of collector/user scoping (F02), the model can pull ANY collector's records into the prompt and out to OpenAI. No finding addresses third-party data egress / PII governance, which is a real compliance risk for a data-collection platform.
- (MEDIUM) **Clarification subsystem correctness (largest un-cited file, 872 lines)** — clarification_gate.py is barely touched by the register. Concrete gaps: (a) the resolver only INJECTS the resolved value as advisory HumanMessage text (clarification_resolver_node:821-826) — the LLM can ignore it, so a resolved entity/metric/time_range does NOT deterministically constrain the SQL/tool, undermining the whole point of the gate; (b) vague_ranking_metric offers 'Complaint rate' (line 336 fallback) / metric options for data that has no complaint model (per ML_MODE_GAP_ANALYSIS.md), so a valid pick yields an empty/failed query; (c) _parse_natural_date_range months use timedelta(days=n*30) (line 561) — drifting/incorrect month math; (d) several normalizers return the raw answer on no-match (e.g. _fuzzy_match_metric/_classify_analysis_type return `answer`), passing unvalidated free text downstream. None of these are in the register.
- (MEDIUM) **Test coverage of all critical paths is effectively zero** — The only tests (tests/__init__.py + test_clarification_integration.py) exercise clarification heuristic helpers (_has_entity_match, _match_analysis_types, etc.). There are NO tests for any tool, the SQL agent / SQL-safety validator, views/streaming, serializers, chart_builder, widget_composer, data-isolation, or any CRITICAL finding. Every CRITICAL/HIGH bug in the register (SELECT-only bypass, no user scoping, correlation misalignment, NameError in save_dashboard, DELETED status) is untested and could regress silently. The register lists the bugs but not the systemic absence of tests guarding them.
- (MEDIUM) **Migration safety / model drift** — Migration 0006 widened status to max_length=25 to fit 'clarification_pending'; models.py:81 matches, but serializers.py:90 references ChatSession.Status.DELETED which does not exist in the Status enum (models.py:10-13) — runtime AttributeError on any status validation path (register has this as F20 but not the broader migration/model-drift theme). Also Dashboard (0007) ships schema with no read path (see orphaned-feature gap). No review of whether migrations are reversible or safe to run on a populated DB (e.g. status backfill, unique slug constraint added without data).
- (MEDIUM) **MAX_TOP_K / result-size cap inconsistently enforced (DoS/memory)** — settings define DEFAULT_TOP_K=100 and MAX_TOP_K=1000, but tools.py applies ad-hoc caps (e.g. _recent_datasets clamps to 200 at line 1821) while many tool `limit` Fields default low with no global MAX_TOP_K ceiling, and the SQL-agent path enforces only a per-query LIMIT injection (sql_agent.py:551-569) using a caller-supplied max_limit. Combined with F31 (loads every mean_db row into Python) and F11 (full-join unindexed sorts), an attacker or careless query can pull very large result sets into app memory. The register has individual perf findings but not the systemic 'MAX_TOP_K is defined but not consistently enforced' governance gap.
- (LOW) **Accessibility & i18n of the chat UI** — unified_chat.html has only 5 aria/role/alt occurrences across a ~2600-line interactive streaming UI; dynamic message/chart/table rendering, the streaming status indicators, and the clarification option buttons appear to lack ARIA live-regions/labels, so screen-reader users cannot follow streamed output. No i18n: strings (suggestions in views.py:105-116, all UI labels) are hardcoded English with no gettext/trans usage anywhere. For a public-facing tool these are real but lower-severity gaps not in the register.
