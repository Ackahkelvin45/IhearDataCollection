# Phase 6 — Performance

# Phase 6 — Performance (engine reuse, schema caching, indexes, bounded loads)

## Summary

Phase 6 delivered five surgical, behavior-preserving performance optimizations
to the `data_insights/` app: read-only engine/connection-pool reuse, one-time
schema (`table_info`) memoization, an additive DB index on the hot decibel
column, and server-side bounded distribution computation. No output shape, no
Phase 4 statistical result, and no Phase 2 read-only/SELECT-only/statement-timeout
security guarantee was changed. Prior stages confirm: critique = **approve**
(0 must-fix), testing **passed** with no regressions, validation
**acceptance met**.

Verification performed this stage:
- `manage.py makemigrations --check --dry-run` → **"No changes detected"**
  (the single additive migration fully captures the model change; nothing else
  pending).
- `manage.py check` → only the pre-existing unrelated `urls.W005` namespace
  warning; no new issues.
- `manage.py test data_insights.tests.test_phase6_performance` → **18/18 OK**
  (DB-free `SimpleTestCase`s). The repeated read-only-credentials notices in
  output are the expected Phase 2 fallback warnings, not failures.
- All non-test, non-internal callers of `create_readonly_engine` now route
  through `get_readonly_engine`; migration `0031` correctly chains off `0030`.

## 1. Issues fixed (with final status)

| ID | Optimization | Status |
|----|--------------|--------|
| **P6-1** | `SQLDatabaseWrapper.get_table_info` memoizes the fully-rendered schema string per sorted table-name set (keyed `tuple(sorted(table_names))`), gated on `_enable_cache`. `call_llm` now reflects + sample-SELECTs the schema **once** per agent instead of on every LLM turn. Rendered content (incl. Phase 3 audio-column masking) is byte-identical to recomputing. | **Fixed** |
| **P6-2** | New module-level `_READONLY_ENGINE_CACHE` + `get_readonly_engine(...)` memoizes one engine (one connection pool) per **resolved** `(ro_user, host, port, name)` target. `TextToSQLAgent.__init__` reuses the cached engine instead of building a fresh pool per construction. | **Fixed** |
| **P6-3** | `data_insights/views.py` pagination/clarification re-execution path switched from `create_readonly_engine` → `get_readonly_engine`, eliminating the per-request connection-pool **leak** (engines were previously created and never disposed). | **Fixed** |
| **P6-4** | Distribution analysis (`_statistical_analysis` → new `_decibel_distribution_by_group`) now computes `count/avg/max/min` via `GROUP BY` aggregates and `q1/median/q3` via Postgres `percentile_cont` (new `PercentileCont` ordered-set `Aggregate`). Memory now scales with **group count**, not row count. `decibel_values` retained but **bounded** to `DISTRIBUTION_SAMPLE_LIMIT = 500`/group as a representative sample only. `widget_composer._decompose_statistical` prefers the exact precomputed `box_stats`, falling back to raw values for legacy producers. | **Fixed** |
| **P6-5** | Added `db_index=True` to `NoiseAnalysis.mean_db` (the hot sort/aggregate/filter/order-by column) in `data/models.py`; generated exactly one additive migration `0031_alter_noiseanalysis_mean_db.py` (not applied, per instructions). | **Fixed** |

## 2. Files modified

- `data_insights/workflows/sql_agent.py` — `_READONLY_ENGINE_CACHE` + `get_readonly_engine`; `_cached_table_info` memo on `SQLDatabaseWrapper`; `get_table_info` cache short-circuit/store; `TextToSQLAgent.__init__` and `call_llm` use cached engine / memoized schema.
- `data_insights/workflows/tools.py` — `PercentileCont` aggregate; `DISTRIBUTION_SAMPLE_LIMIT`; `_decibel_distribution_by_group`; `_statistical_analysis` category/region branches refactored to call it.
- `data_insights/workflows/widget_composer.py` — `_decompose_statistical` prefers exact `box_stats`, falls back to `decibel_values`.
- `data_insights/views.py` — import + pagination path use `get_readonly_engine`.
- `data/models.py` — `mean_db` gains `db_index=True` (with rationale comment).
- `data/migrations/0031_alter_noiseanalysis_mean_db.py` — **new**, single additive `AlterField`, depends on `0030`, includes DBA `CREATE INDEX CONCURRENTLY` guidance.
- `data_insights/tests/test_phase6_performance.py` — **new**, 18 DB-free unit tests.

## 3. Key decisions & tradeoffs

- **Cache key uses the RESOLVED read-only username**, so a credentials change yields a distinct engine and credentials are never shared across pools — reuse cannot widen the security boundary. Port is normalized to `int` so str/int callers share one engine.
- **`create_readonly_engine` kept** as the non-cached builder (still builds a fresh engine each call): the cache delegates to it, and isolated callers / the Phase 2 tests rely on it. This avoids reworking Phase 2's per-connection `_set_readonly_session` listener, which now lives on the cached engine and fires for every pooled connection.
- **`percentile_cont` modeled as a Django `Aggregate`, not `RawSQL`** — deliberate: RawSQL is opaque to the ORM and gets pushed into `GROUP BY`, producing invalid SQL. `percentile_cont` uses linear interpolation, matching the existing frontend / `_box_stats` `_quartile` method exactly, so the rendered box plot is numerically identical (Phase 4 results preserved).
- **`decibel_values` bounded but not removed** — preserves the output dict shape and backward compatibility for any chart overlaying raw points; box statistics come exclusively from the exact `box_stats`, so the 500-cap changes no statistical result.
- **Migration left unapplied** and ships the default `AlterField` (safe on small/empty tables) plus explicit `CREATE INDEX CONCURRENTLY` / `--fake` instructions for large production tables.

## 4. Risks introduced

- **Process-lifetime engine cache:** engines/pools now live for the process lifetime keyed by target. If DB credentials/host rotate at runtime, a new key creates a new engine (old engine lingers until process exit but is not reused) — bounded and benign for a typical worker lifecycle. No eviction policy.
- **`table_info` memo staleness:** schema is rendered once per wrapper instance per table set; an in-process live schema change would not be reflected until a new wrapper/agent is created. Acceptable given agents are short-lived and schema is effectively static.
- **`percentile_cont` is Postgres-specific.** The app already targets Postgres (engine dialect + Phase 2 session settings), so this is consistent, but the distribution branch would not run on SQLite/other backends. Tests are DB-free and assert SQL generation only.
- **Index migration not applied:** the performance benefit of P6-5 is latent until a human applies (or `CONCURRENTLY`-creates + `--fake`s) migration `0031`.

## 5. Remaining concerns

- Engine cache has **no max size / eviction**; fine for the small fixed set of DB targets here, but worth noting if multi-tenant/per-user credentials are ever introduced.
- Distribution server-side path covers the `category` and `region` group-bys that existed; no new group-by dimensions were added (scope preserved).
- P6-5's write-blocking `CREATE INDEX` on a large `data_noiseanalysis` is documented but depends on the operator following the `CONCURRENTLY` guidance.

## 6. Recommended follow-up work

1. **Apply migration `0031`** in each environment — prefer `CREATE INDEX CONCURRENTLY` + `migrate data 0031 --fake` on large/production tables (instructions embedded in the migration).
2. **Configure dedicated read-only DB credentials** (`AI_INSIGHT_DB_READONLY_USER/PASSWORD`) so the cached engine uses a least-privilege role — currently it falls back to read-write app creds (forced read-only at session level). This is a Phase 2 carry-over surfaced again by the test warnings.
3. Consider a lightweight engine-cache **eviction/health-check** (e.g. `pool_pre_ping`, already-or-not) if runtime credential rotation or many DB targets become real.
4. If further `group_by` dimensions are added to distribution analysis, route them through `_decibel_distribution_by_group` to keep them server-side/bounded.
5. Optional: a real (DB-backed) integration test asserting the `percentile_cont` results equal the previous Python-quartile path on a fixture dataset, to lock in numerical equivalence end-to-end.

