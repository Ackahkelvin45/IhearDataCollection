# Phase 4 — Analytics correctness

All edits in `data_insights/workflows/tools.py` (+ tests). Field semantics verified
against `data/models.py` first (`NoiseAnalysis.mean_db` nullable via reverse
OneToOne; `recording_date` = when recorded, nullable; `created_at` = upload time,
`auto_now_add`). Statistically load-bearing math extracted into pure helpers
(`running_cumulative`, `complete_case_matrix`, `stratified_split_counts`) for
DB-free unit testing.

## Issues fixed

| ID | Issue | Fix | Status |
|----|-------|-----|--------|
| P4-1 (F01) | Correlation matrix correlated **misaligned** rows (per-column null-drop + position truncation) → fabricated coefficients/p-values | `complete_case_matrix`: keep only rows where **every** selected feature is non-null, so row *i* of every feature is the same observation; report true `n`/`sample_size` | resolved |
| P4-2 | `primary_matrix` assigned but never used (likely wrong-matrix bug) | dead variable removed; only the requested method's matrix is computed/returned (consistent labels) | resolved |
| P4-3 (F04) | `cumulative_energy` was a per-period **sum**, not a running total | rename per-period value to `period_energy`; `cumulative_energy` = `running_cumulative(...)` over month/date-ordered rows. (Non-temporal `_energy_analysis` group totals are legitimate and left as-is.) | resolved |
| P4-4 (F05) | `decibel_grouped.sample_count` counted datasets that didn't back the average | `Count('noise_analysis__mean_db')` (NULL-excluding) to match the `Avg` denominator | resolved |
| P4-5 (F28) | Advanced ML tools registered only in the `except` fallback → unreachable in normal operation | all ML tools registered on the **happy path**; only `WebFetchTool` (the import that can fail) is guarded | resolved |
| P4-6 | `top_collectors_monthly` used `recording_date` (not upload); correlation sampling mislabeled "stratified" | leaderboard uses `created_at`; random sampling relabeled "random" | resolved |
| P4-7 | Stratified train/val/test split over-allocated for small classes (independent rounding) | `stratified_split_counts` (floor-then-distribute / largest-remainder); also applied to `MLClassBalanceTool.split_recommendation` | resolved |

## Critique → revise

Critique approved with **no must-fix** items. Refactor normalized formatting only.

## Post-validation regression fix (orchestrator)

P4-5 made `DataAnalysisTool` always construct its `TextToSQLAgent`, which opens a
DB connection at construction (schema reflection). This broke **3** previously-passing
`test_clarification_integration` tests when the DB is unreachable, and reduced
production resilience (a DB hiccup at first tool load would disable the whole tool
set). **Fix:** the SQL agent is now built **lazily** on first query
(`DataAnalysisTool._get_agent`) instead of in a `model_validator`, so tool
construction is DB-free. This is also a partial fix for perf finding **F09**
(eager engine per instantiation). Added `LazyAgentConstructionTests` (2 tests).

## Tests / verification

- 25 Phase 4 tests (`test_phase4_analytics_correctness.py`), DB-free `SimpleTestCase`.
- Full `data_insights` suite: **164 passing** (1 skipped = DB-available branch).
- `manage.py check`: clean (only the pre-existing unrelated `urls.W005`).
- `black`: clean.

## Files modified

- `data_insights/workflows/tools.py`
- `data_insights/tests/test_phase4_analytics_correctness.py`

## Risks / remaining

- The corrected `cumulative_energy` and `sample_count` change the **values** users
  see (keys unchanged) — intended; charts consuming these keys are unaffected in shape.
- F09 only partially addressed (lazy agent); full engine-reuse/caching remains for
  **Phase 6**.
- Live correctness over real data not exercisable locally (no Postgres); helpers are
  unit-tested and the ORM aggregations are straightforward — verify on staging.

## Backlog (next phases)

- Phase 5: F18 XSS sanitize, ratio→pie misclassification, missing units, hidden truncation, box-plot fallback.
- Phase 6: F09 (full engine reuse/caching), F08/G05 schema-reflection caching, F10 paginate engine leak, F11 indexes, F31 unbounded distribution load.
- Phase 7: ContextVar threading, message state machine, decompose KeyError, dead code, duplication.
