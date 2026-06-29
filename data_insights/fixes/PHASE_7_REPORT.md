# Phase 7 — Remaining correctness & cleanup

# Phase 7 — Remaining Correctness & Cleanup

**Branch:** `data-insights-fixes`  **App:** `data_insights/`
**Prior-stage signals:** critique = approve (must_fix: 0) · testing check_passed: true (no regressions) · validation acceptance_met: true
**Verification this stage:** Phase 7 unit tests 20/20 pass; full `data_insights.tests` suite 66/66 pass; all six modified modules parse cleanly; dead-code removals confirmed callerless via grep.

This is the final phase: a set of remaining correctness bugs plus safe, verified cleanup. Edits were kept surgical, especially on the low-coverage streaming request path, where existing behavior was preserved except where a fix required changing it. The org-wide visibility decision, Phase 2 security, Phase 4 analytics, and Phase 5/6 work were left untouched.

## 1. Issues fixed (final status)

| ID | Issue | Status |
|----|-------|--------|
| **P7-1** | User-id `ContextVar` (`_current_user_id`) was `set()` on the request thread *before* returning the streaming response, but the tools read it inside the generator — which can run on a different worker thread — and it was never reset, so the value could leak across requests on a reused thread. | **Fixed.** `set()` now happens inside the generator (same execution context the tools consume it in via `tools.py`), capturing the token, and `reset(token)` runs in a new `finally` block on the streaming loop. Applied to both stream entry points in `views.py`. |
| **P7-2 (mark_processing)** | `message.mark_processing()` ran before the response was returned; a stream the client never consumed would strand the message in `PROCESSING` forever. | **Fixed.** `mark_processing()` moved to the first `yield` inside the generator. |
| **P7-2 (disconnect)** | On `GeneratorExit` (client disconnect) the handler unconditionally set the message to `FAILED`, which could clobber an already-`COMPLETED` message back to `FAILED` if the client dropped after a full answer. | **Fixed.** The disconnect handler now `refresh_from_db()`s and only marks `FAILED` when the message is not already `COMPLETED` and no assistant text was produced. |
| **P7-2 (fix_stuck_messages)** | The management command failed *all* `PROCESSING` messages, including in-flight ones currently streaming. | **Fixed.** Added a configurable `--minutes` age threshold (default 15) constraining `updated_at__lt=cutoff`; negative inputs fall back to the default. In-flight (recently updated) messages are never touched. |
| **P7-3** | `_decompose_grouped` / `_decompose_ranked` indexed `result["rows"]` directly; a routed `analysis_type` whose result lacked rows would raise `KeyError` *outside* the `post_process` guard and crash the request. | **Fixed.** Both now use `result.get("rows")` and fall through to a new shared `_fallback_single_chart()` helper when rows are missing/empty; return type widened to `Optional[...]`. `decompose()` stays exception-free. |
| **P7-4** | `ChatSession.increment_total_messages` was a read-modify-write (`self.total_messages += 1; self.save()`) — concurrent messages in the same session lost increments. | **Fixed.** Now an atomic `F("total_messages") + 1` UPDATE filtered by `pk`, followed by `refresh_from_db(fields=["total_messages"])` so the in-memory instance reflects the new value. |
| **P7-5 (cleanup)** | Dead/duplicate code: `_humanise` defined twice (chart_builder + widget_composer); no-op `DataAgent._update_query_handles`; deprecated `get_pending_clarification` returning `None`. | **Removed.** `widget_composer` now imports the single `_humanise` from `chart_builder`. All removals verified callerless by grep before deletion. |

## 2. Files modified

- `data_insights/views.py` — ContextVar set inside generator + `finally` reset; `mark_processing()` moved inside generator; disconnect handler made non-clobbering; removed unused `get_pending_clarification` import (both stream entry points, +65/−16).
- `data_insights/models.py` — `increment_total_messages` made atomic via `F()` + `refresh_from_db` (+7/−2).
- `data_insights/management/commands/fix_stuck_messages.py` — added `--minutes` age threshold and `updated_at__lt` cutoff guard (+40/−12).
- `data_insights/workflows/widget_composer.py` — import shared `_humanise`; new `_fallback_single_chart` helper; `.get("rows")` guards in grouped/ranked decompose (+22/−13).
- `data_insights/workflows/agent_workflow.py` — removed no-op `_update_query_handles` and its dead call site (+2/−16).
- `data_insights/workflows/clarification_gate.py` — removed deprecated `get_pending_clarification` (0/−9).
- `data_insights/tests/test_phase7_correctness_cleanup.py` — **new**, 294 lines, 20 DB-free `SimpleTestCase`s covering all five fix groups.

## 3. Key decisions & tradeoffs

- **ContextVar token-based reset over unconditional clear.** Used `reset(token)` rather than `set(None)` so nested/sequential contexts restore their prior value correctly; placed in `finally` so a client disconnect (`GeneratorExit`) still cleans up. Verified the only consumer is `tools.py` (`created_by_id=_current_user_id.get()` at two sites).
- **Idempotent disconnect handling via DB re-read.** Chose `refresh_from_db` + a `COMPLETED`/`produced_answer` check over tracking a local flag, so the guard is correct even if completion was persisted by a different code path before the disconnect.
- **Age threshold default 15 min, configurable.** Conservative enough to never fail a live stream, with `--minutes` to tune for operators. Negative values coerced to the default rather than erroring.
- **`_fallback_single_chart` as a named helper** instead of inlining the fallback three times — keeps the flat-result path identical across `decompose`, grouped, and ranked branches and documents intent.
- **DB-free tests.** All 20 tests are `SimpleTestCase` and assert behavior by capturing ORM filter/update kwargs and exercising the real `ContextVar`/generator shape, so they run without Postgres or OpenAI and stay fast/CI-portable. The streaming view bodies themselves remain integration-tested-by-shape (the test mirrors the `set/try/finally(reset)` structure) since the real path needs ASGI + DB.

## 4. Risks introduced

- **Streaming path edits (low coverage).** The `views.py` changes touch the request-streaming generator, which lacks end-to-end automated coverage. Risk is mitigated by keeping the edits structurally minimal and identical across both entry points, but only shape-level (not live ASGI) tests exercise them.
- **`refresh_from_db` on disconnect adds one extra query** in the disconnect path; negligible and only on the abnormal-termination branch.
- **`increment_total_messages` now always refreshes from DB**, so callers that previously relied on a stale in-memory `total_messages` would now see the committed value — this is the intended correction, but any caller depending on the old non-atomic semantics would observe the change.

## 5. Remaining concerns

- No live ASGI integration test asserts the ContextVar is reset and `mark_processing` ordering end-to-end through Django's async handler; current coverage is structural.
- `fix_stuck_messages` default threshold (15 min) is a heuristic; an unusually long-running legitimate query could exceed it. Operators can raise `--minutes`, but there is no per-message hard deadline tying the two together.
- Untracked vendored directories (`WrenAI/`, `open-claude-code/`, `opencode/`, `vanna/`) remain in the working tree but are unrelated to `data_insights/` and were not modified; they are noise in `git status`.

## 6. Recommended follow-up work

1. Add an ASGI/integration smoke test for the streaming endpoints that asserts (a) `_current_user_id` is `None` after a request, (b) a never-consumed stream does not strand `PROCESSING`, and (c) a post-completion disconnect leaves the message `COMPLETED`.
2. Consider a periodic/scheduled invocation of `fix_stuck_messages --minutes N` (cron/management) so stuck messages self-heal without manual runs.
3. Evaluate replacing the time-based stuck heuristic with an explicit per-message processing deadline persisted at `mark_processing`.
4. Gitignore or remove the unrelated vendored top-level directories to keep the working tree clean.
