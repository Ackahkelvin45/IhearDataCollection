# Clarification Pipeline — Implementation Record

## Date: 2026-05-12

## Files Modified

| File | Changes |
|------|---------|
| `data_insights/workflows/clarification_gate.py` | Core fixes: entity detection, dead lambda, pending-clarification state, new-message-during-pending, observability |
| `data_insights/workflows/agent_workflow.py` | Added `pending_clarification_payload` to `AgentState`, `create_initial_state()`, `process_clarification_response()`, `process_clarification_response_async()` |
| `data_insights/views.py` | Read `pending_clarification_payload` from checkpointer state instead of module dict |
| `data_insights/workflows/tools.py` | Added `force_analysis_type` to `AudioAnalysisInput` and `_run()` |
| `data_insights/workflows/prompt.py` | Narrowed LLM "ask a clarifying question" instruction; updated `CLARIFICATION_CONTEXT_TEMPLATE` for `force_analysis_type` |

---

## Fix 1: `_has_entity_match()` substring → word-boundary matching

**File:** `clarification_gate.py:118-124`

Changed from `name.lower() in q` to `re.search(r'\b' + re.escape(name.lower()) + r'\b', q)`.

This prevents short entity names like "Ho" from matching common English words like "how", "who", "shows", "hours". Also prevents "Central" matching "centralized", "Bono" matching "bonus".

Tests: `_has_entity_match("show me data for ho region", {"available_regions": ["Ho"]})` → True. `_has_entity_match("how do I get data", {"available_regions": ["Ho"]})` → False.

---

## Fix 2: `_needs_entity()` — remove prepositions, remove hardcoded cities, use DB context

**File:** `clarification_gate.py:50-53, 113-121`

- Removed `"in"`, `"from"`, `"within"` from `ENTITY_HINT_WORDS`
- Removed hardcoded city names `"accra"`, `"kumasi"`, `"tamale"` from `ENTITY_HINT_WORDS`
- Changed `_needs_entity(query)` to `_needs_entity(query, context=None)` — now uses DB-derived entity names from `context` for detection, with word-boundary matching
- Updated signal detection lambda to pass `context`: `lambda query, context: _needs_entity(query, context) and not _has_entity_match(query, context)`

Verified: `_needs_entity()` is only called from the signal detection lambda (line 169). No other callers.

---

## Fix 3: Dead lambda for `multi_analysis_type`

**File:** `clarification_gate.py:244-256`

Removed the inline special-case `if signal_name == "multi_analysis_type":` block. Now all signals use the unified `signal["detection"](query, context)` call. `matched_types` is captured after detection for the `options_fn` call.

---

## Fix 4: `_pending_clarifications` → checkpointer state

**Files:** `clarification_gate.py:470-496`, `agent_workflow.py:32-51, 548-565, 631-642, 663-674`, `views.py:651-664`

- Removed module-level `_pending_clarifications` dict
- `emit_clarification_node()` now returns `{"pending_clarification_payload": payload}` — written to LangGraph state, persisted in PostgresSaver
- `get_pending_clarification()` returns `None` — directs callers to the checkpointer path
- Added `pending_clarification_payload: Optional[Dict[str, Any]]` to `AgentState` and all initial state constructors
- `views.py` reads `pending_clarification_payload` from `final_snapshot.values` after stream completes — same pattern as `pending_chart`/`pending_artifact` at lines 374-382

This fixes the multi-worker correctness bug where the module dict was per-process and workers couldn't share clarification state.

---

## Fix 5: New message during pending clarification

**File:** `clarification_gate.py:196-238`

When a user sends a new message while `clarification_pending=True` and the new message differs from `original_query`, the gate resets the clarification state and re-evaluates from the new message:

```python
if new_query and new_query != original_query:
    return {
        "clarification_pending": False,
        "clarification_answer": None,
        "clarification_question": None,
        ...
    }
```

Previously the new message was silently ignored and the old clarification was re-emitted.

---

## Fix 6: `force_analysis_type` on `AudioAnalysisTool`

**Files:** `tools.py:652-685, 694-715`

- Added `force_analysis_type: Optional[str]` to `AudioAnalysisInput` Pydantic model
- Added `force_analysis_type` parameter to `_run()`
- When `force_analysis_type` is set, it takes precedence over both the LLM's `analysis_type` and the internal auto-detector:

```python
if force_analysis_type:
    analysis_type = force_analysis_type
elif not analysis_type:
    analysis_type = self._auto_detect_analysis(query)
```

This prevents `_auto_detect_analysis()` from silently overriding the user's explicit analysis type choice.

---

## Fix 7: LLM prompt — defer structured disambiguation to gate

**File:** `prompt.py:61-62`

Changed from:
> When in doubt, ask a clarifying question rather than guessing.

To:
> For ambiguous queries about analysis type, time range, entity, or metric: proceed with a reasonable default — the system handles structured disambiguation before you are invoked. Only ask clarifying questions when the user's intent is genuinely ambiguous in a way structured options cannot resolve.

Updated `CLARIFICATION_CONTEXT_TEMPLATE` to instruct the LLM to use `force_analysis_type` when the dimension is "analysis_type".

---

## Fix 8: Observability logging

**File:** `clarification_gate.py:258-264, 482-488, 492-496`

Added structured logging at three checkpoints:
- **Gate no-signal:** `"Clarification gate: no signal fired query={query}"`
- **Gate signal fired:** `"Clarification gate: signal fired signal={name} dimension={dim} query={query}"`
- **Resolver:** `"Clarification resolved: dimension={dim} answer={answer} resolved_value={resolved} is_custom={is_custom}"`
- **Emit:** `"Clarification emitted: dimension={dim} question={question} thread={thread}"`
- **New message cancel:** `"New message received during pending clarification — cancelling pending and re-evaluating"`

All log at INFO level via `loguru.logger`.

---

---

## Round 2: `_auto_detect_analysis()` constraint + tests

### Fix 9: `_auto_detect_analysis()` returns `None` on zero confidence

**File:** `tools.py:750-794`

Changed return type from `str` to `Optional[str]`. On zero confidence (all keyword scores = 0), returns `None` instead of `"overview"`.

When `_run()` receives `None` from auto-detection (and `force_analysis_type` is not set), it returns an `"unresolved"` result:
```python
{
    "analysis_type": "unresolved",
    "message": "The analysis type could not be determined from the query. "
               "Ask the user what kind of analysis they want: energy, "
               "spectral, frequency, correlation, statistical, temporal, "
               "or overview.",
    "skip_visualization": True,
}
```

This closes the silent routing path: the tool no longer silently defaults to overview when it has zero confidence.

### Fix 10: `_match_analysis_types()` no longer defaults to `["overview"]` on zero matches

**File:** `clarification_gate.py:82-86`

Removed the `if not matched: matched.append("overview")` fallback. Zero-match queries now return `[]`. This means:
- `multi_analysis_type` signal does NOT fire (len 0, not > 1)
- The query passes through the gate as "clear"
- The tool's `_auto_detect_analysis()` also returns `None` (no keywords match)
- The tool returns "unresolved" → LLM asks the user what kind of analysis

Previously: zero-match → `["overview"]` → gate passes → tool auto-detects to `"overview"` silently.

### Unit tests added

**File:** `data_insights/tests.py` — 50 tests, 7 test classes:

| Class | Tests | Covers |
|-------|-------|--------|
| `HasEntityMatchTests` | 10 | Word-boundary matching for "Ho", "Central", "Bono", multi-word entities, empty context |
| `NeedsEntityTests` | 11 | Structural hints, DB context, preposition removal, word-boundary in context |
| `MatchAnalysisTypesTests` | 10 | All 7 analysis types, zero-match, single match, multi match |
| `HasRankingIntentTests` | 6 | All ranking words (best, worst, top, loudest, highest) + negative |
| `HasExplicitMetricTests` | 6 | decibel, db, rms, count, frequency + negative |
| `TimeRangeTests` | 7 | needs_time_range, has_time_reference (patterns, dates, today, negative) |

Run: `python manage.py test data_insights.tests --verbosity=2` — 50 tests, all pass.

---

## Verification

All modified Python files pass syntax check. 50 unit tests cover the heuristic helper functions.

To verify in the running app:
1. Start the Django dev server
2. Open `/insights/` in browser
3. Send `"show me the worst recordings"` → should see `signal_fired signal=vague_ranking_metric` in logs and clarification panel in UI
4. Send `"show me trends in the last 30 days"` → should see `no signal fired` in logs (gate passes through)
5. Send `"help me understand"` → gate passes through, tool returns "unresolved", LLM asks user what kind of analysis
6. Send `"show me data for ho"` with a region "Ho" in DB → should fire `missing_entity` (entity-seeking but no match)
7. Send `"how do I understand this"` → should NOT fire `missing_entity` (no substring match on "Ho")
8. Pick "Energy analysis" from clarification panel → verify `force_analysis_type` is set in the tool call
