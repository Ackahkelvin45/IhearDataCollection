# Clarification Intent Pipeline — Evaluation

## Overview

The `data_insights` app intercepts ambiguous user queries before the LangGraph agent dispatches tools. A heuristic gate node detects ambiguity, emits structured questions to the frontend, collects answers, and re-routes with a resolved intent.

This document evaluates the implementation and recommends fixes. All call-site analysis and code paths referenced below have been verified against the current codebase.

---

## Architecture

```
user_message (POST /insights/sessions/{id}/messages/)
     │
     ▼
[cleanup_expired_handles]                      ← existing
     │
     ▼
[clarification_gate]                           ← heuristic gate node
     │
     ├── intent is clear ────► [agent] → [tools] → [post_process] → [finalize_dashboard] → [format] → __end__
     │
     └── intent is ambiguous ──► [emit_clarification] → __end__
                                          │
                                  user answers (POST /messages/{id}/clarify/)
                                          │
                                          ▼
                                  [clarification_resolver] → [agent] → ...
```

Key files:
- [clarification_gate.py](data_insights/workflows/clarification_gate.py) — gate node, resolver, normalizers, signal definitions, `_pending_clarifications` store
- [agent_workflow.py](data_insights/workflows/agent_workflow.py) — LangGraph graph wiring, `AgentState`, `process_clarification_response()`
- [prompt.py](data_insights/workflows/prompt.py) — LLM system prompt + `CLARIFICATION_CONTEXT_TEMPLATE`
- [tools.py](data_insights/workflows/tools.py) — `AudioAnalysisTool._auto_detect_analysis()` at line 734
- [views.py](data_insights/views.py) — streaming handlers, `_process_message_sync`, `_process_clarification_response`
- [unified_chat.html](data_insights/templates/data_insights/unified_chat.html) — frontend, clarification panel at line 104

**Infrastructure note:** Production uses `PostgresSaver` as the LangGraph checkpointer ([views.py:38-86](data_insights/views.py#L38-L86)), backed by a PostgreSQL connection pool. Conversation state persists across server restarts and is shared across all workers. There are **no existing tests** for any gate, signal, or clarification function.

---

## Flaw 1 (CRITICAL): `_has_entity_match()` substring matching causes false positives on short names

**File:** [clarification_gate.py:118-124](data_insights/workflows/clarification_gate.py#L118-L124)

```python
def _has_entity_match(query: str, context: Dict[str, Any]) -> bool:
    available = context.get("available_regions", []) + context.get("available_communities", [])
    q = query.lower()
    for name in available:
        if name.lower() in q:
            return True
    return False
```

`name.lower() in q` does substring matching without word boundaries. This means:

- Region `"Central"` matches `"centralized"` (false positive)
- Region `"Bono"` matches `"bonus"` (false positive)
- Region `"Ho"` (a real Ghanaian region) matches `"how"`, `"who"`, `"shows"`, `"hours"` — essentially every English query (catastrophic false positive)

When `_has_entity_match()` returns `True`, the `missing_entity` signal does NOT fire. So these false positives **suppress** the entity question when it should be asked. This silently corrupts behavior on a large fraction of queries.

This is a concrete, verified bug with immediate user impact.

### Fix

Use word-boundary matching:

```python
import re

def _has_entity_match(query: str, context: Dict[str, Any]) -> bool:
    available = context.get("available_regions", []) + context.get("available_communities", [])
    q = query.lower()
    for name in available:
        name_lower = name.lower()
        if re.search(r'\b' + re.escape(name_lower) + r'\b', q):
            return True
    return False
```

**Edge cases to test:**
- Multi-word names (`"Greater Accra"`) — `\b` anchors at whitespace, so this works correctly
- Hyphenated names (`"Sekondi-Takoradi"`) — `-` is `\W`, so `\b` boundaries land at `S` and `i`, which is correct
- Names with leading/trailing punctuation in the DB — `re.escape` will escape the punctuation and `\b` won't match around it. Normalize entity names before matching: strip punctuation, collapse whitespace
- Very short names (≤2 characters like `"Ho"`) — word-boundary matching reduces false positives by orders of magnitude but doesn't eliminate them entirely (the standalone word "ho" could appear in casual English). This is acceptable — a partial fix that works for 99% of cases is better than the current behavior

**Rollout:** This changes which queries trigger `missing_entity` across the board. No existing tests cover this function. Deploy with logging of signal firing rates to compare before/after.

---

## Flaw 2 (CRITICAL): `_needs_entity()` fires on common prepositions, blocking all other signals

**File:** [clarification_gate.py:50-53](data_insights/workflows/clarification_gate.py#L50-L53)

```python
ENTITY_HINT_WORDS = {
    "region", "community", "category", "area", "district", "zone",
    "accra", "kumasi", "tamale", "in", "from", "within",
}
```

`"in"`, `"from"`, `"within"` are prepositions, not entity hints. `missing_entity` has the highest priority in `SIGNAL_PRIORITY_ORDER` ([line 57](data_insights/workflows/clarification_gate.py#L57)), so it wins over all other signals.

Verified example: `"show me trends in the last 30 days"` triggers `_needs_entity()` (matches `"in"`), `_has_entity_match()` returns `False` (after Flaw 1's substring fix, no entity name matches), so `missing_entity` fires and asks "Which area?" — even though this is a temporal query with an explicit time reference.

The hardcoded city names (`"accra"`, `"kumasi"`, `"tamale"`) are also fragile. `_build_entity_context()` already queries `Region` and `Community` tables and passes results as `context` to the signal detection lambdas. The detection function for `missing_entity` at [line 169](data_insights/workflows/clarification_gate.py#L169) already receives `context` — it just doesn't use it for entity *presence* checks.

No other function calls `_needs_entity()` — it is only used in the signal detection lambda at line 169. This makes it safe to modify.

### Fix

1. Remove `"in"`, `"from"`, `"within"` from `ENTITY_HINT_WORDS`
2. Remove hardcoded city names; instead, check the DB-derived entity names from `context`:

```python
ENTITY_HINT_WORDS = {
    "region", "community", "category", "area", "district", "zone",
}

def _needs_entity(query: str, context: Dict[str, Any] = None) -> bool:
    q = query.lower()
    # Structural hint words
    if any(w in q for w in ENTITY_HINT_WORDS):
        return True
    # Entity names from DB context
    if context:
        all_entities = (context.get("available_regions", []) +
                        context.get("available_communities", []))
        for name in all_entities:
            if name.lower() in q:
                return True
    return False
```

3. Update the signal detection lambda at line 169 to pass `context`:
```python
"detection": lambda query, context: _needs_entity(query, context) and not _has_entity_match(query, context),
```

**What happens when `_build_entity_context()` fails?** If the DB query fails, it returns empty lists ([line 136](data_insights/workflows/clarification_gate.py#L136)). With the fix, `_needs_entity()` would still fire on structural hint words (`"region"`, `"area"`, etc.) but `_has_entity_match()` won't match any entity — so the signal fires. This is acceptable: if the DB is down, the system has bigger problems than a false-positive entity question.

---

## Flaw 3 (MODERATE): Three uncoordinated disambiguation systems

The system has three mechanisms that make or influence routing decisions. They use independent keyword sets and don't coordinate.

### Path A — Heuristic gate (structured, pre-tool)

[clarification_gate.py](data_insights/workflows/clarification_gate.py) — 5 regex-based ambiguity signals. Emits structured options (buttons) via a `clarification` streaming chunk. Runs BEFORE the LangGraph agent.

### Path B — LLM system prompt (free-text, during conversation)

[prompt.py:61-62](data_insights/workflows/prompt.py#L61-L62):
```
When in doubt, ask a clarifying question rather than guessing.
```

[prompt.py:111](data_insights/workflows/prompt.py#L111):
```
If no result is available, ask a clarifying question.
```

The LLM can generate free-text clarifying questions inline with no structure, no options, and no awareness of the gate.

### Path C — Silent tool-level auto-detection (most dangerous)

[tools.py:734-774](data_insights/workflows/tools.py#L734-L774) — `AudioAnalysisTool._auto_detect_analysis()` scores 6 analysis types using its own keyword system. If all scores are 0, defaults to `"overview"`. The user never sees this decision.

Concrete coordination failure: if the gate's `multi_analysis_type` signal fires and the user picks "Energy analysis", the gate resolves this via the `CLARIFICATION_CONTEXT_TEMPLATE` in [prompt.py:142-149](data_insights/workflows/prompt.py#L142-L149), which tells the LLM to use the resolved value as a "hard constraint." But the tool's `_auto_detect_analysis()` runs independently inside `AudioAnalysisTool._run()` at line 709. If the tool receives `analysis_type=None` (because the LLM followed the gate's instruction but didn't set the parameter explicitly, or because the LLM misunderstood), the auto-detector re-runs its own scoring and may pick a different type — silently overriding the user's answer.

### Fix

The tool-level auto-detector is the priority. The LLM prompt is secondary.

**Fix Path C (concrete, verifiable):** Add a `force_analysis_type` parameter to `AudioAnalysisTool` that, when set, skips `_auto_detect_analysis()`. Have the gate's resolver path propagate the resolved value to this parameter. The mechanism is:

1. When `clarification_dimension == "analysis_type"`, store the resolved type in the state (already done via `clarification_answer` in `clarification_resolver_node` at [line 459](data_insights/workflows/clarification_gate.py#L459))
2. The `CLARIFICATION_CONTEXT_TEMPLATE` injects the resolved value into the system prompt ([prompt.py:142-149](data_insights/workflows/prompt.py#L142-L149))
3. Add a `force_analysis_type` field to `AudioAnalysisInput` that the LLM is instructed to populate from the clarification context
4. In `_run()`: if `force_analysis_type` is set, use it and skip `_auto_detect_analysis()`

**Fix Path B (lower priority, probabilistic):** Narrow the LLM prompt instruction at [line 61-62](data_insights/workflows/prompt.py#L61-L62). The current instruction is too broad — it tells the LLM to ask clarifying questions for *any* doubt. But the gate already handles time ranges, entities, metrics, and analysis types. Change to:

```
"When in doubt about analysis type, time range, entity, or metric, proceed with a reasonable default — the system handles structured disambiguation before you are invoked. Only ask clarifying questions when the user's intent is genuinely ambiguous in a way the structured options cannot resolve (e.g., what a technical term means)."
```

This is a prompt change — probabilistic, not deterministic. But combined with the `force_analysis_type` parameter on the tool, it moves the system in the right direction. The prompt edit reduces overlap; the parameter enforces it.

---

## Flaw 4 (MODERATE): `_pending_clarifications` module dict breaks with multiple Gunicorn workers

**File:** [clarification_gate.py:470](data_insights/workflows/clarification_gate.py#L470)

```python
_pending_clarifications: Dict[str, Dict[str, Any]] = {}
```

The flow: `emit_clarification_node()` stores the clarification payload (question, options, dimension) in this dict keyed by `thread_id`. After the stream completes, `views.py` calls `get_pending_clarification(str(session.id))` to read and pop the entry.

This dict lives in Python module memory — each Gunicorn worker process has its own copy. If Worker A handles the initial message (stores the clarification in Worker A's dict) but Worker B handles any subsequent request that needs to read it, `get_pending_clarification()` returns `None` — the entry is in Worker A's memory.

The fix from the first version of this document ("add TTL") addressed the wrong problem. The issue is not memory growth (entries are small, per-process, bounded by worker lifetime) but **correctness** — the clarification payload is unreachable from other workers.

Note: the LangGraph checkpointer (`PostgresSaver`) is already shared across all workers. The checkpointer stores conversation state keyed by `thread_id` in PostgreSQL. So the infrastructure for cross-worker state sharing already exists.

### Fix

Store the clarification payload in the LangGraph state (which persists to Postgres via the checkpointer) instead of a side-channel dict. In `emit_clarification_node()`, instead of writing to the module dict, set a state field that survives through the checkpointer:

```python
# Replace:
_pending_clarifications[thread_id] = payload

# With a state update:
return {
    "pending_clarification_payload": payload,
}
```

This requires adding `pending_clarification_payload: Optional[Dict[str, Any]]` to `AgentState` in [agent_workflow.py:32-51](data_insights/workflows/agent_workflow.py#L32-L51).

Then in `views.py`, read it from the checkpointer state after the stream completes (the same pattern used for `pending_chart`/`pending_artifact` at [views.py:374-382](data_insights/views.py#L374-L382)).

**Sequencing:** LangGraph commits state at node boundaries. The `views.py` read happens after `workflow.stream()` completes, which guarantees the `emit_clarification_node` has committed its state update. The existing `pending_chart`/`pending_artifact` reads at lines 374-382 prove this pattern works.

If adding to `AgentState` is undesirable, the minimum fix is Django's cache framework (`django.core.cache`) with a 5-minute TTL — this gives cross-worker sharing and expiration. But this introduces a second state store, which adds complexity. The checkpointer approach is cleaner.

---

## Flaw 5 (MINOR): `vague_ranking_metric` false-positives on temporal queries

**File:** [clarification_gate.py:103-110](data_insights/workflows/clarification_gate.py#L103-L110)

```python
RANKING_WORDS = {
    "best", "worst", "top", "bottom", "highest", "lowest",
    "loudest", "quietest", "most", "least", "maximum", "minimum",
    "max", "min", "strongest", "weakest",
}

EXPLICIT_METRIC_WORDS = {
    "decibel", "db", "rms", "energy", "centroid", "spectral",
    "frequency", "zcr", "zero crossing", "duration", "count",
    "recordings", "complaint", "peak", "amplitude", "loudness",
}
```

`"most"` and `"least"` are in `RANKING_WORDS`. These are common English words that frequently appear in non-ranking contexts. A query like `"show me the most recent recordings"` triggers `_has_ranking_intent()` (matches `"most"`), `_has_explicit_metric()` returns `False` (nothing in the query matches the metric set), so `vague_ranking_metric` fires and asks "Rank by which metric?" But the user asked about temporal recency, not metric ranking.

Similarly, `"most frequent"` or `"most common"` would fire the signal even though "most" here is a superlative modifier, not a ranking intent.

The `ambiguous_temporal_word` signal at [line 176](data_insights/workflows/clarification_gate.py#L176) has the same class of problem: `"current"` and `"new"` are extremely common. A query like `"what is the current average decibel level"` fires `ambiguous_temporal_word` (matches `"current"`) even though this is not an ambiguous temporal reference.

### Fix

For `vague_ranking_metric`: check whether `RANKING_WORDS` matches are followed by a metric-dimension noun (from the EXPLICIT_METRIC_WORDS set) within a small window. If the ranking word is followed by "recordings", "data", "audio" etc. without a metric, fire. If followed by nothing that looks rankable, don't.

This is a tuning problem, not an architecture problem. Instrument the signal's firing rate in production and adjust keywords based on data. No immediate code change is required unless data shows this firing at a high rate on non-ranking queries.

---

## Flaw 6 (MINOR): Dead detection lambda for `multi_analysis_type` (maintenance hazard)

**File:** [clarification_gate.py:216-226](data_insights/workflows/clarification_gate.py#L216-L226)

```python
if signal_name == "multi_analysis_type":
    matched_types = _match_analysis_types(query)
    if len(matched_types) > 1:
        fired_signal = (signal_name, signal)
        break
elif signal["detection"](query, context):
    fired_signal = (signal_name, signal)
    break
```

The signal dict at [line 148](data_insights/workflows/clarification_gate.py#L148) stores a detection lambda:
```python
"detection": lambda query, context: len(_match_analysis_types(query)) > 1,
```

This lambda is never called — the inline special-case at line 216 bypasses it. The special-case exists to capture `matched_types` for the `options_fn` call at [line 234](data_insights/workflows/clarification_gate.py#L234), which needs the list of matched analysis types to build option buttons.

This is a maintenance hazard: if someone updates the lambda expecting it to change behavior, nothing happens. Not a bug — both code paths are functionally equivalent — but a trap for future developers.

### Fix

Use the lambda uniformly and capture `matched_types` after detection:

```python
for signal_name in SIGNAL_PRIORITY_ORDER:
    signal = AMBIGUITY_SIGNALS[signal_name]
    try:
        if signal["detection"](query, context):
            fired_signal = (signal_name, signal)
            if signal_name == "multi_analysis_type":
                matched_types = _match_analysis_types(query)
            break
    except Exception as exc:
        logger.warning(f"Signal detection error for {signal_name}: {exc}")
        continue
```

The surrounding context at lines 232-235 already handles `matched_types` correctly for the `options_fn` call, so this refactor is safe:

```python
if "options_fn" in signal:
    if signal_name == "multi_analysis_type" and matched_types:
        options = signal["options_fn"](matched_types)
    else:
        options = signal["options_fn"](context)
```

---

## Verified non-issues

Several things were flagged in the initial review but turn out not to be problems after verification:

### `original_query` loss on server restart — NOT an issue

Production uses `PostgresSaver` as the checkpointer ([views.py:83](data_insights/views.py#L83)), backed by a PostgreSQL connection pool. State persists across server restarts. `process_clarification_response()` at [agent_workflow.py:626](data_insights/workflows/agent_workflow.py#L626) creates a fresh state, but `original_query` and `clarification_dimension` are read from the checkpointer's persisted state when the resolver node runs. This is reliable in production. If local dev uses `MemorySaver`, this only affects local dev.

### Confirmation step — deferred feature, not a bug

[clarification_gate.py:443](data_insights/workflows/clarification_gate.py#L443) sets `confirmation_needed = False` with the comment "Confirmation step deferred to future iteration." The `views.py` confirmation check at lines 340-349 and the frontend `showConfirmationUI()` at [unified_chat.html:2333](data_insights/templates/data_insights/unified_chat.html#L2333) are forward-compatible scaffolding. The spec requires this feature, the code infrastructure is ready for it, and it was intentionally deferred. This is not a bug.

When this feature is built, it needs more than flipping the flag — the "No, let me rephrase" path, the `__CONFIRMED__` sentinel routing, and the resolver's handling of the rephrase flow must all be implemented and tested.

### `call_model()` inline import — negligible impact

[agent_workflow.py:224](data_insights/workflows/agent_workflow.py#L224) imports `CLARIFICATION_CONTEXT_TEMPLATE` inside `call_model()`. Python caches imports after the first execution, so the performance impact is zero after the first call. Unusual style but not a real problem.

---

## Missing: Correctness issues not in the code

### 1. User sends new message during pending clarification — new message is silently ignored

**Verified behavior at [clarification_gate_node:196-201](data_insights/workflows/clarification_gate.py#L196-L201):**

```python
if state.get("clarification_pending"):
    if state.get("clarification_answer"):
        return {}  # routes to resolver
    return {}  # already waiting, no answer yet — don't re-evaluate
```

When the user ignores the clarification panel and types a new query:
1. The new `HumanMessage` is appended to `messages` by LangGraph's `add` reducer
2. `clarification_gate_node()` hits line 196, sees `clarification_pending=True`, and returns `{}` without evaluating the new message
3. `should_clarify()` at [line 253](data_insights/workflows/clarification_gate.py#L253) sees `clarification_pending=True, clarification_answer=None` → routes to `emit_clarification`
4. The original clarification question is re-emitted; the new user message sits in the conversation history unaddressed

When the user eventually answers the clarification, the resolver treats the answer as a response to the *original* ambiguity — the unaddressed new message remains in the conversation history and may confuse the agent.

**Fix:** Either (a) clear `clarification_pending` and re-evaluate from the new message when a new message arrives during pending, or (b) reject the new message with a "please answer the clarification first" response. Option (a) is simpler: detect that there's a new message beyond the one that triggered the original clarification, and reset the pending state.

### 2. No test coverage for any gate function

Zero tests exist for `_needs_entity()`, `_has_entity_match()`, `_match_analysis_types()`, `_needs_time_range()`, `_has_ranking_intent()`, `_has_explicit_metric()`, or the gate node itself. Every fix proposed in this document changes behavior in untested code. The rollout risk is that a fix improves one case but silently breaks another, and no test catches it.

**Fix:** Write at minimum:
- Unit tests for each helper function with representative queries (positive and negative cases)
- Integration tests for the `clarification_gate_node` with a mock state dict
- Use the actual entity names from the development database

### 3. Signal priority ordering is unexamined

`SIGNAL_PRIORITY_ORDER` at [line 56-62](data_insights/workflows/clarification_gate.py#L56-L62):

1. `missing_entity` — "without an entity, query scope is undefined"
2. `vague_ranking_metric` — "wrong metric = completely wrong answer"
3. `multi_analysis_type` — "wrong analysis path = wrong chart type"
4. `missing_time_range` — "defaults are acceptable but confirmation is better"
5. `ambiguous_temporal_word` — "has a reasonable default (30 days)"

No production data validates this ordering. If `vague_ranking_metric` fires more often and more usefully than `missing_entity` in practice, its lower priority means users disproportionately get entity questions when they needed metric questions.

**Fix:** Instrument each signal's firing rate in production logs. After a week of data, validate whether the priority order matches actual query patterns. This is a low-effort, high-value improvement that requires no code changes — just add logging.

### 4. No observability

The only log line for the gate is at [line 492-496](data_insights/workflows/clarification_gate.py#L492-L496) — a single `logger.info` when a clarification is emitted. There is no logging for:
- Which signal fired (and which query triggered it)
- How many clarifications are answered vs abandoned
- What answer the user gave
- Whether the subsequent agent response succeeded

Without this, tuning the gate (adding/removing keywords, changing priority) is guesswork.

### 5. i18n — all keywords are English-only

Every keyword set — `ENTITY_HINT_WORDS`, `RANKING_WORDS`, `TEMPORAL_VAGUE_WORDS`, `_match_analysis_types()` keywords, and `_auto_detect_analysis()` keywords — is English-only. For a product deployed in Ghana with a multilingual user base, the heuristic gate will systematically under-fire for non-English queries. This is a product limitation, not a code bug, but it should be explicit.

### 6. `_match_analysis_types()` fallback hides zero-match queries

[clarification_gate.py:85-86](data_insights/workflows/clarification_gate.py#L85-L86):

```python
if not matched:
    matched.append("overview")
```

A query with zero analysis type keyword matches defaults to `["overview"]` (length 1), so `multi_analysis_type` never fires for it. A query like `"help me understand the data"` silently routes to overview analysis instead of asking "what kind of analysis?" This may be intentional — overview is a reasonable default for truly vague queries — but the behavior should be acknowledged rather than assumed correct.

---

## Prioritized fix list

| Priority | Issue | File(s) | Effort | Risk |
|----------|-------|---------|--------|------|
| 1 | `_has_entity_match()` substring matching | `clarification_gate.py:118-124` | Small | Medium — changes firing behavior for all entity queries |
| 2 | `_needs_entity()` fires on prepositions + hardcoded cities | `clarification_gate.py:50-53, 113-115` | Small | Medium — changes firing behavior (Flaw 1 partially mitigates this) |
| 3a | Path C: `_auto_detect_analysis()` can override gate resolution | `tools.py:709, 734-774` | Medium | Low — additive change, gate resolution is opt-in |
| 3b | Path B: LLM prompt asks clarifying questions the gate handles | `prompt.py:61-62` | Trivial | Low — probabilistic, not deterministic |
| 4 | `_pending_clarifications` module dict breaks with multiple workers | `clarification_gate.py:470` + `agent_workflow.py:32-51` | Medium | Low — additive, stores in existing infrastructure |
| 5 | `vague_ranking_metric` false-positives on "most"/"least" | `clarification_gate.py:28-31` | Small | Medium — changes which queries trigger ranking signal |
| 6 | Dead lambda for `multi_analysis_type` | `clarification_gate.py:216-226` | Trivial | Low — behavior-preserving refactor |
| — | New message silently ignored during pending clarification | `clarification_gate.py:196-201` | Small | Low — edge case, rarely triggered |
| — | No tests | `tests/` (new) | Large | N/A — reduces risk of all other changes |
| — | No observability | `clarification_gate.py` | Small | N/A — additive |
| — | Signal priority unvalidated | `clarification_gate.py:56-62` | Small | N/A — data collection, no code change |

---

## Summary

The pipeline is well-designed: a heuristic gate before tool dispatch is the right architecture for structured disambiguation. The core problems are:

1. **`_has_entity_match()` substring matching** — the most impactful concrete bug. Short entity names like "Ho" match nearly every English query, silently suppressing entity questions when they're needed.
2. **`_needs_entity()` fires on prepositions** — common words like "in" and "from" trigger entity questions on temporal or metric queries. Combined with #1 (which suppresses the signal when entity names *are* present), the entity detection system fires when it shouldn't and stays silent when it should.
3. **`_auto_detect_analysis()` is a silent router** — it makes consequential decisions with zero user visibility and no coordination with the gate. When the user explicitly picks an analysis type, the tool can silently pick a different one. This needs a `force_analysis_type` parameter that defers to the gate.
4. **`_pending_clarifications` is per-process memory** — correct in single-worker dev, broken in multi-worker production where consecutive requests may hit different workers. The PostgresSaver checkpointer (already in use) is the right storage layer.
