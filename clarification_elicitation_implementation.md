# Proactive Clarification & Intent Elicitation — Implementation Notes

## Overview

Added a **proactive clarification system** to the `data_insights` NL-to-data pipeline. When a user's query is ambiguous, the agent pauses before dispatching tools, asks one targeted clarifying question, and re-routes with a fully specified intent after the user answers.

## Architecture

```
user_message
     │
     ▼
[cleanup_expired_handles]
     │
     ▼
[clarification_gate]  ← NEW
     │
     ├── intent is clear ─────► [agent] → [tools] → [post_process] → [format]
     │
     └── intent is ambiguous ──► [emit_clarification] → (graph ends, chunk sent to frontend)
                                         │
                                 user answers (option or custom text)
                                         │
                                         ▼
                                 [clarification_resolver]  ← NEW
                                         │
                                         ▼
                                 [agent] → [tools] → [post_process] → [format]
```

## Files Changed

| File | Action | Purpose |
|------|--------|---------|
| `data_insights/workflows/clarification_gate.py` | **Created** | Gate node, resolver node, 5 ambiguity signals, dimension-specific normalizers |
| `data_insights/workflows/agent_workflow.py` | Modified | 9 new AgentState fields, 3 new graph nodes, conditional edges rewired, `process_clarification_response()` method added, clarification context injected into system prompt |
| `data_insights/workflows/prompt.py` | Modified | Added `CLARIFICATION_CONTEXT_TEMPLATE` |
| `data_insights/views.py` | Modified | Clarification state check in stream loop, `clarification` chunk emission, `clarify_message` REST action, `_process_clarification_response()` handler |
| `data_insights/models.py` | Modified | Added `CLARIFICATION_PENDING` to `MessageStatus` choices |
| `data_insights/templates/data_insights/unified_chat.html` | Modified | `showClarificationUI()`, `showConfirmationUI()`, `sendClarificationResponse()`, stream handler updates for `clarification`/`confirmation`/`completed` actions |

## Ambiguity Signals

Five signals are checked in priority order. Only the **highest-priority** signal fires — one question per turn, hard rule.

| Priority | Signal | Trigger | Question |
|----------|--------|---------|----------|
| 1 | `missing_entity` | Query implies region/category filter but none specified | "Which area or category should I focus on?" |
| 2 | `vague_ranking_metric` | Ranking words ("best", "worst", "top") without a metric | "Rank by which metric?" |
| 3 | `multi_analysis_type` | Query maps to 2+ analysis_type buckets | "What kind of analysis do you want?" |
| 4 | `missing_time_range` | Time-based query with no date reference | "What time range should I look at?" |
| 5 | `ambiguous_temporal_word` | Contains "recent", "latest", "new" without baseline | "How far back should 'recent' go?" |

## Frontend Payload Contract

### Agent → Frontend (clarification chunk)

```json
{
  "type": "clarification",
  "question": "What time range should I look at?",
  "options": ["Last 7 days", "Last 30 days", "Last 3 months", "Last year", "All time"],
  "allow_custom": true,
  "custom_placeholder": "e.g. Jan 2024 to March 2024",
  "dimension": "time_range"
}
```

### Frontend → Agent (clarification response)

```json
{
  "answer": "Last 30 days",
  "is_custom": false
}
```

### Agent → Frontend (confirmation, for custom text)

```json
{
  "type": "confirmation",
  "message": "Interpreting that as Oct 1 – Dec 31, 2024 — does that look right?",
  "resolved_value": "2024-10-01 to 2024-12-31",
  "dimension": "time_range"
}
```

## Frontend UI Behavior

- Option buttons are **mutually exclusive** — selecting one deselects others
- Typing in the "Other" field **deselects** any active button
- Selecting a button **clears** the Other field
- Submit is disabled until something is selected or typed
- Enter key in the Other field triggers submit
- UI is disabled after submission (no re-answering mid-stream)
- "No, let me rephrase" on confirmation re-shows the original question from stored DOM data

## Dimension-Specific Normalizers

| Dimension | Structured mapping | Custom text handling |
|-----------|-------------------|---------------------|
| `time_range` | "Last 30 days" → `last_30_days` | `dateutil` parsing of relative ranges ("last 2 weeks") and explicit dates ("Jan to March 2024") |
| `metric` | "Decibel level (dB)" → `mean_db` | Edit-distance matching via `difflib` against known metric names |
| `entity` | "All areas" → `all` | DB lookup + edit-distance matching against Region/Community names |
| `analysis_type` | "Energy analysis..." → `energy` | Keyword matching against `ANALYSIS_TYPE_LABELS` |

## AgentState Additions

```python
clarification_pending: bool           # True when waiting for user answer
clarification_question: Optional[str] # The question text emitted to frontend
clarification_options: Optional[List[str]]  # Structured choices offered
clarification_answer: Optional[str]   # User's resolved answer
clarification_is_custom: bool         # True if user typed free text
clarification_dimension: Optional[str] # Which dimension was ambiguous
original_query: Optional[str]         # Preserved raw query for re-routing
confirmation_needed: bool             # True when custom answer needs confirmation
confirmation_resolved_value: Optional[str]  # The parsed value to confirm
```

## End-to-End Flow Example

1. User types: **"show me recent trends"**
2. Gate fires `ambiguous_temporal_word` signal
3. Stream yields clarification chunk, message saved with `CLARIFICATION_PENDING` status
4. Frontend renders: *"How far back should 'recent' go?"* with buttons [Last 7 days] [Last 30 days] [Last 3 months] and an "Other" input
5. User clicks **"Last 30 days"**
6. Frontend POSTs to `/messages/{id}/clarify/` with `{"answer": "Last 30 days", "is_custom": false}`
7. Backend runs `process_clarification_response()`, graph routes through `clarification_resolver`
8. Resolver maps "Last 30 days" → `last_30_days`, injects context into message history
9. Agent sees: `[Clarification resolved] dimension=time_range resolved_value=last_30_days`
10. Agent routes to the right tool with the time constraint, streams back results

## Key Design Decisions

- **Heuristics first, no LLM cost on clear queries**: Detection runs regex/keyword checks; the LLM is never called unless a signal fires. Happy path has zero added latency.
- **One question per turn**: Multiple signals → only highest-priority fires. Asking one question feels conversational; multiple feels like a form.
- **Always allow custom input**: Every clarification includes a free-text "Other" field. Users know their domain better than any option list.
- **Graph terminates at clarification, resumes on answer**: The `emit_clarification` node goes to `__end__`. The checkpointer preserves state. The second invocation with `clarification_answer` set routes through `clarification_resolver` → `agent`.
- **Confirmation step deferred**: The resolver skeleton supports `confirmation_needed=True` for custom text, but it's set to `False` for now. The frontend confirmation UI is ready but won't trigger until the backend enables it.

## Edge Cases Handled

- **Clear queries**: Gate is transparent — no latency, no message, no round-trip
- **Multiple ambiguity signals**: Only highest-priority fires
- **Empty Other field + no button**: Submit button is disabled (frontend validation)
- **New message while clarification pending**: `create_initial_state` resets all clarification fields to defaults; `TypedDict` overwrite reducer ensures stale state doesn't leak
- **Client disconnect mid-clarification**: `GeneratorExit` handler saves current state
- **Confirmation "No, rephrase"**: Re-shows original clarification UI from stored DOM data without a server round-trip
