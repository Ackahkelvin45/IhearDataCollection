# Proactive Clarification & Intent Elicitation — `data_insights` Agent

## Purpose

This document specifies the design, engineering, and implementation details for adding a **proactive clarification system** to the `data_insights` NL→data pipeline. Claude Code should treat this as the single source of truth when implementing this feature.

The goal is to intercept ambiguous user queries before tool dispatch, ask one targeted clarifying question, and collect a structured (or free-text) answer that allows the agent to route accurately and return the right data the first time.

---

## Background & Motivation

The existing pipeline has two failure modes that this system directly fixes:

1. **Forced single-classification** — `AudioAnalysisInput.analysis_type` forces the LLM to pick one enum value (e.g., `"correlation"` or `"temporal"`) even when a query spans multiple dimensions. Information is lost and the wrong tool path fires.

2. **Silent mis-routing** — Vague queries like "show me the worst recordings" or "recent trends" get confidently routed to a tool that returns the wrong data. The user gets an answer, but not the one they wanted.

A clarification gate solves both by pausing before dispatch when ambiguity is detected, asking the user one question, and re-routing with a fully specified intent.

---

## Architecture Overview

### New Node in LangGraph Graph

Insert a `clarification_gate` node between message receipt and tool dispatch:

```
user_message
     │
     ▼
[cleanup_expired_handles]   ← existing node
     │
     ▼
[clarification_gate]        ← NEW NODE
     │
     ├── intent is clear ──────────────────────► [agent] → [tools] → [post_process]
     │
     └── intent is ambiguous ──► emit clarification payload to frontend
                                         │
                                 user answers (option OR custom text)
                                         │
                                         ▼
                                 [clarification_resolver]  ← NEW NODE
                                         │
                                         ▼
                                 [agent] → [tools] → [post_process]
```

### State additions to `AgentState`

```python
# agent_workflow.py

class AgentState(TypedDict):                    # also fix Dict → TypedDict here
    messages: Annotated[List[AnyMessage], add]
    # --- existing fields ---
    user_id: str
    mode: str
    ai_answer: bool
    # --- new fields ---
    clarification_pending: bool                 # True when waiting for user answer
    clarification_question: Optional[str]       # The question text emitted to frontend
    clarification_options: Optional[List[str]]  # Structured choices offered
    clarification_answer: Optional[str]         # User's resolved answer (after response)
    clarification_is_custom: bool               # True if user typed free text
    clarification_dimension: Optional[str]      # Which dimension was ambiguous: "time_range" | "metric" | "entity" | "analysis_type"
    original_query: Optional[str]               # Preserved raw query for re-routing
```

---

## Clarification Gate — Detection Logic

**Location:** `workflows/clarification_gate.py` (new file)

The gate uses **heuristic signal detection first**, only calling the LLM when a signal fires. This keeps the happy path (clear queries) zero-latency.

### Ambiguity Signals

```python
AMBIGUITY_SIGNALS = {

    "multi_analysis_type": {
        "description": "Query maps to more than one analysis_type bucket",
        "detection": lambda query, context: len(_match_analysis_types(query)) > 1,
        "dimension": "analysis_type",
        "question": "What kind of analysis do you want?",
        "options_fn": lambda matched_types: [ANALYSIS_TYPE_LABELS[t] for t in matched_types] + ["All of the above"],
    },

    "missing_time_range": {
        "description": "Query implies time-based analysis but no time reference found",
        "detection": lambda query, context: _needs_time_range(query) and not _has_time_reference(query),
        "dimension": "time_range",
        "question": "What time range should I look at?",
        "options": ["Last 7 days", "Last 30 days", "Last 3 months", "Last year", "All time"],
    },

    "vague_ranking_metric": {
        "description": "User asks for 'best', 'worst', 'top', 'lowest' without specifying by what",
        "detection": lambda query, context: _has_ranking_intent(query) and not _has_explicit_metric(query),
        "dimension": "metric",
        "question": "Rank by which metric?",
        "options": ["Decibel level (dB)", "RMS energy", "Recording count", "Spectral centroid", "Complaint rate"],
    },

    "missing_entity": {
        "description": "Query implies filtering by region/category/community but none found",
        "detection": lambda query, context: _needs_entity(query) and not _has_entity_match(query, context),
        "dimension": "entity",
        "question": "Which area or category should I focus on?",
        "options_fn": lambda context: context.get("available_regions", [])[:4] + ["All areas"],
    },

    "ambiguous_temporal_word": {
        "description": "Query contains 'recent', 'latest', 'new' with no baseline",
        "detection": lambda query, context: bool(re.search(r'\b(recent|latest|new|current)\b', query, re.I)),
        "dimension": "time_range",
        "question": "How far back should 'recent' go?",
        "options": ["Last 7 days", "Last 30 days", "Last 3 months"],
    },
}
```

### Signal Priority

When multiple signals fire, pick only the **highest-priority** one and ask one question. Priority order:

1. `missing_entity` — without an entity, query scope is undefined
2. `vague_ranking_metric` — wrong metric = completely wrong answer
3. `multi_analysis_type` — wrong analysis path = wrong chart type
4. `missing_time_range` — defaults are acceptable but confirmation is better
5. `ambiguous_temporal_word` — lowest priority, has a reasonable default (30 days)

### Gate Node Implementation

```python
# workflows/clarification_gate.py

def clarification_gate_node(state: AgentState) -> AgentState:
    """
    Evaluates the latest user message for ambiguity signals.
    If ambiguous: sets clarification_pending=True and emits clarification payload.
    If clear: passes through unchanged.
    """
    if state.get("clarification_pending"):
        # Already waiting — this message is the answer, not a new query
        return state

    last_message = state["messages"][-1]
    query = last_message.content
    context = _build_entity_context()  # fetch available regions/categories from DB

    fired_signal = None
    for signal_name in SIGNAL_PRIORITY_ORDER:
        signal = AMBIGUITY_SIGNALS[signal_name]
        if signal["detection"](query, context):
            fired_signal = (signal_name, signal)
            break

    if fired_signal is None:
        # Clear intent — pass straight through
        return {**state, "clarification_pending": False}

    signal_name, signal = fired_signal

    # Build options list
    if "options_fn" in signal:
        options = signal["options_fn"](context)
    else:
        options = signal["options"]

    return {
        **state,
        "clarification_pending": True,
        "clarification_question": signal["question"],
        "clarification_options": options,
        "clarification_dimension": signal["dimension"],
        "original_query": query,
    }


def should_clarify(state: AgentState) -> str:
    """LangGraph conditional edge function."""
    if state.get("clarification_pending") and not state.get("clarification_answer"):
        return "emit_clarification"
    return "agent"
```

---

## Frontend Payload Contract

### Outbound — Agent → Frontend

When `clarification_pending=True`, emit this as a streaming chunk **before** the spinner resolves:

```json
{
  "type": "clarification",
  "question": "What time range should I look at?",
  "options": [
    "Last 7 days",
    "Last 30 days",
    "Last 3 months",
    "Last year",
    "All time"
  ],
  "allow_custom": true,
  "custom_placeholder": "e.g. Jan 2024 to March 2024",
  "dimension": "time_range"
}
```

**All clarification payloads must include `allow_custom: true`.** There are no exceptions. Users must always be able to type a free-text answer if none of the structured options fit.

### Inbound — Frontend → Agent (structured pick)

```json
{
  "type": "clarification_response",
  "dimension": "time_range",
  "answer": "Last 30 days",
  "is_custom": false
}
```

### Inbound — Frontend → Agent (custom free text)

```json
{
  "type": "clarification_response",
  "dimension": "time_range",
  "answer": "between the rainy season and December",
  "is_custom": true
}
```

---

## Frontend UI Behaviour

The frontend renders the clarification payload as:

```
┌─────────────────────────────────────────────────────┐
│  What time range should I look at?                   │
│                                                      │
│  [ Last 7 days ]  [ Last 30 days ]  [ Last 3 months ]│
│  [ Last year   ]  [ All time     ]                   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │ Other: e.g. Jan 2024 to March 2024           │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

Rules:
- Option buttons are **mutually exclusive** — selecting one deselects all others
- Typing in the "Other" field **deselects** any active button
- Selecting a button **clears** the Other field
- Submission is triggered by button click OR Enter key in the Other field
- The UI is disabled after submission (no re-answering mid-stream)

---

## Clarification Resolver Node

**Location:** `workflows/clarification_gate.py`

After the user responds, this node normalises the answer — especially for custom free-text — before passing to the agent.

```python
def clarification_resolver_node(state: AgentState) -> AgentState:
    """
    Receives the clarification answer from the frontend.
    - Structured pick: maps answer directly to tool parameter.
    - Custom text: runs targeted NLP extraction for that dimension only.
    Appends a synthetic context message to state["messages"] so the agent
    sees the resolved answer as part of conversation history.
    """
    answer = state["clarification_answer"]
    is_custom = state["clarification_is_custom"]
    dimension = state["clarification_dimension"]
    original_query = state["original_query"]

    if is_custom:
        resolved = _normalize_custom_answer(answer, dimension)
        # Ask for confirmation before proceeding (see: Confirmation Step below)
        confirmation_needed = True
    else:
        resolved = _map_structured_answer(answer, dimension)
        confirmation_needed = False

    # Inject resolved context into message history as a HumanMessage
    context_injection = f"[Clarification resolved] {dimension}: {resolved}"
    enriched_messages = state["messages"] + [HumanMessage(content=context_injection)]

    return {
        **state,
        "messages": enriched_messages,
        "clarification_pending": False,
        "clarification_answer": resolved,
        "confirmation_needed": confirmation_needed,
        "confirmation_resolved_value": resolved if confirmation_needed else None,
    }
```

### `_normalize_custom_answer` — Dimension-Specific Extractors

Each dimension has its own normalisation function. These are lightweight and do **not** make an LLM call:

```python
def _normalize_custom_answer(answer: str, dimension: str) -> str:
    if dimension == "time_range":
        return _parse_natural_date_range(answer)        # returns ISO date range string
    elif dimension == "metric":
        return _fuzzy_match_metric(answer)              # matches to known metric names
    elif dimension == "entity":
        return _fuzzy_match_entity(answer)              # matches to DB region/category
    elif dimension == "analysis_type":
        return _classify_analysis_type(answer)          # maps to enum value
    else:
        return answer                                   # pass through unknown dimensions
```

For `time_range`, use `dateparser` library as the backbone. For `metric` and `entity`, use edit-distance matching against known values. Only fall back to an LLM extraction call if fuzzy matching confidence is below 0.7.

---

## Confirmation Step (Custom Answers Only)

When `confirmation_needed=True`, before running the query, emit a confirmation message to the frontend:

```json
{
  "type": "confirmation",
  "message": "Interpreting that as October 1 – December 31, 2024 — does that look right?",
  "resolved_value": "2024-10-01 to 2024-12-31",
  "dimension": "time_range"
}
```

Frontend renders:

```
┌─────────────────────────────────────────────────────────┐
│  Interpreting that as Oct 1 – Dec 31, 2024              │
│  — does that look right?                                 │
│                                                          │
│  [ Yes, proceed ]          [ No, let me rephrase ]       │
└─────────────────────────────────────────────────────────┘
```

If the user picks "No, let me rephrase", clear `clarification_answer` and re-emit the original clarification payload (loop back to the question). This prevents silent mis-parsing.

**Do not show the confirmation step for structured picks.** The user selected a labelled button — no interpretation ambiguity exists.

---

## Acknowledgement Before Query Execution

After the clarification answer is confirmed (or for structured picks, immediately after selection), emit an inline acknowledgement before the data fetch begins:

**Structured pick:**
> "Got it — looking at the last 30 days. Fetching decibel trends by region..."

**Custom pick (after confirmation):**
> "Looking at Oct 1 – Dec 31 2024. Fetching decibel trends by region..."

This closes the interaction loop and signals to the user that their input was understood.

---

## Rules & Constraints

### One Question Per Turn — Hard Rule

Never ask more than one clarifying question per user message. If multiple ambiguity signals fire, resolve only the **highest-priority** one (see Signal Priority above). Subsequent ambiguities, if any remain after the first answer, can be resolved in the next turn.

Rationale: Multiple questions feel like a form. One question feels like a conversation. User drop-off increases sharply past one question.

### Never Block Clear Queries

If zero ambiguity signals fire, the clarification gate must be completely transparent — no latency, no extra round-trip, no message to the user. The gate is an interceptor, not a mandatory step.

### Always Allow Custom Input

Every clarification question must include the free-text "Other" input. No exceptions. Users always know their domain better than the options list does. Forcing a pick from a finite list when none fit creates frustration and incorrect results.

### Preserve Original Query

Always store the original user query in `original_query` before enriching with clarification context. The agent uses both: the original query for intent and phrasing, the clarification answer for parameters.

### Maximum One Confirmation Per Message Thread

If the user has already gone through a clarification + confirmation loop for the current query, do not trigger another confirmation even if a subsequent custom answer is parsed. Trust the user at that point.

---

## Integration Points

### `views.py` — Streaming

In the `stream()` function, add handling for the two new chunk types:

```python
# views.py — inside the stream generator

if chunk_type == "clarification":
    yield json.dumps({
        "type": "clarification",
        "question": chunk["question"],
        "options": chunk["options"],
        "allow_custom": True,
        "custom_placeholder": chunk.get("placeholder", "Type your answer..."),
        "dimension": chunk["dimension"],
    }) + "\n"

elif chunk_type == "confirmation":
    yield json.dumps({
        "type": "confirmation",
        "message": chunk["message"],
        "resolved_value": chunk["resolved_value"],
        "dimension": chunk["dimension"],
    }) + "\n"
```

### `agent_workflow.py` — Graph Wiring

```python
# Add nodes
graph.add_node("clarification_gate", clarification_gate_node)
graph.add_node("clarification_resolver", clarification_resolver_node)
graph.add_node("emit_clarification", emit_clarification_node)

# Rewire entry
graph.set_entry_point("cleanup")
graph.add_edge("cleanup", "clarification_gate")

# Conditional: clear → agent, ambiguous → emit
graph.add_conditional_edges(
    "clarification_gate",
    should_clarify,
    {
        "agent": "agent",
        "emit_clarification": "emit_clarification",
    }
)

# Resolver feeds back into agent
graph.add_edge("clarification_resolver", "agent")
```

### `prompt.py` — System Template Injection

Inject clarification context into the system prompt so the agent is aware of what was resolved:

```python
CLARIFICATION_CONTEXT_TEMPLATE = """
The user's query has been clarified. The following was confirmed before this request:
- Dimension: {dimension}
- Resolved value: {resolved_value}

Use this resolved value as a hard constraint when selecting parameters for your tool call.
Do not second-guess or re-interpret this value.
"""
```

Append this to `SYSTEM_TEMPLATE` only when `clarification_answer` is present in state.

---

## Files to Create / Modify

| Action | File | Change |
|--------|------|--------|
| **Create** | `workflows/clarification_gate.py` | Gate node, resolver node, signal definitions, normalizers |
| **Modify** | `workflows/agent_workflow.py` | Add nodes, rewire edges, fix `AgentState(Dict)` → `TypedDict` |
| **Modify** | `workflows/prompt.py` | Add `CLARIFICATION_CONTEXT_TEMPLATE`, inject when relevant |
| **Modify** | `views.py` | Handle `clarification` and `confirmation` chunk types in stream |
| **Modify** | `frontend/` | Render clarification UI — buttons + Other input + confirmation step |

---

## Testing Checklist

- [ ] Clear query bypasses gate with zero added latency
- [ ] "show me recent trends" fires `ambiguous_temporal_word` signal
- [ ] "show me the worst recordings" fires `vague_ranking_metric` signal
- [ ] "show me energy and frequency for last month" fires `multi_analysis_type` signal
- [ ] Structured pick routes directly to tool without confirmation
- [ ] Custom text answer triggers confirmation step before query runs
- [ ] "No, let me rephrase" on confirmation loops back to the original question
- [ ] Acknowledgement message appears before data fetch begins
- [ ] SQL fallback path also benefits from clarification (resolved answer is in messages history)
- [ ] Multiple ambiguity signals → only highest-priority question is asked
- [ ] Empty Other field + no button selected → submit is disabled (frontend validation)
