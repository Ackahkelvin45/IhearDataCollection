# Data Insights — Smart Dashboard & Analysis Upgrade v2

## 1. Target Outcome

When a user asks "analyze decibel across regions and communities, give me a dashboard," the system should:

1. Retrieve the data (current: works)
2. Build a **multi-widget dashboard** with stat cards, charts, and a highlight section (current: single chart + table)
3. Append **2-3 sharp observations** generated from the data (current: not present)
4. The LLM text should **summarize, not re-list** — 2-4 sentences highlighting the most important finding (current: dumps data)
5. When two related analyses run (region + community), the dashboard should **combine** both cleanly (current: second overwrites first)

---

## 2. Current Architecture (for context)

```
User query
  → _process_message_sync() in views.py
    → LangGraph agent graph:
        cleanup → clarification_gate → agent (LLM+tools) → tools → post_process_tools → agent (loop) → format → end
    → view reads pending_chart / pending_artifact from final state
    → emits SSE events: "visualization" / "dashboard"
  → Frontend renderArtifact() renders widgets
```

**Dashboard pipeline inside post_process_tools:**
```
Tool result (rows, columns, chart_hint, analysis_type)
  → if analysis_type in ("overview_analysis", "temporal_analysis", "ml_dataset_profile"):
      decompose(result) → multi-widget artifact
    else:
      resolve_chart(result) → single chart → wrap_as_artifact()
  → write {pending_chart, pending_artifact} to state
```

**Current multi-widget coverage:** 3 types out of 16+

---

## 3. Changes

### Change A — `finalize_dashboard` node (collapses old Changes 1, 5, 7)

**Problem:** Three separate mechanisms fighting each other:
- Change 1 (generate_insights inside agent loop) fires N times, sees partial data, wastes tokens
- Change 5 (regex trimming in format_response) strips content the LLM was asked to produce
- Change 7 (system prompt telling LLM to summarize) is a soft suggestion the LLM ignores under tool pressure

**Solution:** One structured-output LLM call in a post-loop node.

**New graph structure:**
```
cleanup → clarification_gate → agent (LLM+tools) → tools → post_process_tools → agent (loop)
  → finalize_dashboard → format_response → end
```

`finalize_dashboard` runs **once**, after the agent loop terminates, when the complete artifact is available.

```python
def finalize_dashboard(self, state: AgentState) -> Dict[str, Any]:
    """Generate summary + observations from the complete artifact.
    Runs once after the agent loop, not inside it."""
    artifact = state.get("pending_artifact")
    if not artifact or not _artifact_has_data(artifact):
        return {}

    summary_input = _build_data_summary(artifact)

    # Guard: skip replacement for trivial data (single row, no chart)
    if summary_input.get("widget_count", 0) <= 1 or summary_input.get("total_rows", 0) <= 1:
        return {}  # let the agent's original short answer stand

    start = time.monotonic()
    try:
        response = self.dashboard_llm.invoke([
            ("system", _DASHBOARD_SUMMARY_SYSTEM_PROMPT),
            ("human", f"Data summary:\n{json.dumps(summary_input, indent=2)}"),
        ])
        elapsed = int((time.monotonic() - start) * 1000)
        logger.info("finalize_dashboard llm_ms=%d widget_count=%d rows=%d",
                    elapsed, len(artifact.get("widgets", [])), summary_input.get("total_rows", 0))

        parsed = json.loads(response.content)

        # Append observations as insight widget
        observations = (parsed.get("observations") or [])[:3]
        if observations:
            artifact["widgets"].append({
                "id": "insights",
                "type": "insight",
                "title": "Key Observations",
                "data": {"observations": observations},
                "priority": 999,
            })

        # Replace agent's last message content with dashboard-aware summary
        summary_text = (parsed.get("summary") or "").strip()
        if summary_text:
            last_msg = state["messages"][-1]
            # Only replace if terminal text message (no tool_calls, string content)
            if isinstance(last_msg, AIMessage) and not getattr(last_msg, "tool_calls", None) and isinstance(last_msg.content, str):
                state["messages"][-1] = AIMessage(content=summary_text, id=last_msg.id)

        return {"pending_artifact": artifact}

    except Exception as e:
        logger.exception("finalize_dashboard LLM call failed: %s", e)
        return {}  # degraded: keep original message, skip insight widget
}
```

**Structured output contract** (enforced via JSON mode in the prompt):

```json
{
  "summary": "2-4 sentences. The single most important finding, one notable pattern or outlier. Do NOT list data row by row — the dashboard handles that. End with an invitation to drill deeper.",
  "observations": [
    "One surprising finding — 1 sentence.",
    "One notable pattern — 1 sentence.",
    "One thing worth investigating further — 1 sentence."
  ]
}
```

**Three guards in the message replacement:**
1. Only replace when last message has no `tool_calls` (not a terminal tool-call message)
2. Only replace when `content` is a string (not a list of content blocks)
3. Skip entirely for trivial data (single row, no chart) — the agent's short answer is sufficient

**format_response update** (still runs after finalize_dashboard):

```python
def format_response(self, state: AgentState) -> Dict[str, Any]:
    pending_artifact = state.get("pending_artifact")
    last_message = state["messages"][-1]
    if pending_artifact and isinstance(last_message, AIMessage):
        content = last_message.content or ""
        # Only append dashboard pointer if the summary doesn't already mention it
        if "dashboard" not in content.lower() and "chart" not in content.lower():
            content = content.rstrip() + "\n\n*See the dashboard below for the full breakdown.*"
            return {"messages": [AIMessage(content=content)]}
    return {}
```

**What no longer exists:**
- `generate_insights` node inside the agent loop — removed
- Regex stripping of markdown tables — obviated (the LLM was never asked to produce tables)
- System prompt instruction about dashboard-aware writing — obviated (finalize_dashboard's prompt is always dashboard-aware)

**LLM model:** Uses a second ChatOpenAI instance configured from `AGENT_CONFIG` — same source, no hardcoded model name.

---

### Change B — Expand multi-widget decomposition

**File:** `data_insights/workflows/widget_composer.py`

Add **shared helpers** first, then four new decomposition functions, then update `decompose()` routing.

#### Shared helpers

```python
def _build_stat_widget(widget_id: str, title: str, stats: Dict[str, Any], priority: int = 0) -> Dict:
    return {"id": widget_id, "type": "stat_card", "title": title,
            "data": {"stats": stats}, "priority": priority}

def _build_table_widget(widget_id: str, title: str, rows: List[Dict], priority: int) -> Dict:
    return {"id": widget_id, "type": "table", "title": title, "priority": priority,
            "data": {"table": {"columns": list(rows[0].keys()) if rows else [],
                               "rows": rows, "title": title}}}
```

#### 2a. `_decompose_grouped(result)` — for `avg_decibel_by_*`, `group_count`

**Input:** result with `rows`, `columns`, `chart_hint` (x, y), `analysis_type`, `row_count`.

**Output:** stat_card (summary stats), chart (horizontal_bar for >5, else bar), ranking_highlight (top 3 + bottom 3), table.

```python
def _decompose_grouped(result: Dict[str, Any]) -> Dict[str, Any]:
    rows = result["rows"]
    hint = result.get("chart_hint", {})
    y_key = hint.get("y", "")
    n = len(rows)
    analysis_type = result.get("analysis_type", "")
    group_label = _humanise(hint.get("x", ""))

    # Stat card: aggregate
    values = [r.get(y_key, 0) if isinstance(r.get(y_key), (int, float)) else 0 for r in rows]
    if values:
        _avg = round(sum(values) / len(values), 2)
        _max = round(max(values), 2)
        _min = round(min(values), 2)
    else:
        _avg = _max = _min = 0
    stats = {
        f"Total {group_label}s": n,
        f"Average {_humanise(y_key)}": _avg,
        f"Highest {_humanise(y_key)}": _max,
        f"Lowest {_humanise(y_key)}": _min,
    }
    widgets = [_build_stat_widget(f"stats_{analysis_type}", "Summary Statistics", stats, 0)]

    # Chart
    chart_type = "horizontal_bar_chart" if n > 5 else "bar_chart"
    chart_widget = _rows_to_widget(rows, f"chart_{analysis_type}",
                                    f"{_humanise(y_key)} by {group_label}",
                                    chart_type, 1,
                                    x_key=hint.get("x"), y_key=hint.get("y"))
    if chart_widget:
        widgets.append(chart_widget)

    # Ranking: top 3 + bottom 3 (dedup for small N)
    sorted_rows = sorted(rows, key=lambda r: r.get(y_key, 0) if isinstance(r.get(y_key), (int, float)) else 0, reverse=True)
    top3 = sorted_rows[:3]
    bottom3 = sorted_rows[-3:] if n > 6 else []  # skip if overlap with top
    rank_items = []
    for i, r in enumerate(top3):
        rank_items.append({"rank": i + 1, "label": str(r.get(hint.get("x", ""), "")),
                           "primary": str(r.get(y_key, "")), "secondary": "", "trend": None})
    offset = max(0, n - 2)
    for i, r in enumerate(bottom3):
        rank_items.append({"rank": offset + i, "label": str(r.get(hint.get("x", ""), "")),
                           "primary": str(r.get(y_key, "")), "secondary": "", "trend": None})
    if rank_items:
        widgets.append({"id": f"ranking_{analysis_type}", "type": "ranking_highlight",
                        "title": "Top & Bottom Performers",
                        "data": {"items": rank_items, "limit": 3, "total": n}, "priority": 2})

    widgets.append(_build_table_widget(f"table_{analysis_type}", "Full Results", rows, 3))
    return {
        "widgets": widgets,
        "layout_template": "grid",
        "version": 1,
        "follow_ups": _get_follow_ups(analysis_type),
    }
```

**Key fix from original plan:** bottom3 is skipped when n <= 6 (overlap with top3). Ranking dedup prevents showing the same row twice. Values guard with type check prevents crash on None/non-numeric.

#### 2b. `_decompose_ranked(result)` — for `highest_decibel`, `lowest_decibel`

1. stat_card (priority 0) — spotlight on #1: name and value
2. chart (priority 1) — horizontal_bar for top 10
3. table (priority 2)

#### 2c. `_decompose_statistical(result)` — for `statistical_distribution`

1. stat_card (priority 0) — per-group min/avg/max
2. chart (priority 1) — box_plot from distribution_data
3. table (priority 2)

#### 2d. `_decompose_recent(result)` — for `recent_datasets`

1. stat_card (priority 0) — total count, date range
2. table (priority 1)

#### Update `decompose()` routing

```python
def decompose(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    analysis_type = result.get("analysis_type", "")
    if analysis_type == "overview_analysis":
        return _decompose_overview(result)
    elif analysis_type == "temporal_analysis":
        return _decompose_temporal(result)
    elif analysis_type == "ml_dataset_profile":
        return _decompose_ml_profile(result)
    elif analysis_type.startswith("avg_decibel_by_") or analysis_type == "group_count":
        return _decompose_grouped(result)
    elif analysis_type in ("highest_decibel", "lowest_decibel"):
        return _decompose_ranked(result)
    elif analysis_type == "statistical_distribution":
        return _decompose_statistical(result)
    elif analysis_type == "recent_datasets":
        return _decompose_recent(result)
    # Fallback: single chart (existing behavior)
    chart = resolve_chart(result)
    return wrap_as_artifact(chart, analysis_type=analysis_type or None)
```

---

### Change C — ChartDecision dataclass + cardinality fix

**File:** `data_insights/workflows/chart_builder.py`

**Problem:** `select_chart_type` returns a raw string. The 13+ cardinality case was handled with a magic `_truncate` dict key — a code smell that leaks chart-builder internals across module boundaries.

**Solution:** Typed `ChartDecision` dataclass.

```python
@dataclass
class TruncateSpec:
    limit: int
    sort_by: str
    sort_order: str  # "asc" | "desc"

@dataclass
class ChartDecision:
    chart_type: str
    truncate: Optional[TruncateSpec] = None
```

**`select_chart_type` now returns a `ChartDecision`:**

```python
def select_chart_type(x_cardinality: int, y_col: str) -> ChartDecision:
    if x_cardinality <= 5:
        return ChartDecision(chart_type="bar_chart")
    if x_cardinality <= 12:
        return ChartDecision(chart_type="horizontal_bar_chart")
    # 13+ categories: show top 12, sorted descending
    return ChartDecision(
        chart_type="horizontal_bar_chart",
        truncate=TruncateSpec(limit=12, sort_by=y_col, sort_order="desc"),
    )
```

**In `build_chart_config`,** apply the decision object:

```python
decision = select_chart_type(x_cardinality, y_col)
if decision.truncate:
    t = decision.truncate
    reverse = t.sort_order == "desc"
    rows = sorted(rows, key=lambda r: r.get(t.sort_by, 0) or 0, reverse=reverse)
    rows = rows[:t.limit]
    total = len(tool_result.get("rows", rows)) if isinstance(tool_result, dict) else len(rows)
    title = f"{_humanise(y_col)} by {_humanise(x_col)} (Top {t.limit} of {total})"
```

No magic dict keys, no underscored internals leaking through function boundaries.

---

### Change D — Artifact merging with merge_group

**File:** `data_insights/workflows/agent_workflow.py`

**Problem:** The string-prefix heuristic in `_are_sibling_analyses` creates a central registry that rots every time a new `analysis_type` is added.

**Solution:** `merge_group` declared at the tool layer, not guessed in the merge function.

#### Tool layer change

Each tool result that can participate in merging returns a `merge_group` field:
- `avg_decibel_by_region`, `avg_decibel_by_community` → `merge_group: "decibel_by_dimension"`
- `highest_decibel`, `lowest_decibel` → `merge_group: "decibel_extremes"`
- `group_count` → `merge_group: None` (not mergeable by default)

The merge check becomes:
```python
existing_mg = existing_meta.get("merge_group")
new_mg = new_meta.get("merge_group")
if existing_mg and existing_mg == new_mg:
    merged = _merge_artifacts(existing_artifact, new_artifact)
```

#### Merge function with global priority re-sort

```python
def _merge_artifacts(existing: Dict, new: Dict) -> Dict:
    existing_widgets = existing.get("widgets", [])
    new_widgets = new.get("widgets", [])

    # Dedup: rename colliding IDs
    existing_ids = {w["id"] for w in existing_widgets}
    for w in new_widgets:
        if w["id"] in existing_ids:
            w["id"] = f"{w['id']}_2"
        existing_ids.add(w["id"])

    # Re-sort ALL widgets globally by priority
    all_widgets = existing_widgets + new_widgets
    all_widgets.sort(key=lambda w: w.get("priority", 0))

    return {
        **existing,
        "widgets": all_widgets,
        "layout_template": _best_layout(len(all_widgets)),
        "follow_ups": existing.get("follow_ups", []) + new.get("follow_ups", []),
    }

def _best_layout(count: int) -> str:
    if count <= 1: return "single"
    if count <= 3: return "two_column"
    return "grid"
```

**Widget ordering policy (explicit):** After merge, all widgets are re-sorted globally by ascending `priority`. So you always get `[stats, stats, chart, chart, table, table]` — grouped by widget type — regardless of merge order.

---

### Change E — New frontend widgets (unchanged from v1)

**File:** `data_insights/templates/data_insights/unified_chat.html`

Add two widget handlers in `renderArtifact()`:

**`insight` widget handler:**
```javascript
if (widget.type === 'insight') {
    const obs = (widget.data.observations || []).map(o =>
        `<div class="insight-item">
            <span class="insight-marker">•</span>
            <span>${this.escapeHtml(o)}</span>
        </div>`
    ).join('');
    widgetDiv.innerHTML = `
        <h4 style="font-weight:600; color:#fff; margin-bottom:0.75rem;">
            ${this.escapeHtml(widget.title)}
        </h4>
        <div class="insight-list">${obs}</div>`;
}
```

**`ranking_highlight` widget handler:**
```javascript
if (widget.type === 'ranking_highlight') {
    const items = (widget.data.items || []).map(item => {
        const rankBadge = item.rank <= 3
            ? `<span class="rank-badge rank-top">#${item.rank}</span>`
            : `<span class="rank-badge rank-bottom">#${item.rank}</span>`;
        return `<div class="rank-item">
            ${rankBadge}
            <span class="rank-label">${this.escapeHtml(item.label)}</span>
            <span class="rank-primary">${this.escapeHtml(item.primary)}</span>
            ${item.secondary ? `<span class="rank-secondary">${this.escapeHtml(item.secondary)}</span>` : ''}
        </div>`;
    }).join('');
    widgetDiv.innerHTML = `
        <h4 style="font-weight:600; color:#fff; margin-bottom:0.75rem;">
            ${this.escapeHtml(widget.title)}
        </h4>
        <div class="rank-list">${items}</div>
        ${widget.data.total > widget.data.limit
            ? `<p class="rank-note">Showing top and bottom ${widget.data.limit} of ${widget.data.total}</p>`
            : ''}`;
}
```

---

## 4. New Architecture After Changes

```
User query
  → views.py: _process_message_sync()
    → LangGraph agent graph:
        cleanup → clarification_gate
          → agent (LLM+tools)
            → tools (execute tool calls)
              → post_process_tools (compute chart/artifact, merge sibling artifacts via merge_group)
                → agent (loop)
                  → finalize_dashboard (one structured LLM call:
                      {summary, observations} → replace last msg + insight widget)
                    → format_response (add dashboard pointer if not redundant)
                      → __end__
    → view reads final state
    → emits SSE: thinking, llm, visualization, dashboard (with multi-widget artifact)
  → Frontend renderArtifact():
      stat_card → progress_bar → chart → table → ranking_highlight → insight
      + Save Dashboard + Follow-up chips
```

---

## 5. Files to Modify

| File | Change | Lines |
|---|---|---|
| `widget_composer.py` | `_decompose_grouped`, `_decompose_ranked`, `_decompose_statistical`, `_decompose_recent` + shared helpers + update `decompose()` routing | ~160 new |
| `agent_workflow.py` | `finalize_dashboard` node with structured LLM call, message-replacement guards, merge_group-based `_merge_artifacts`, update graph edges, INFO timing log | ~100 new, ~15 modified |
| `chart_builder.py` | `ChartDecision` dataclass, `select_chart_type` returns `ChartDecision`, apply in `build_chart_config` | ~20 modified |
| `unified_chat.html` | `insight` widget handler, `ranking_highlight` widget handler | ~60 new |

**Removed from plan:** system prompt update (obviated by finalize_dashboard), regex trimming (obviated by structured output), generate_insights node (merged into finalize_dashboard).

---

## 6. Verification

### Happy path
1. "Decibel across regions and communities" → multi-widget: stat cards + region chart + community chart + ranking + table + insights. LLM text is 2-4 sentences written by finalize_dashboard. "See dashboard below" appended unless LLM already mentions it.
2. "Top 10 loudest recordings" → stat card spotlighting #1 + bar chart + table + insights
3. "Overview" → existing 3-widget behavior preserved (regression check)
4. "ML profile" → existing 4-widget behavior preserved (regression check)

### Edge cases & failures
5. `finalize_dashboard` structured LLM call fails → keep original agent message, skip insight widget, error logged. Dashboard renders without insights. Not degraded.
6. `finalize_dashboard` returns empty summary for trivial data (1 row) → no replacement, agent's short answer preserved. Dashboard footer added by format_response.
7. Agent's last message has `tool_calls` → finalize_dashboard skips replacement. message remains unmodified.
8. 16 communities → chart shows "Top 12 of 16" horizontal bars, sorted by avg_db descending. ChartDecision with TruncateSpec drives this.
9. Two sibling queries (region + community) → both declare `merge_group: "decibel_by_dimension"`. merged artifact: all widgets re-sorted by priority. Layout updates from "single" to "grid".
10. `highest_decibel` + `lowest_decibel` → both declare `merge_group: "decibel_extremes"`. Merged. Not caught by old string-prefix heuristic.
11. Scalar count query → no chart, no dashboard (`skip_visualization: True`). Without a widget-ful artifact, finalize_dashboard does nothing.
12. 4 communities → decomposition handles n <= 6 correctly: bottom3 skipped, no duplicate rows in ranking.
13. format_response with dashboard pointer: when finalize_dashboard summary already says "the dashboard shows...", format_response skips adding redundant footer.

### Visual verification in browser
14. Open unified_chat.html, send "analyze decibel across regions" → confirm:
    - Stat cards render above charts
    - Chart renders with Chart.js
    - Ranking highlight shows top 3 and bottom 3 with green/red rank badges
    - Table renders below
    - Insight section shows 2-3 observations
    - LLM text above is brief, not a data dump
    - Save Dashboard button works
    - Follow-up chips are relevant
