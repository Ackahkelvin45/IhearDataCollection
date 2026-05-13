# data_insights — Visualization Pipeline Fix
## Issues 1 & 2: Wrong Chart Dimensions + Chart Type Selection

**Version:** 1.0
**Scope:** Visualization pipeline only. Clarification system already implemented separately.
**Claude Code:** Treat this as the single source of truth. Do not make architectural decisions not covered here.

---

## Problems Being Solved

| # | Problem | Symptom | Root Cause |
|---|---------|---------|------------|
| 1 | **Wrong chart dimensions** | Chart shows `recording_device × count` instead of `date × count` | `_inject_chart_data` scans column names with keyword matching and grabs the first label-like column it finds — `recording_device` before `recording_date` |
| 2 | **Wrong chart type selected** | Line chart recommended for sparse discrete dates; chart type does not reflect actual data shape | `VisualizationAnalysisTool` makes an LLM call to select a type, but heuristics both precede and override the LLM output — the call is wasted and the heuristics are wrong |

---

## Decision Evaluation

### Decision 1: How to fix wrong chart dimensions

**Option A — Improve column scanning heuristics (REJECTED)**
Prioritise `date`/`time` column names over categorical columns. Add word-boundary checks, type detection, semantic scoring.
**Why rejected:** Still guessing. Any new tool with an unexpected column name breaks it silently. You are patching the same bug class repeatedly. `_inject_chart_data` has no semantic knowledge of what the tool intended — it is structurally incapable of getting this right reliably.

**Option B — chart_hint contract from the tool (CHOSEN)**
Each tool explicitly declares its chart axes in its return value. `_inject_chart_data` reads the hint directly — no scanning, no guessing.
**Why chosen:** The tool knows its own output shape. A `recent_datasets` query knows dates are the x-axis. A `decibel_ranked` query knows the dataset name is the label. This is proper separation of concerns — deterministic, debuggable, one place to update per tool.

**Migration fallback:** Tools without `chart_hint` fall back to an improved column priority scanner (temporal columns first, then categorical). This fixes the `recording_device` bug even during migration because `recording_date` is temporal and gets priority. No tool breaks during the transition.

---

### Decision 2: How to fix chart type selection

**Option A — Keep LLM selection (REJECTED)**
`VisualizationAnalysisTool` already calls an LLM to select a chart type.
**Why rejected:** The heuristics in `_analyze_query_characteristics` and `_suggest_chart_type` determine the type before the LLM is called. `_validate_recommendation` then overrides the LLM output back to what the heuristics said. The LLM call is provably adding zero value while costing latency and money on every visualisation.

**Option B — Pure Python decision tree (CHOSEN)**
Data characteristics are measurable: does it have a temporal column? How many rows? How many unique category values? Is the data ratio-based?
**Why chosen:** Free, instant, deterministic, debuggable. Same data always produces the same chart. When a wrong chart appears you trace exactly which condition fired. Works for structured tool results AND the SQL fallback path (which has no tool schema).

**Option C — Tool-declared type in chart_hint only (REJECTED)**
Works for structured tools but the SQL agent has no tool schema and cannot provide a chart_hint.
**Why rejected:** Incomplete — does not cover the SQL path.

**Correct combination:** `chart_hint` declares the **axes** (what to plot). The decision tree selects the **chart type** (how to plot it). If `chart_hint` explicitly sets a `type`, it overrides the decision tree. This covers all cases including SQL fallback.

---

### Decision 3: Where to compute visualisation

**Current problem:** Visualisation is computed in three places in `views.py` (lines 288–296, 330–365, 466–481). The last computation wins. LLM calls are paid but results from earlier computations are discarded.

**Fix:** Compute visualisation exactly once in `post_process_tools_node`, immediately after tool results arrive. Attach to state as `pending_chart`. The stream emits it once. All three existing computation sites in `views.py` are removed.

---

## Implementation

### Step 1 — New file: `workflows/chart_builder.py`

Create this file. It contains all chart logic. Nothing else imports chart logic from anywhere else after this.

```python
# workflows/chart_builder.py

from typing import Optional
import re


# ── Axis detection ─────────────────────────────────────────────────────────────

TEMPORAL_KEYWORDS = ["date", "time", "period", "month", "year", "day", "week"]
MEANINGFUL_NUMERIC_KEYWORDS = ["count", "total", "avg", "average", "sum", "rate", "level", "score"]


def auto_detect_axes(rows: list, columns: list) -> dict:
    """
    Fallback axis detection when chart_hint is not provided (SQL path, legacy tools).
    Priority: temporal > categorical > any string column.
    This fixes the recording_device bug: recording_date is temporal, gets priority.
    """
    if not rows or not columns:
        return {"x": None, "y": None, "group_by": None, "type": None}

    temporal_cols = [c for c in columns if any(kw in c.lower() for kw in TEMPORAL_KEYWORDS)]
    numeric_cols = [c for c in columns if _is_numeric_column(rows, c)]
    categorical_cols = [
        c for c in columns
        if c not in numeric_cols and c not in temporal_cols
    ]

    x = None
    y = None

    if temporal_cols and numeric_cols:
        x = temporal_cols[0]
        y = _pick_most_meaningful_numeric(numeric_cols)

    elif categorical_cols and numeric_cols:
        # Pick the categorical column with the highest cardinality as x
        x = max(categorical_cols, key=lambda c: len(set(r.get(c) for r in rows)))
        y = _pick_most_meaningful_numeric(numeric_cols)

    elif len(numeric_cols) >= 2:
        x = numeric_cols[0]
        y = numeric_cols[1]

    return {"x": x, "y": y, "group_by": None, "type": None}


def _is_numeric_column(rows: list, col: str) -> bool:
    """Check if a column contains numeric values."""
    sample = [r.get(col) for r in rows[:10] if r.get(col) is not None]
    return bool(sample) and all(isinstance(v, (int, float)) for v in sample)


def _pick_most_meaningful_numeric(numeric_cols: list) -> str:
    """
    Prefer columns whose names suggest they are aggregated values
    rather than raw IDs or codes.
    """
    for keyword in MEANINGFUL_NUMERIC_KEYWORDS:
        for col in numeric_cols:
            if keyword in col.lower():
                return col
    return numeric_cols[0]


# ── Chart type selection ───────────────────────────────────────────────────────

def select_chart_type(rows: list, columns: list, hint: dict) -> str:
    """
    Pure Python decision tree. No LLM. No external calls.
    Returns one of: "bar", "horizontal_bar", "line", "scatter", "pie", "donut", "table"

    hint.type overrides this function if explicitly set.
    hint.x and hint.y are used to evaluate data characteristics.
    """

    # Hard override from tool or chart_hint
    if hint.get("type"):
        return hint["type"]

    x_col = hint.get("x")
    y_col = hint.get("y")

    if not x_col or not y_col:
        return "table"

    n_rows = len(rows)
    if n_rows == 0:
        return "table"

    is_temporal_x = any(kw in x_col.lower() for kw in TEMPORAL_KEYWORDS)
    is_numeric_x = _is_numeric_column(rows, x_col)
    is_numeric_y = _is_numeric_column(rows, y_col)
    x_cardinality = len(set(r.get(x_col) for r in rows if r.get(x_col) is not None))

    # ── Temporal x-axis ────────────────────────────────────────────────────────
    if is_temporal_x and is_numeric_y:
        if n_rows <= 7:
            return "bar"        # sparse dates: bars are clearer than a connected line
        return "line"           # 8+ time points: line shows trend

    # ── Two numeric axes (scatter / correlation) ───────────────────────────────
    if is_numeric_x and is_numeric_y:
        return "scatter"

    # ── Categorical x-axis ─────────────────────────────────────────────────────
    if is_numeric_y:
        if x_cardinality <= 5:
            return "bar"            # few categories: vertical bar
        if x_cardinality <= 12:
            return "horizontal_bar" # medium: horizontal fits long labels
        return "table"              # 13+ categories: chart is unreadable

    # ── Ratio / percentage data ────────────────────────────────────────────────
    if _is_ratio_data(rows, y_col):
        if x_cardinality <= 6:
            return "pie"
        return "donut"

    # ── Final fallback — always render something ───────────────────────────────
    return "table"


def _is_ratio_data(rows: list, col: str) -> bool:
    """
    Returns True if column values look like proportions (0.0–1.0)
    or percentages (0–100 summing to ~100).
    """
    values = [r.get(col) for r in rows if isinstance(r.get(col), (int, float))]
    if not values:
        return False
    if all(0.0 <= v <= 1.0 for v in values):
        return True
    total = sum(values)
    return 85 <= total <= 115  # percentage data summing to ~100


# ── Chart config builder ───────────────────────────────────────────────────────

def build_chart_config(chart_type: str, rows: list, hint: dict) -> dict:
    """
    Produces the final chart object that views.py streams to the frontend.
    Shape is consistent regardless of chart type.
    """
    return {
        "type": chart_type,
        "x_key": hint.get("x"),
        "y_key": hint.get("y"),
        "group_key": hint.get("group_by"),
        "x_label": hint.get("x_label") or _humanise(hint.get("x")),
        "y_label": hint.get("y_label") or _humanise(hint.get("y")),
        "data": rows,
        "title": None,   # frontend generates title from context
    }


def _humanise(col_name: Optional[str]) -> Optional[str]:
    """Convert snake_case column name to Title Case label."""
    if not col_name:
        return None
    return col_name.replace("_", " ").title()


# ── Main entry point ───────────────────────────────────────────────────────────

def resolve_chart(data_block: dict) -> Optional[dict]:
    """
    Single entry point called by post_process_tools_node.
    Takes the tool's data block, returns a chart config or None.

    data_block shape:
    {
        "rows": [...],
        "columns": [...],
        "chart_hint": {             ← optional; absent for SQL path and legacy tools
            "x": "recording_date",
            "y": "count",
            "group_by": "category", ← optional
            "type": None,           ← optional override
            "x_label": "...",       ← optional
            "y_label": "...",       ← optional
        }
    }
    """
    rows = data_block.get("rows", [])
    columns = data_block.get("columns", [])
    hint = data_block.get("chart_hint") or {}

    if not rows or not columns:
        return None

    # Fill missing hint axes using fallback scanner
    if not hint.get("x") or not hint.get("y"):
        detected = auto_detect_axes(rows, columns)
        # hint values take precedence over detected values
        hint = {**detected, **{k: v for k, v in hint.items() if v is not None}}

    if not hint.get("x") or not hint.get("y"):
        return None  # cannot determine axes — do not render a chart

    chart_type = select_chart_type(rows, columns, hint)
    return build_chart_config(chart_type, rows, hint)
```

---

### Step 2 — Add `chart_hint` to every tool in `tools.py`

Every tool that returns tabular data adds a `chart_hint` block. The tool knows its own output shape — this is authoritative.

**Standard return shape:**

```python
return {
    "message": formatted_text_for_display,
    "data": {
        "rows": raw_rows,
        "columns": column_names,
        "query_type": "recent_datasets",
        "row_count": len(raw_rows),
        "chart_hint": {
            "x": "recording_date",       # column name for x-axis
            "y": "count",                # column name for y-axis
            "group_by": "category",      # colour/group dimension (optional, None if not needed)
            "type": None,                # force a specific chart type (None = let decision tree choose)
            "x_label": "Recording Date", # human-readable label (optional)
            "y_label": "Recordings",     # human-readable label (optional)
        }
    },
    "skip_visualization": False,  # set True for pure text/list results
}
```

**`chart_hint` for each `query_type`:**

| query_type | x | y | group_by | type override | notes |
|------------|---|---|----------|---------------|-------|
| `recent_datasets` | `recording_date` | `count` | `category` | `None` | Sparse dates → decision tree picks bar |
| `top_collectors` | `name` | `recording_count` | `region` | `None` | Ranking → bar or horizontal_bar |
| `decibel_ranked` | `dataset_name` | `avg_decibel` | `category` | `None` | Ranking by dB |
| `count_by_region` | `region` | `count` | `None` | `None` | Regional distribution |
| `count_by_category` | `category` | `count` | `None` | `None` | Category distribution |
| `count_by_community` | `community` | `count` | `region` | `None` | Community breakdown |
| `correlation` | first numeric col | second numeric col | `None` | `"scatter"` | Force scatter — always |
| `statistical` | `metric_name` | `value` | `None` | `"horizontal_bar"` | Stats summary — force horizontal |
| `temporal` | `period` | `value` | `category` | `None` | Time series — decision tree picks line/bar |
| `sql` | — | — | — | — | No chart_hint — fallback scanner used |

**For `skip_visualization`:**
Set `True` on: count-only results (single number), pure list results with no numeric column, error messages.

---

### Step 3 — Update `post_process_tools_node` in `agent_workflow.py`

Visualisation is computed **once here**. Remove all visualisation computation from `views.py`.

```python
# agent_workflow.py
from workflows.chart_builder import resolve_chart

def post_process_tools_node(state: AgentState) -> AgentState:
    last_result = _get_last_tool_result(state)

    if not last_result:
        return state

    data_block = last_result.get("data")
    skip_viz = last_result.get("skip_visualization", False)

    if not data_block or skip_viz:
        return {**state, "pending_chart": None}

    chart_config = resolve_chart(data_block)   # None if chart cannot be determined
    return {**state, "pending_chart": chart_config}
```

---

### Step 4 — Update `views.py`

**Remove** all three existing visualisation computation sites:
- Lines 288–296 (table viz on ToolMessage receipt)
- Lines 330–365 (auto viz on ToolMessage receipt)
- Lines 466–481 (auto viz attempt after stream loop ends)

**Add** a single chart chunk emitter that reads `state["pending_chart"]`:

```python
# views.py — inside the stream generator, after the agent finishes

pending_chart = final_state.get("pending_chart")
if pending_chart:
    yield json.dumps({
        "type": "chart",
        "config": pending_chart,
    }) + "\n"
```

**Add** handling for the `chart` chunk type on the frontend side — this is the only place charts are emitted.

---

### Step 5 — Remove the wasted LLM call in `VisualizationAnalysisTool`

In `tools.py`, `VisualizationAnalysisTool` currently:
1. Runs `_analyze_query_characteristics` (heuristic)
2. Runs `_suggest_chart_type` (heuristic)
3. Calls an internal `ChatOpenAI` instance (LLM)
4. Runs `_validate_recommendation` which overrides the LLM back to the heuristic result

Steps 3 and 4 are removed entirely. `VisualizationAnalysisTool` becomes:

```python
class VisualizationAnalysisTool:
    def _run(self, data_block: dict) -> dict:
        """
        Thin wrapper — delegates entirely to chart_builder.
        The internal ChatOpenAI instance and all LLM calls are removed.
        """
        from workflows.chart_builder import resolve_chart
        chart_config = resolve_chart(data_block)
        return {"chart": chart_config} if chart_config else {"chart": None}
```

Remove: the internal `ChatOpenAI` instance, `_analyze_query_characteristics`, `_suggest_chart_type`, `_validate_recommendation`. These are replaced by `chart_builder.py`.

---

## Chart Type Decision Reference

| Data shape | Chart type | Threshold |
|------------|------------|-----------|
| Temporal x + numeric y, ≤7 rows | `bar` | Sparse dates: line implies false continuity |
| Temporal x + numeric y, >7 rows | `line` | Enough points to show trend |
| Two numeric columns, no temporal | `scatter` | Correlation / distribution |
| Categorical x, ≤5 unique values | `bar` | Few categories: vertical bar |
| Categorical x, 6–12 unique values | `horizontal_bar` | Long labels need horizontal space |
| Categorical x, >12 unique values | `table` | 13+ bars are cognitively unreadable |
| Ratio data, ≤6 categories | `pie` | Clear proportional breakdown |
| Ratio data, 7–12 categories | `donut` | Slightly more slices than pie handles |
| Anything else | `table` | Never fail silently — always render something |

---

## Files to Create / Modify

| Action | File | Change |
|--------|------|--------|
| **Create** | `workflows/chart_builder.py` | `auto_detect_axes`, `select_chart_type`, `build_chart_config`, `resolve_chart` |
| **Modify** | `workflows/tools.py` | Add `chart_hint` to all tool return values; gut `VisualizationAnalysisTool` LLM call |
| **Modify** | `workflows/agent_workflow.py` | Add `pending_chart` to `AgentState`; update `post_process_tools_node` |
| **Modify** | `views.py` | Remove 3x visualisation computation; add single chart chunk emit from `pending_chart` |
| **Modify** | `frontend/` | Handle `{"type": "chart", "config": {...}}` chunk type and render chart |

---

## Testing Checklist

### Wrong dimensions fix
- [ ] "show me recent data" → chart uses `recording_date` on x-axis, not `recording_device`
- [ ] Fallback scanner: `recording_date` (temporal) detected before `recording_device` (categorical)
- [ ] Tool with explicit `chart_hint.x = "recording_date"` → scanner not called, hint used directly
- [ ] SQL path (no chart_hint) → fallback scanner applied, chart still renders

### Chart type selection
- [ ] Temporal data, 3 dates → `bar`
- [ ] Temporal data, 15 dates → `line`
- [ ] Two numeric columns, no dates → `scatter`
- [ ] 4 categories → `bar`
- [ ] 8 categories → `horizontal_bar`
- [ ] 15 categories → `table`
- [ ] Percentage data, 5 slices → `pie`
- [ ] `chart_hint.type = "scatter"` (correlation query) → always scatter regardless of row count
- [ ] `skip_visualization = True` → no chart emitted

### Computation deduplication
- [ ] Visualisation computed exactly once per message
- [ ] No visualisation LLM call made (check token usage)
- [ ] `VisualizationAnalysisTool` contains no `ChatOpenAI` instance

### Known gap (out of scope for this spec)
- SQL agent returns `{"message": msg}` with no `data` block. Charts will not render for SQL queries until the SQL agent is updated to return structured rows. Track separately — do not address here.
