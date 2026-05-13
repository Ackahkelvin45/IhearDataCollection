from typing import Optional, List, Dict, Any
from dataclasses import dataclass

TEMPORAL_KEYWORDS = ["date", "time", "period", "month", "year", "day", "week"]
MEANINGFUL_NUMERIC_KEYWORDS = [
    "count",
    "total",
    "avg",
    "average",
    "sum",
    "rate",
    "level",
    "score",
]


@dataclass
class TruncateSpec:
    limit: int
    sort_by: str
    sort_order: str  # "asc" | "desc"


@dataclass
class ChartDecision:
    chart_type: str
    truncate: Optional[TruncateSpec] = None


_CHART_TYPE_LABELS: Dict[str, str] = {
    "bar_chart": "Bar Chart",
    "horizontal_bar_chart": "Horizontal Bar Chart",
    "line_chart": "Line Chart",
    "scatter_plot": "Scatter Plot",
    "pie_chart": "Pie Chart",
    "donut_chart": "Donut Chart",
    "box_plot": "Box Plot",
    "area_chart": "Area Chart",
    "heatmap": "Heatmap",
    "class_distribution_bar": "Class Distribution",
    "correlation_heatmap": "Correlation Heatmap",
    "feature_importance_bar": "Feature Importance",
}


def auto_detect_axes(rows: list, columns: list) -> dict:
    """
    Fallback axis detection when chart_hint is not provided (SQL path, legacy tools).
    Priority: temporal > categorical > any string column.
    This fixes the recording_device bug: recording_date is temporal, gets priority.
    """
    if not rows or not columns:
        return {"x": None, "y": None, "group_by": None, "type": None}

    temporal_cols = [
        c for c in columns if any(kw in c.lower() for kw in TEMPORAL_KEYWORDS)
    ]
    numeric_cols = [c for c in columns if _is_numeric_column(rows, c)]
    categorical_cols = [
        c for c in columns if c not in numeric_cols and c not in temporal_cols
    ]

    x = None
    y = None

    if temporal_cols and numeric_cols:
        x = temporal_cols[0]
        y = _pick_most_meaningful_numeric(numeric_cols)

    elif categorical_cols and numeric_cols:
        x = max(categorical_cols, key=lambda c: len(set(r.get(c) for r in rows)))
        y = _pick_most_meaningful_numeric(numeric_cols)

    elif len(numeric_cols) >= 2:
        x = numeric_cols[0]
        y = numeric_cols[1]

    return {"x": x, "y": y, "group_by": None, "type": None}


def _is_numeric_column(rows: list, col: str) -> bool:
    sample = [r.get(col) for r in rows[:10] if r.get(col) is not None]
    return bool(sample) and all(isinstance(v, (int, float)) for v in sample)


def _pick_most_meaningful_numeric(numeric_cols: list) -> str:
    for keyword in MEANINGFUL_NUMERIC_KEYWORDS:
        for col in numeric_cols:
            if keyword in col.lower():
                return col
    return numeric_cols[0]


# ── Chart type selection ───────────────────────────────────────────────────────


def select_chart_type(rows: list, columns: list, hint: dict) -> Optional[ChartDecision]:
    """
    Pure Python decision tree. No LLM. No external calls.
    Returns a ChartDecision with chart type and optional truncation spec,
    or None when no chart should be rendered.

    hint.type overrides this function if explicitly set.
    hint.x and hint.y are used to evaluate data characteristics.
    """

    # Hard override from tool or chart_hint
    if hint.get("type"):
        return ChartDecision(chart_type=hint["type"])

    x_col = hint.get("x")
    y_col = hint.get("y")

    if not x_col or not y_col:
        return None

    n_rows = len(rows)
    if n_rows == 0:
        return None

    is_temporal_x = any(kw in (x_col or "").lower() for kw in TEMPORAL_KEYWORDS)
    is_numeric_x = _is_numeric_column(rows, x_col)
    is_numeric_y = _is_numeric_column(rows, y_col)
    x_cardinality = len(set(r.get(x_col) for r in rows if r.get(x_col) is not None))

    # ── Temporal x-axis ────────────────────────────────────────────────────────
    if is_temporal_x and is_numeric_y:
        if n_rows <= 7:
            return ChartDecision(chart_type="bar_chart")
        return ChartDecision(chart_type="line_chart")

    # ── Two numeric axes (scatter / correlation) ───────────────────────────────
    if is_numeric_x and is_numeric_y:
        return ChartDecision(chart_type="scatter_plot")

    # ── Categorical x-axis ─────────────────────────────────────────────────────
    if is_numeric_y:
        if _is_ratio_data(rows, y_col):
            if x_cardinality <= 6:
                return ChartDecision(chart_type="pie_chart")
            return ChartDecision(chart_type="donut_chart")
        if x_cardinality <= 5:
            return ChartDecision(chart_type="bar_chart")
        if x_cardinality <= 12:
            return ChartDecision(chart_type="horizontal_bar_chart")
        # 13+ categories: show top 12, sorted by y-column descending
        return ChartDecision(
            chart_type="horizontal_bar_chart",
            truncate=TruncateSpec(limit=12, sort_by=y_col, sort_order="desc"),
        )

    return None


def _is_ratio_data(rows: list, col: str) -> bool:
    """
    Returns True if column values look like proportions (0.0–1.0)
    or percentages (0–100 summing to ~100) AND the column name
    suggests ratio/percentage semantics.
    """
    values = [r.get(col) for r in rows if isinstance(r.get(col), (int, float))]
    if not values:
        return False
    # Proportions in 0.0–1.0 range — always treat as ratio data
    if all(0.0 <= v <= 1.0 for v in values):
        return True
    # For 0–100 range, also require column name to suggest percentages
    ratio_keywords = ["pct", "percent", "percentage", "proportion", "share", "ratio"]
    if not any(kw in (col or "").lower() for kw in ratio_keywords):
        return False
    total = sum(values)
    return 85 <= total <= 115  # percentage data summing to ~100


# ── Chart config builder ───────────────────────────────────────────────────────


def _humanise(col_name: Optional[str]) -> str:
    """Convert snake_case column name to Title Case label."""
    if not col_name:
        return ""
    return col_name.replace("_", " ").title()


def build_chart_config(
    chart_type: str,
    rows: List[Dict[str, Any]],
    hint: Dict[str, Any],
    decision: Optional[ChartDecision] = None,
) -> Dict[str, Any]:
    """
    Produces the final chart object in the format the frontend expects.
    Extracts labels and data arrays from rows using the resolved axes.

    When a ChartDecision with truncation is provided, rows are sorted and
    trimmed before building labels/data, and the title reflects the truncation.
    """
    x_key = hint.get("x")
    y_key = hint.get("y")
    total_before_truncate = len(rows)

    # Apply truncation from ChartDecision
    if decision and decision.truncate:
        t = decision.truncate
        reverse = t.sort_order == "desc"
        rows = sorted(rows, key=lambda r: r.get(t.sort_by, 0) or 0, reverse=reverse)
        rows = rows[: t.limit]

    max_rows = 12
    display_rows = rows[:max_rows]

    labels = [str(r.get(x_key) or f"Row {i + 1}") for i, r in enumerate(display_rows)]

    def _to_float(v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(str(v or "0").replace(",", ""))
        except (ValueError, TypeError):
            return 0.0

    data = [_to_float(r.get(y_key)) for r in display_rows]

    title = _humanise(y_key) or "Results"
    if decision and decision.truncate:
        title = f"{title} (Top {decision.truncate.limit} of {total_before_truncate})"

    return {
        "visualization_type": chart_type,
        "visualization_name": _CHART_TYPE_LABELS.get(chart_type, "Chart"),
        "frontend_data": {
            "type": chart_type,
            "title": title,
            "labels": labels,
            "data": data,
            "colors": None,
            "description": f"{_CHART_TYPE_LABELS.get(chart_type, 'Chart')} for {_humanise(y_key)} by {_humanise(x_key)}",
        },
    }


def _build_table_config(
    rows: List[Dict[str, Any]], columns: List[str], tool_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a table visualization config when chart axes cannot be determined."""
    max_rows = 20
    display_rows = rows[:max_rows]
    table_rows = [{c: r.get(c) for c in columns} for r in display_rows]

    pagination = None
    if isinstance(tool_result, dict):
        pagination = tool_result.get("pagination") or {
            "limit": tool_result.get("limit"),
            "offset": tool_result.get("offset"),
            "has_more": tool_result.get("has_more"),
            "total_count": tool_result.get("total_count"),
            "query_kind": tool_result.get("analysis_type")
            or tool_result.get("query_kind"),
        }

    title = (
        tool_result.get("title", "Results")
        if isinstance(tool_result, dict)
        else "Results"
    )

    return {
        "visualization_type": "table",
        "visualization_name": "Table",
        "frontend_data": {
            "type": "none",
            "title": title,
            "table": {
                "columns": columns,
                "rows": table_rows,
                "title": title,
                "pagination": pagination,
            },
            "description": "Table results",
        },
    }


# ── Main entry point ───────────────────────────────────────────────────────────


def resolve_chart(tool_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Single entry point called by post_process_tools_node.
    Takes the full tool result dict, returns a chart config or None.

    The tool result is expected to have:
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

    Falls back to a table config when rows exist but chart axes cannot be determined.
    """
    rows = tool_result.get("rows", [])
    columns = tool_result.get("columns", [])
    hint = tool_result.get("chart_hint") or {}

    if not rows or not columns:
        return None

    # Fill missing hint axes using fallback scanner
    if not hint.get("x") or not hint.get("y"):
        detected = auto_detect_axes(rows, columns)
        # hint values take precedence over detected values
        hint = {**detected, **{k: v for k, v in hint.items() if v is not None}}

    if not hint.get("x") or not hint.get("y"):
        return _build_table_config(rows, columns, tool_result)

    decision = select_chart_type(rows, columns, hint)
    if not decision:
        return _build_table_config(rows, columns, tool_result)

    return build_chart_config(decision.chart_type, rows, hint, decision)
