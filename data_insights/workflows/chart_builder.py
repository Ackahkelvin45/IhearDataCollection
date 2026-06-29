from typing import Optional, List, Dict, Any, Tuple
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

# ── Part-of-whole (composition) classification ─────────────────────────────────
# Pie/donut charts are only meaningful when slices are non-negative components
# that sum to a meaningful total. A column name must carry one of these signals
# before its 0..1 values are treated as composition data (P5-2).
COMPOSITION_KEYWORDS = [
    "pct",
    "percent",
    "percentage",
    "proportion",
    "share",
    "ratio",
    "fraction",
    "composition",
    "distribution",
    "breakdown",
]

# Metrics that frequently live in [0, 1] but are NOT parts of a whole, so they
# must never be drawn as pie/donut slices (RMS energy, entropy, correlation,
# mutual information, coefficients, scores, probabilities-as-magnitude). If a
# column name matches one of these it is forced off the composition path even
# when it also matches a COMPOSITION_KEYWORD (P5-2).
NON_COMPOSITIONAL_KEYWORDS = [
    "rms",
    "energy",
    "entropy",
    "correlation",
    "corr",
    "mutual_info",
    "mutual_information",
    "mi_score",
    "spearman",
    "pearson",
    "coefficient",
    "coef",
    "score",
    "centroid",
    "bandwidth",
    "zcr",
    "variance",
    "std",
    "stddev",
]

# Known units for unit-bearing metrics, keyed by column-name substring. Used to
# re-attach units to axis labels and stat-card values/labels (P5-3). Only attach
# a unit when it is known for that metric — unknown columns get no unit.
# Order matters: more specific substrings are matched first.
COLUMN_UNITS: List[Tuple[str, str]] = [
    ("mean_db", "dB"),
    ("avg_db", "dB"),
    ("max_db", "dB"),
    ("min_db", "dB"),
    ("std_db", "dB"),
    ("median_db", "dB"),
    ("decibel", "dB"),
    ("_db", "dB"),
    ("centroid", "Hz"),
    ("bandwidth", "Hz"),
    ("rolloff", "Hz"),
    ("frequency", "Hz"),
    ("duration", "s"),
]


def unit_for_column(col: Optional[str]) -> Optional[str]:
    """Return the known unit for a metric column, or None when unknown.

    Matches on column-name substrings (case-insensitive) so e.g. ``avg_db``,
    ``mean_db`` and ``noise_mean_db`` all resolve to ``"dB"``. Used to re-attach
    units stripped from chart axes and stat cards (P5-3).
    """
    if not col:
        return None
    lowered = col.lower()
    for needle, unit in COLUMN_UNITS:
        if needle in lowered:
            return unit
    return None


def label_with_unit(label: str, unit: Optional[str]) -> str:
    """Append ``(unit)`` to a human label when a unit is known and not already present."""
    if not unit or not label:
        return label
    if f"({unit})" in label:
        return label
    return f"{label} ({unit})"


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
        if _is_composition_data(rows, y_col):
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


def _is_composition_data(rows: list, col: str) -> bool:
    """Return True only when the column is a *part-of-whole* composition.

    Pie/donut charts are only meaningful when slices are non-negative components
    that sum to a meaningful total. Bare 0..1 values are NOT enough: metrics like
    RMS energy, entropy, correlation or mutual information also live in [0, 1] but
    are not compositions, and drawing them as pie slices misleads users (P5-2).

    Requirements:
      * the column name carries a composition signal (pct/share/proportion/…),
      * the column name does NOT match a known non-compositional metric,
      * all values are non-negative, and
      * the values plausibly sum to a whole — either fractions summing to ~1.0,
        or percentages summing to ~100.
    """
    lowered = (col or "").lower()

    # Block known magnitude metrics that happen to fall in [0, 1].
    if any(kw in lowered for kw in NON_COMPOSITIONAL_KEYWORDS):
        return False

    # Require an explicit part-of-whole signal in the column name.
    if not any(kw in lowered for kw in COMPOSITION_KEYWORDS):
        return False

    values = [r.get(col) for r in rows if isinstance(r.get(col), (int, float))]
    if not values:
        return False

    # Components must be non-negative to form a meaningful whole.
    if any(v < 0 for v in values):
        return False

    total = sum(values)
    # Fractions summing to ~1.0, or percentages summing to ~100.
    if all(0.0 <= v <= 1.0 for v in values) and 0.85 <= total <= 1.15:
        return True
    return 85 <= total <= 115


# Backwards-compatible alias — older callers/tests may reference _is_ratio_data.
_is_ratio_data = _is_composition_data


# ── Chart config builder ───────────────────────────────────────────────────────


def _humanise(col_name: Optional[str]) -> str:
    """Convert snake_case column name to Title Case label."""
    if not col_name:
        return ""
    return col_name.replace("_", " ").title()


def _coerce_numeric(v: Any) -> Optional[float]:
    """Coerce a y-value to float, or return None when it is not a real number.

    P5-6: the previous behaviour returned 0.0 for None / unparseable values,
    which drew misleading zero-height bars (a fabricated "0" that looks like a
    genuine measurement — especially wrong for dB where 0 is meaningful). We now
    return None so the caller can drop the row instead of plotting a fake zero.
    Booleans are rejected (``True``/``False`` are ints in Python but are not
    meaningful chart magnitudes).
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


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

    # P5-6: pair each label with a coerced numeric value, then drop rows whose
    # y-value is missing/non-numeric so we never fabricate a zero-height bar that
    # looks like a genuine zero. Labels and data stay aligned because we filter
    # the paired list, not the two arrays independently.
    labels: List[str] = []
    data: List[float] = []
    dropped_non_numeric = 0
    for i, r in enumerate(display_rows):
        value = _coerce_numeric(r.get(y_key))
        if value is None:
            dropped_non_numeric += 1
            continue
        labels.append(str(r.get(x_key) or f"Row {i + 1}"))
        data.append(value)

    unit = unit_for_column(y_key)
    title = label_with_unit(_humanise(y_key) or "Results", unit)

    # P5-4: surface truncation so users know data was capped, not exhaustive.
    # ChartDecision truncation (sorted top-N) and the hard max_rows cap are both
    # accounted for; the smaller of the two effective limits is what the user
    # actually sees.
    truncate_limit = (
        decision.truncate.limit if (decision and decision.truncate) else None
    )
    shown = len(data)
    is_truncated = total_before_truncate > shown or dropped_non_numeric > 0
    if truncate_limit is not None and total_before_truncate > truncate_limit:
        title = (
            f"{title} (Top {min(truncate_limit, max_rows)} of {total_before_truncate})"
        )
    elif total_before_truncate > max_rows:
        title = f"{title} (Top {shown} of {total_before_truncate})"

    caption = None
    if total_before_truncate > shown + dropped_non_numeric:
        caption = f"Showing top {shown} of {total_before_truncate}"
    if dropped_non_numeric:
        note = f"{dropped_non_numeric} row(s) omitted (non-numeric value)"
        caption = f"{caption}; {note}" if caption else note

    x_label = label_with_unit(_humanise(x_key), unit_for_column(x_key))
    y_label = label_with_unit(_humanise(y_key), unit)
    description = (
        f"{_CHART_TYPE_LABELS.get(chart_type, 'Chart')} for {y_label} by {x_label}"
    )

    frontend_data: Dict[str, Any] = {
        "type": chart_type,
        "title": title,
        "labels": labels,
        "data": data,
        "colors": None,
        "description": description,
        # P5-3: units travel alongside the data so the frontend can label axes.
        "y_unit": unit,
        "y_label": y_label,
        "x_label": x_label,
        # P5-4: explicit truncation metadata for the frontend to render.
        "total_count": total_before_truncate,
        "shown_count": shown,
        "truncated": is_truncated,
        "dropped_rows": dropped_non_numeric,
    }
    if caption:
        frontend_data["caption"] = caption

    return {
        "visualization_type": chart_type,
        "visualization_name": _CHART_TYPE_LABELS.get(chart_type, "Chart"),
        "frontend_data": frontend_data,
    }


def _build_table_config(
    rows: List[Dict[str, Any]], columns: List[str], tool_result: Dict[str, Any]
) -> Dict[str, Any]:
    """Build a table visualization config when chart axes cannot be determined."""
    max_rows = 20
    total_count = len(rows)
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

    # P5-4: surface truncation so the frontend can show "Showing N of M rows".
    shown = len(table_rows)
    table_block: Dict[str, Any] = {
        "columns": columns,
        "rows": table_rows,
        "title": title,
        "pagination": pagination,
        "total_count": total_count,
        "shown_count": shown,
        "truncated": total_count > shown,
    }
    if total_count > shown:
        table_block["caption"] = f"Showing {shown} of {total_count} rows"

    return {
        "visualization_type": "table",
        "visualization_name": "Table",
        "frontend_data": {
            "type": "none",
            "title": title,
            "table": table_block,
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
