"""
Widget Composer — converts tool results into multi-widget artifact specs.

Phase 1: wrap_as_artifact() wraps a single chart config into artifact format.
Phase 2: decompose() splits rich tool outputs into multiple widgets deterministically.
"""

from typing import Dict, Any, Optional, List

from .chart_builder import (
    build_chart_config,
    select_chart_type,
    auto_detect_axes,
    resolve_chart,
    ChartDecision,
    unit_for_column,
    label_with_unit,
)


# ── Follow-up suggestion mapping ────────────────────────────────────────────────
# Deterministic, no LLM. Each analysis_type maps to 3 follow-up questions.

FOLLOW_UP_MAP: Dict[str, List[str]] = {
    "recent_datasets": [
        "Filter by region",
        "Show me more recordings",
        "Which collectors uploaded these?",
    ],
    "top_collectors_monthly": [
        "Show me all-time top collectors",
        "Which regions do they collect from?",
        "Show me their recent uploads",
    ],
    "category_class_count": [
        "Break down by region instead",
        "Show me which has the highest decibel levels",
        "Show me the raw data as a table",
    ],
    "group_count": [
        "Show me the top 10",
        "Break this down further by category",
        "Show me which has the most recordings",
    ],
    "dataset_count": [
        "Show me the datasets",
        "What is the breakdown by category?",
        "What is the breakdown by region?",
    ],
    "highest_decibel": [
        "Show me the quietest recordings instead",
        "Break this down by region",
        "Show me the trend over time",
    ],
    "lowest_decibel": [
        "Show me the loudest recordings instead",
        "Break this down by region",
        "Show me the trend over time",
    ],
    "avg_decibel_by_region": [
        "Show me the highest individual recordings",
        "Compare by category instead",
        "Show me which region has the most data",
    ],
    "avg_decibel_by_category": [
        "Show me the highest individual recordings",
        "Compare by region instead",
        "Show me which category has the most data",
    ],
    "avg_decibel_by_community": [
        "Show me the highest individual recordings",
        "Compare by region instead",
        "Show me the trend over time",
    ],
    "avg_decibel_by_class": [
        "Show me the highest individual recordings",
        "Compare by category instead",
        "Show me the overview",
    ],
    "avg_decibel_by_recording_device": [
        "Show me the highest individual recordings",
        "Which device has the most recordings?",
        "Show me the overview",
    ],
    "overview_analysis": [
        "Show me trends over time",
        "Which region has the most data?",
        "Profile this dataset for ML readiness",
    ],
    "temporal_analysis": [
        "Show me the overview",
        "What drove the peaks?",
        "Filter to the last 30 days",
    ],
    "energy_analysis": [
        "Show me spectral analysis instead",
        "Break down by region",
        "Show me the loudest recordings",
    ],
    "spectral_analysis": [
        "Show me energy analysis instead",
        "Break down by category",
        "Show me the overview",
    ],
    "frequency_analysis": [
        "Show me the loudest recordings",
        "Break down by region",
        "Show me spectral analysis instead",
    ],
    "correlation_analysis": [
        "Show me the raw data as a table",
        "Show me overview statistics",
        "Show me energy analysis",
    ],
    "statistical_analysis": [
        "Show me the trends over time",
        "Compare with another dimension",
        "Show me the overview",
    ],
    "statistical_distribution": [
        "Show me the highest values",
        "Break down by another dimension",
        "Show me the overview",
    ],
    "ml_dataset_profile": [
        "Show me feature statistics",
        "Show me label distribution by region",
        "What is the data quality like?",
    ],
    "ml_feature_stats": [
        "Show me dataset profile",
        "Show me label distribution",
        "What features have the most variance?",
    ],
    "ml_class_balance": [
        "Show me a 70/15/15 stratified split",
        "Profile the dataset for ML",
        "Which features best separate these classes?",
    ],
    "ml_train_test_split": [
        "Show me class balance first",
        "Show me feature statistics",
        "Export features for training",
    ],
    "ml_correlation_matrix": [
        "Which features best separate the classes?",
        "Show me feature statistics",
        "Export features for training",
    ],
    "ml_statistical_test": [
        "Compare with another group",
        "Show me the correlation matrix",
        "What features best separate these groups?",
    ],
    "ml_export_features": [
        "Show me class balance",
        "Give me a train/test split",
        "Which features are most important?",
    ],
    "ml_feature_importance": [
        "Show me the correlation matrix",
        "Show me class balance",
        "Export features for training",
    ],
}


def _get_follow_ups(analysis_type: Optional[str]) -> List[str]:
    """Return follow-up questions for a given analysis_type, or generic defaults."""
    if analysis_type and analysis_type in FOLLOW_UP_MAP:
        return FOLLOW_UP_MAP[analysis_type]
    return [
        "Show me the overview",
        "Break this down by region",
        "Show me the raw data as a table",
    ]


def wrap_as_artifact(
    chart_config: Optional[Dict[str, Any]],
    analysis_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Wrap a single chart config into the standard artifact format."""
    if not chart_config:
        return None

    frontend_data = chart_config.get("frontend_data", {})
    return {
        "widgets": [
            {
                "id": "widget_1",
                "type": chart_config.get("visualization_type", "bar_chart"),
                "title": chart_config.get("visualization_name", "Chart"),
                "data": frontend_data,
                "config": {},
                "priority": 0,
            }
        ],
        "layout_template": "single",
        "version": 1,
        "follow_ups": _get_follow_ups(analysis_type),
    }


# ── Phase 2: Multi-widget decomposition ──────────────────────────────────────────


def _humanise(col_name: Optional[str]) -> str:
    """Convert snake_case column name to Title Case label."""
    if not col_name:
        return ""
    return col_name.replace("_", " ").title()


def _quartile(sorted_values: List[float], q: float) -> float:
    """Linear-interpolation quantile, matching the frontend calculateQuartile."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_values[0])
    pos = (n - 1) * q
    base = int(pos)
    rest = pos - base
    if base + 1 < n:
        return float(
            sorted_values[base] + rest * (sorted_values[base + 1] - sorted_values[base])
        )
    return float(sorted_values[base])


def _box_stats(values: List[float]) -> Optional[Dict[str, float]]:
    """Compute box-plot statistics (min/q1/median/q3/max) for a group of values.

    Returns None when there are no values so the caller can omit the group rather
    than emit a degenerate box (P5-5: never fabricate a box from nothing).
    """
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return None
    s = sorted(numeric)
    return {
        "min": s[0],
        "q1": _quartile(s, 0.25),
        "median": _quartile(s, 0.5),
        "q3": _quartile(s, 0.75),
        "max": s[-1],
        "count": len(s),
    }


# Shared helpers — small building blocks shared by all decompose functions


def _build_stat_widget(
    widget_id: str, title: str, stats: Dict[str, Any], priority: int = 0
) -> Dict[str, Any]:
    return {
        "id": widget_id,
        "type": "stat_card",
        "title": title,
        "data": {"stats": stats},
        "config": {},
        "priority": priority,
    }


def _build_table_widget(
    widget_id: str,
    title: str,
    rows: List[Dict[str, Any]],
    priority: int,
    total_count: Optional[int] = None,
) -> Dict[str, Any]:
    # P5-4: when the caller passes a total_count larger than the rows shown, the
    # table was truncated — surface a visible "Showing N of M" caption and a
    # machine-readable truncated/total_count flag so users know data was cut.
    shown = len(rows)
    total = total_count if total_count is not None else shown
    table: Dict[str, Any] = {
        "columns": list(rows[0].keys()) if rows else [],
        "rows": rows,
        "title": title,
        "total_count": total,
        "shown_count": shown,
        "truncated": total > shown,
    }
    if total > shown:
        table["caption"] = f"Showing {shown} of {total} rows"
    return {
        "id": widget_id,
        "type": "table",
        "title": title,
        "priority": priority,
        "data": {"table": table},
        "config": {},
    }


def decompose(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Decompose a tool result into multiple widgets based on analysis_type."""
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
    elif analysis_type == "ml_feature_stats":
        return _decompose_ml_feature_stats(result)
    elif analysis_type == "ml_class_balance":
        return _decompose_class_balance(result)
    elif analysis_type == "ml_train_test_split":
        return _decompose_train_test_split(result)
    elif analysis_type == "ml_correlation_matrix":
        return _decompose_correlation(result)
    elif analysis_type == "ml_statistical_test":
        return _decompose_statistical_test(result)
    elif analysis_type == "ml_export_features":
        return _decompose_export_features(result)
    elif analysis_type == "ml_feature_importance":
        return _decompose_feature_importance(result)

    # Flat tool — fall back to single chart
    chart = resolve_chart(result)
    return wrap_as_artifact(chart, analysis_type=analysis_type or None)


def _decompose_overview(result: Dict[str, Any]) -> Dict[str, Any]:
    widgets: List[Dict[str, Any]] = []
    priority = 0

    overview = result.get("overview_statistics", {})
    if overview:
        widgets.append(
            {
                "id": "overview_stats",
                "type": "stat_card",
                "title": "Overview Statistics",
                "data": {"stats": overview},
                "config": {},
                "priority": priority,
            }
        )
        priority += 1

    regional = result.get("regional_breakdown", [])
    if regional:
        w = _rows_to_widget(
            regional,
            "regional_breakdown",
            "Regional Breakdown",
            "bar_chart",
            priority,
            x_key="region__name",
            y_key="count",
        )
        if w:
            widgets.append(w)
            priority += 1

    category = result.get("category_breakdown", [])
    if category:
        w = _rows_to_widget(
            category,
            "category_breakdown",
            "Category Breakdown",
            "pie_chart",
            priority,
            x_key="category__name",
            y_key="count",
        )
        if w:
            widgets.append(w)
            priority += 1

    return {
        "widgets": widgets,
        "layout_template": "grid" if len(widgets) > 2 else "two_column",
        "version": 1,
        "follow_ups": _get_follow_ups("overview_analysis"),
    }


def _decompose_temporal(result: Dict[str, Any]) -> Dict[str, Any]:
    widgets: List[Dict[str, Any]] = []
    priority = 0

    monthly = result.get("monthly_trends", [])
    if monthly:
        w = _rows_to_widget(
            monthly,
            "monthly_trends",
            "Monthly Trends",
            "line_chart",
            priority,
            x_key="month",
            y_key="dataset_count",
        )
        if w:
            widgets.append(w)
            priority += 1

    daily = result.get("daily_trends", [])
    if daily:
        w = _rows_to_widget(
            daily,
            "daily_trends",
            "Daily Trends (Last 30 Days)",
            "line_chart",
            priority,
            x_key="date",
            y_key="dataset_count",
        )
        if w:
            widgets.append(w)
            priority += 1

    return {
        "widgets": widgets,
        "layout_template": "two_column" if len(widgets) >= 2 else "single",
        "version": 1,
        "follow_ups": _get_follow_ups("temporal_analysis"),
    }


def _decompose_ml_profile(result: Dict[str, Any]) -> Dict[str, Any]:
    widgets: List[Dict[str, Any]] = []
    priority = 0

    totals = result.get("totals", {})
    if totals:
        widgets.append(
            {
                "id": "ml_totals",
                "type": "stat_card",
                "title": "Dataset Totals",
                "data": {"stats": totals},
                "config": {},
                "priority": priority,
            }
        )
        priority += 1

    coverage = result.get("coverage", {})
    if coverage:
        items = [
            {
                "label": "With Audio Features",
                "value": coverage.get("features_pct", 0),
                "color": "#22c55e",
            },
            {
                "label": "With Noise Analysis",
                "value": coverage.get("analysis_pct", 0),
                "color": "#3b82f6",
            },
        ]
        widgets.append(
            {
                "id": "ml_coverage",
                "type": "progress_bar",
                "title": "Feature Coverage",
                "data": {"items": items},
                "config": {},
                "priority": priority,
            }
        )
        priority += 1

    missing = result.get("missingness", {})
    total = totals.get("noise_datasets", 1) if totals else 1
    if missing and total:
        items: List[Dict[str, Any]] = []
        for field, missing_count in sorted(missing.items()):
            present_pct = ((total - missing_count) / total * 100) if total > 0 else 0
            if present_pct < 50:
                color = "#ef4444"
            elif present_pct > 80:
                color = "#22c55e"
            else:
                color = "#f59e0b"
            items.append(
                {
                    "label": field.replace("_", " ").title(),
                    "value": round(present_pct, 1),
                    "color": color,
                }
            )
        widgets.append(
            {
                "id": "ml_missingness",
                "type": "progress_bar",
                "title": "Field Completeness",
                "data": {"items": items},
                "config": {},
                "priority": priority,
            }
        )
        priority += 1

    label_dist = result.get("label_distribution", {})
    counts = label_dist.get("counts", [])
    group_by = label_dist.get("group_by", "category")
    if counts:
        label_key = f"{group_by}__name"
        w = _rows_to_widget(
            counts,
            "label_distribution",
            f"Label Distribution by {group_by.title()}",
            "horizontal_bar_chart",
            priority,
            x_key=label_key,
            y_key="count",
        )
        if w:
            widgets.append(w)
            priority += 1

    return {
        "widgets": widgets,
        "layout_template": "grid",
        "version": 1,
        "follow_ups": _get_follow_ups("ml_dataset_profile"),
    }


def _rows_to_widget(
    rows: List[Dict[str, Any]],
    widget_id: str,
    title: str,
    fallback_chart_type: str,
    priority: int,
    x_key: Optional[str] = None,
    y_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Convert a list of row dicts into a chart widget using chart_builder logic."""
    if not rows:
        return None

    columns = list(rows[0].keys())
    hint: Dict[str, Any] = {"x": x_key, "y": y_key}

    if not x_key or not y_key:
        detected = auto_detect_axes(rows, columns)
        hint = {**detected, **{k: v for k, v in hint.items() if v is not None}}

    resolved_x = hint.get("x")
    resolved_y = hint.get("y")
    if not resolved_x or not resolved_y:
        return None

    decision = select_chart_type(rows, columns, hint)
    chart_type = (decision.chart_type if decision else None) or fallback_chart_type
    config = build_chart_config(chart_type, rows, hint, decision)

    return {
        "id": widget_id,
        "type": chart_type,
        "title": title,
        "data": config.get("frontend_data", {}),
        "config": {},
        "priority": priority,
    }


# ── New decompose functions ─────────────────────────────────────────────────


def _decompose_grouped(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose grouped results (avg_decibel_by_*, group_count)."""
    rows = result["rows"]
    hint = result.get("chart_hint", {})
    y_key = hint.get("y", "")
    n = len(rows)
    analysis_type = result.get("analysis_type", "")
    group_label = _humanise(hint.get("x", ""))

    # Stat card: aggregate values
    values = [
        r.get(y_key, 0) if isinstance(r.get(y_key), (int, float)) else 0 for r in rows
    ]
    if values:
        _avg = round(sum(values) / len(values), 2)
        _max = round(max(values), 2)
        _min = round(min(values), 2)
    else:
        _avg = _max = _min = 0
    # P5-3: re-attach the metric's known unit (e.g. dB) to the stat labels so a
    # bare number like 62 is unambiguous.
    y_unit = unit_for_column(y_key)
    y_metric = label_with_unit(_humanise(y_key), y_unit)
    stats = {
        f"Total {group_label}s": n,
        f"Average {y_metric}": _avg,
        f"Highest {y_metric}": _max,
        f"Lowest {y_metric}": _min,
    }
    widgets = [
        _build_stat_widget(f"stats_{analysis_type}", "Summary Statistics", stats, 0)
    ]

    # Chart
    chart_type = "horizontal_bar_chart" if n > 5 else "bar_chart"
    chart_widget = _rows_to_widget(
        rows,
        f"chart_{analysis_type}",
        f"{_humanise(y_key)} by {group_label}",
        chart_type,
        1,
        x_key=hint.get("x"),
        y_key=hint.get("y"),
    )
    if chart_widget:
        widgets.append(chart_widget)

    # Ranking: top 3 + bottom 3 (skip bottom for n <= 6 to avoid overlap)
    sorted_rows = sorted(
        rows,
        key=lambda r: r.get(y_key, 0) if isinstance(r.get(y_key), (int, float)) else 0,
        reverse=True,
    )
    top3 = sorted_rows[:3]
    bottom3 = sorted_rows[-3:] if n > 6 else []
    rank_items = []
    for i, r in enumerate(top3):
        rank_items.append(
            {
                "rank": i + 1,
                "label": str(r.get(hint.get("x", ""), "")),
                "primary": str(r.get(y_key, "")),
                "secondary": "",
                "trend": None,
            }
        )
    offset = max(0, n - 2)
    for i, r in enumerate(bottom3):
        rank_items.append(
            {
                "rank": offset + i,
                "label": str(r.get(hint.get("x", ""), "")),
                "primary": str(r.get(y_key, "")),
                "secondary": "",
                "trend": None,
            }
        )
    if rank_items:
        widgets.append(
            {
                "id": f"ranking_{analysis_type}",
                "type": "ranking_highlight",
                "title": "Top & Bottom Performers",
                "data": {"items": rank_items, "limit": 3, "total": n},
                "priority": 2,
            }
        )

    widgets.append(
        _build_table_widget(f"table_{analysis_type}", "Full Results", rows, 3)
    )
    return {
        "widgets": widgets,
        "layout_template": "grid",
        "version": 1,
        "follow_ups": _get_follow_ups(analysis_type),
    }


def _decompose_ranked(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose ranked results (highest_decibel, lowest_decibel)."""
    rows = result["rows"]
    hint = result.get("chart_hint", {})
    y_key = hint.get("y", "")
    x_key = hint.get("x", "")
    analysis_type = result.get("analysis_type", "")
    n = len(rows)

    # Stat card: spotlight on #1
    top_row = rows[0] if rows else {}
    top_label = str(top_row.get(x_key, ""))
    top_value = top_row.get(y_key, "")
    # P5-3: append the metric's known unit (e.g. dB) to the spotlight value.
    y_unit = unit_for_column(y_key)
    top_value_str = (
        f"{top_value} {y_unit}".strip()
        if y_unit and top_value != ""
        else f"{top_value}"
    )
    stats = {
        "#1": f"{top_label}: {top_value_str}",
        "Total items": n,
    }
    widgets = [_build_stat_widget(f"stats_{analysis_type}", "Top Result", stats, 0)]

    # Chart: horizontal bar for top 10
    display_rows = rows[:10]
    chart_label = "Lowest" if "lowest" in analysis_type else "Highest"
    chart_widget = _rows_to_widget(
        display_rows,
        f"chart_{analysis_type}",
        f"Top 10 {chart_label} Decibels",
        "horizontal_bar_chart",
        1,
        x_key=x_key,
        y_key=y_key,
    )
    if chart_widget:
        widgets.append(chart_widget)

    # Table: full list
    widgets.append(
        _build_table_widget(f"table_{analysis_type}", "Full Results", rows, 2)
    )
    return {
        "widgets": widgets,
        "layout_template": "two_column" if len(widgets) >= 2 else "single",
        "version": 1,
        "follow_ups": _get_follow_ups(analysis_type),
    }


def _decompose_statistical(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose statistical distribution results."""
    rows = result.get("rows", [])
    distribution_data = result.get("distribution_data", {})
    analysis_type = result.get("analysis_type", "")
    group_by = result.get("grouped_by", "group")

    # Stat card: per-group min/avg/max from distribution_data
    stats = {}
    if distribution_data and isinstance(distribution_data, dict):
        for name, data in distribution_data.items():
            if isinstance(data, dict):
                avg = data.get("avg")
                max_v = data.get("max")
                min_v = data.get("min")
                count = data.get("count", 0)
                stats[f"{name} avg"] = round(avg, 2) if avg else 0
                stats[f"{name} max"] = round(max_v, 2) if max_v else 0
                stats[f"{name} min"] = round(min_v, 2) if min_v else 0
                stats[f"{name} count"] = count

    widgets = []
    if stats:
        widgets.append(
            _build_stat_widget(
                f"stats_{analysis_type}",
                "Distribution by " + group_by.title(),
                stats,
                0,
            )
        )

    # Box plot chart if distribution data available.
    # P5-5: emit a COMPLETE box-plot spec (precomputed min/q1/median/q3/max per
    # group as box_plot_data) so the frontend renders a true box plot instead of
    # silently degrading to a mislabeled bar chart. Groups without raw values are
    # omitted (never fabricated). If NO group has raw values we fall back to a bar
    # chart of group means but label it honestly as a mean comparison — not a box
    # plot — and keep the per-group stats available.
    if distribution_data:
        chart_data = []
        chart_labels = []
        box_plot_data: List[List[float]] = []
        for name, data in distribution_data.items():
            if isinstance(data, dict) and data.get("decibel_values"):
                vals = data["decibel_values"]
                box = _box_stats(vals)
                if box is None:
                    continue
                chart_labels.append(name)
                box_plot_data.append(
                    [box["min"], box["q1"], box["median"], box["q3"], box["max"]]
                )
                chart_data.append(
                    {
                        "label": name,
                        "data": vals,
                        "box_stats": box,
                        "avg": data.get("avg", 0),
                        "max": data.get("max", 0),
                        "min": data.get("min", 0),
                    }
                )

        if chart_data:
            widgets.append(
                {
                    "id": f"chart_{analysis_type}",
                    "type": "box_plot",
                    "title": f"Decibel Distribution by {group_by.title()}",
                    "priority": 1,
                    "data": {
                        "type": "box_plot",
                        "title": f"Decibel Distribution by {group_by.title()}",
                        "labels": chart_labels,
                        "data": [d["data"] for d in chart_data],
                        # Precomputed [min, q1, median, q3, max] per group — what
                        # the frontend boxplot renderer consumes directly.
                        "box_plot_data": box_plot_data,
                        "box_stats": [d["box_stats"] for d in chart_data],
                        "y_unit": "dB",
                        "description": f"Distribution of decibel levels across {len(chart_labels)} groups",
                    },
                    "config": {},
                }
            )
        else:
            # No raw distributions available — show an honest mean comparison bar
            # chart rather than a box plot we cannot actually draw.
            mean_labels = []
            mean_values = []
            for name, data in distribution_data.items():
                if isinstance(data, dict) and data.get("avg") is not None:
                    mean_labels.append(name)
                    mean_values.append(data.get("avg"))
            if mean_labels:
                widgets.append(
                    {
                        "id": f"chart_{analysis_type}",
                        "type": "bar_chart",
                        "title": f"Mean Decibel by {group_by.title()}",
                        "priority": 1,
                        "data": {
                            "type": "bar_chart",
                            "title": f"Mean Decibel by {group_by.title()}",
                            "labels": mean_labels,
                            "data": mean_values,
                            "y_unit": "dB",
                            "description": (
                                "Mean decibel level per group (raw distributions "
                                "unavailable, so a box plot cannot be drawn)"
                            ),
                        },
                        "config": {},
                    }
                )

    # Table from rows if available
    if rows:
        widgets.append(
            _build_table_widget(f"table_{analysis_type}", "Distribution Data", rows, 2)
        )

    return {
        "widgets": widgets,
        "layout_template": "grid" if len(widgets) > 2 else "two_column",
        "version": 1,
        "follow_ups": _get_follow_ups(analysis_type),
    }


def _decompose_class_balance(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose ml_class_balance results into stat card + distribution bar + table."""
    widgets: List[Dict[str, Any]] = []
    priority = 0

    imbalance_severity = result.get("imbalance_severity", "balanced")
    entropy = result.get("normalized_entropy", 0)
    minority_pct = result.get("minority_pct", 0)
    n_classes = result.get("n_classes", 0)
    total_rows = result.get("total_rows", 0)

    severity_color = {
        "severe": "#ef4444",
        "moderate": "#f59e0b",
        "mild": "#eab308",
        "balanced": "#22c55e",
    }.get(imbalance_severity, "#6b7280")

    stats = {
        "Total rows": total_rows,
        "Classes": n_classes,
        "Minority class %": f"{minority_pct}%",
        "Imbalance": imbalance_severity.title(),
        "Normalized entropy": round(entropy, 3),
    }
    widgets.append(
        {
            "id": "class_balance_stats",
            "type": "stat_card",
            "title": f"Class Balance — {imbalance_severity.title()}",
            "data": {"stats": stats, "severity_color": severity_color},
            "config": {},
            "priority": priority,
        }
    )
    priority += 1

    # Class distribution bar chart
    per_class = result.get("per_class", [])
    if per_class:
        widgets.append(
            {
                "id": "class_distribution_chart",
                "type": "class_distribution_bar",
                "title": f"Class Distribution ({n_classes} classes)",
                "priority": priority,
                "data": {
                    "type": "class_distribution_bar",
                    "title": f"Class Distribution ({n_classes} classes)",
                    "labels": [c["label"] for c in per_class],
                    "data": [c["count"] for c in per_class],
                    "percentages": [c["percentage"] for c in per_class],
                    "description": f"Distribution of {total_rows} rows across {n_classes} classes",
                },
                "config": {},
            }
        )
        priority += 1

    # Warnings
    warnings = result.get("warnings", [])
    if warnings:
        widgets.append(
            {
                "id": "class_balance_warnings",
                "type": "insight",
                "title": "Warnings",
                "priority": priority,
                "data": {"observations": warnings},
                "config": {"color": severity_color},
            }
        )
        priority += 1

    # Split recommendation table
    split_rec = result.get("split_recommendation", [])
    if split_rec:
        widgets.append(
            _build_table_widget(
                "class_balance_splits",
                "Recommended Stratified Split (70/15/15)",
                split_rec,
                priority,
            )
        )

    return {
        "widgets": widgets,
        "layout_template": "grid" if len(widgets) > 2 else "two_column",
        "version": 1,
        "follow_ups": _get_follow_ups("ml_class_balance"),
    }


def _decompose_train_test_split(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose ml_train_test_split results into stat card + split bar + table."""
    widgets: List[Dict[str, Any]] = []
    priority = 0

    train = result.get("train_count", 0)
    val = result.get("val_count", 0)
    test = result.get("test_count", 0)
    total = result.get("total_rows", 1)
    strategy = result.get("split_strategy", "stratified")
    seed = result.get("seed", 42)

    stats = {
        "Train": f"{train} ({result.get('train_pct', 0)}%)",
        "Validation": f"{val} ({result.get('val_pct', 0)}%)",
        "Test": f"{test} ({result.get('test_pct', 0)}%)",
        "Strategy": strategy.title(),
        "Seed": seed,
    }
    widgets.append(
        _build_stat_widget("split_stats", "Train / Val / Test Split", stats, priority)
    )
    priority += 1

    # Split composition bar
    widgets.append(
        {
            "id": "split_composition_chart",
            "type": "bar_chart",
            "title": f"Split Composition ({strategy.title()})",
            "priority": priority,
            "data": {
                "type": "bar_chart",
                "title": f"Split Composition ({strategy.title()})",
                "labels": ["Train", "Validation", "Test"],
                "data": [train, val, test],
                "description": f"{strategy.title()} split of {total} total rows (seed={seed})",
            },
            "config": {},
        }
    )
    priority += 1

    # Per-class breakdown if available
    splits_per_class = result.get("splits_per_class")
    if splits_per_class:
        widgets.append(
            _build_table_widget(
                "split_per_class",
                "Per-Class Split Breakdown",
                splits_per_class,
                priority,
            )
        )
        priority += 1

    # Warnings (data leakage, unstratifiable classes)
    warnings = result.get("warnings", [])
    if warnings:
        widgets.append(
            {
                "id": "split_warnings",
                "type": "insight",
                "title": "Warnings",
                "priority": 99,
                "data": {"observations": warnings},
                "config": {"color": "#f59e0b"},
            }
        )

    return {
        "widgets": widgets,
        "layout_template": "grid" if len(widgets) > 2 else "two_column",
        "version": 1,
        "follow_ups": _get_follow_ups("ml_train_test_split"),
    }


def _decompose_correlation(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose ml_correlation_matrix results into stat card + heatmap + table."""
    widgets: List[Dict[str, Any]] = []
    priority = 0

    top_pairs = result.get("top_pairs", [])
    features = result.get("features", [])
    n_features = result.get("n_features", 0)
    sample_size = result.get("sample_size", 0)
    total_available = result.get("total_available", 0)
    sampling_method = result.get("sampling_method", "random")

    # Stat card: strongest pair
    if top_pairs:
        strongest = top_pairs[0]
        rho = strongest.get("spearman_rho")
        r = strongest.get("pearson_r")
        stats = {
            "Strongest pair": f"{strongest['feature_a']} ↔ {strongest['feature_b']}",
            "Spearman ρ": round(rho, 4) if rho is not None else "N/A",
            "Pearson r": round(r, 4) if r is not None else "N/A",
            "Features": n_features,
            "Sample": f"{sample_size} of {total_available} ({sampling_method})",
        }
        widgets.append(
            _build_stat_widget(
                "correlation_stats", "Correlation Summary", stats, priority
            )
        )
        priority += 1

    # Heatmap
    spearman_matrix = result.get("spearman_matrix")
    pearson_matrix = result.get("pearson_matrix")
    matrix = spearman_matrix or pearson_matrix
    if matrix and features:
        widgets.append(
            {
                "id": "correlation_heatmap",
                "type": "correlation_heatmap",
                "title": f"Feature Correlation Matrix (n={sample_size})",
                "priority": priority,
                "data": {
                    "type": "correlation_heatmap",
                    "title": f"Feature Correlation Matrix (n={sample_size})",
                    "labels": features,
                    "matrix": matrix,
                    "description": "Pairwise correlations between audio features. Color intensity = strength.",
                },
                "config": {},
            }
        )
        priority += 1

    # Top pairs table
    if top_pairs:
        widgets.append(
            _build_table_widget(
                "correlation_top_pairs", "Top 10 Strongest Pairs", top_pairs, priority
            )
        )

    return {
        "widgets": widgets,
        "layout_template": "grid",
        "version": 1,
        "follow_ups": _get_follow_ups("ml_correlation_matrix"),
    }


def _decompose_statistical_test(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose ml_statistical_test results into stat card + bar + table."""
    widgets: List[Dict[str, Any]] = []
    priority = 0

    plain_english = result.get("plain_english", "")
    cohens_d = result.get("cohens_d", 0)
    effect_label = result.get("effect_size", "negligible")
    group_a = result.get("group_a", {})
    group_b = result.get("group_b", {})

    stats = {
        "Result": plain_english,
        "Effect size": f"d = {cohens_d} ({effect_label})",
        f"{group_a.get('name', 'A')} n": group_a.get("n", 0),
        f"{group_b.get('name', 'B')} n": group_b.get("n", 0),
    }
    widgets.append(
        _build_stat_widget("stat_test_card", "Statistical Test", stats, priority)
    )
    priority += 1

    # Group comparison bar
    a_name = group_a.get("name", "Group A")
    b_name = group_b.get("name", "Group B")
    widgets.append(
        {
            "id": "stat_test_bar",
            "type": "bar_chart",
            "title": f"{a_name} vs {b_name}",
            "priority": priority,
            "data": {
                "type": "bar_chart",
                "title": f"{a_name} vs {b_name}",
                "labels": [a_name, b_name],
                "data": [group_a.get("mean", 0), group_b.get("mean", 0)],
                "description": f"Mean comparison — {plain_english}",
            },
            "config": {},
        }
    )
    priority += 1

    # Test details table
    tt = result.get("welch_t_test", {})
    mw = result.get("mann_whitney_u", {})
    test_rows = [
        {
            "test": "Welch's t-test",
            "statistic": tt.get("t_statistic"),
            "p_value": tt.get("p_value"),
        },
        {
            "test": "Mann-Whitney U",
            "statistic": mw.get("u_statistic"),
            "p_value": mw.get("p_value"),
        },
        {"test": "Cohen's d", "statistic": cohens_d, "interpretation": effect_label},
        {
            "test": f"{a_name} mean ± std",
            "statistic": f"{group_a.get('mean', 0)} ± {group_a.get('std', 0)}",
            "p_value": None,
        },
        {
            "test": f"{b_name} mean ± std",
            "statistic": f"{group_b.get('mean', 0)} ± {group_b.get('std', 0)}",
            "p_value": None,
        },
    ]
    widgets.append(
        _build_table_widget("stat_test_details", "Test Details", test_rows, priority)
    )
    priority += 1

    # Warnings
    warnings = result.get("warnings", [])
    if warnings:
        widgets.append(
            {
                "id": "stat_test_warnings",
                "type": "insight",
                "title": "Warnings",
                "priority": 99,
                "data": {"observations": warnings},
                "config": {"color": "#f59e0b"},
            }
        )

    return {
        "widgets": widgets,
        "layout_template": "grid",
        "version": 1,
        "follow_ups": _get_follow_ups("ml_statistical_test"),
    }


def _decompose_export_features(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose ml_export_features into stat card + table."""
    widgets: List[Dict[str, Any]] = []
    priority = 0

    stats = {
        "Total rows": result.get("total_rows", 0),
        "Exported rows": result.get("exported_rows", 0),
        "Labeled rows": result.get("labeled_rows", 0),
        "Features": result.get("feature_count", 0),
        "Format": result.get("format", "flat"),
    }
    widgets.append(
        _build_stat_widget("export_stats", "Feature Export", stats, priority)
    )
    priority += 1

    rows = result.get("rows", [])
    columns = result.get("columns_meta", [])
    if rows and columns:
        widgets.append(
            _build_table_widget(
                "export_table",
                "Exported Data (Preview)",
                rows[:20],
                priority,
                total_count=len(rows),
            )
        )

    return {
        "widgets": widgets,
        "layout_template": "single",
        "version": 1,
        "follow_ups": _get_follow_ups("ml_export_features"),
    }


def _decompose_feature_importance(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose ml_feature_importance results into stat card + bar + table."""
    widgets: List[Dict[str, Any]] = []
    priority = 0

    features = result.get("features", [])
    estimator = result.get("estimator", "mutual_info_classif")
    n_samples = result.get("n_samples", 0)
    n_classes = result.get("n_classes", 0)

    stats = {
        "Samples": n_samples,
        "Classes": n_classes,
        "Top feature": features[0]["feature"] if features else "N/A",
        "Estimator": estimator,
    }
    widgets.append(
        _build_stat_widget("fi_stats", "Feature Importance", stats, priority)
    )
    priority += 1

    # Feature importance bar chart
    if features:
        top_features = features[:10]
        widgets.append(
            {
                "id": "feature_importance_bar",
                "type": "feature_importance_bar",
                "title": f"Top Features for {result.get('label_column', 'label')}",
                "priority": priority,
                "data": {
                    "type": "feature_importance_bar",
                    "title": f"Top Features for {result.get('label_column', 'label')}",
                    "labels": [f["feature"] for f in top_features],
                    "data": [f["mi_score"] for f in top_features],
                    "f_scores": [f["f_score"] for f in top_features],
                    "description": f"Mutual information scores (k-NN, n_neighbors=3). Higher = more predictive.",
                },
                "config": {},
            }
        )
        priority += 1

    # Full table
    if features:
        widgets.append(
            _build_table_widget("fi_table", "All Features (Ranked)", features, priority)
        )
        priority += 1

    # Caveats
    warnings = result.get("warnings", [])
    if warnings:
        widgets.append(
            {
                "id": "fi_warnings",
                "type": "insight",
                "title": "Caveats",
                "priority": 99,
                "data": {"observations": warnings},
                "config": {"color": "#f59e0b"},
            }
        )

    return {
        "widgets": widgets,
        "layout_template": "grid",
        "version": 1,
        "follow_ups": _get_follow_ups("ml_feature_importance"),
    }


def _decompose_recent(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose recent datasets results."""
    rows = result.get("rows", [])
    analysis_type = result.get("analysis_type", "")
    total = result.get("total_count", len(rows))

    stats = {
        "Total datasets": total,
        "Showing": len(rows),
    }
    widgets = [
        _build_stat_widget(f"stats_{analysis_type}", "Dataset Summary", stats, 0)
    ]
    widgets.append(
        _build_table_widget(f"table_{analysis_type}", "Recent Datasets", rows, 1)
    )
    return {
        "widgets": widgets,
        "layout_template": "single",
        "version": 1,
        "follow_ups": _get_follow_ups(analysis_type),
    }


def _decompose_ml_feature_stats(result: Dict[str, Any]) -> Dict[str, Any]:
    """Decompose ml_feature_stats results into stat card + variance bar + table."""
    widgets: List[Dict[str, Any]] = []
    priority = 0

    audio_features = result.get("audio_features", {})
    noise_analysis = result.get("noise_analysis", {})

    # Stat card: key counts and ranges
    stats: Dict[str, Any] = {}
    if audio_features:
        stats["Feature rows"] = audio_features.get("count", 0)
        avg_duration = audio_features.get("avg_duration")
        if avg_duration is not None:
            stats["Avg Duration (s)"] = round(avg_duration, 2)
        avg_rms = audio_features.get("avg_rms")
        if avg_rms is not None:
            stats["Avg RMS Energy"] = round(avg_rms, 4)
        avg_centroid = audio_features.get("avg_centroid")
        if avg_centroid is not None:
            stats["Avg Spectral Centroid"] = round(avg_centroid, 1)
    if noise_analysis:
        stats["Analysis rows"] = noise_analysis.get("count", 0)
        avg_db = noise_analysis.get("avg_mean_db")
        if avg_db is not None:
            stats["Avg Decibel"] = round(avg_db, 1)
        max_db = noise_analysis.get("max_db")
        if max_db is not None:
            stats["Max Decibel"] = round(max_db, 1)
        min_db = noise_analysis.get("min_db")
        if min_db is not None:
            stats["Min Decibel"] = round(min_db, 1)

    if stats:
        widgets.append(
            _build_stat_widget(
                "ml_feature_stats_card", "Feature Statistics", stats, priority
            )
        )
        priority += 1

    # Variance ranking: horizontal bar of std devs
    variance_items: List[Dict[str, Any]] = []
    feature_labels = [
        ("avg_duration", "std_duration", "Duration"),
        ("avg_rms", "std_rms", "RMS Energy"),
        ("avg_centroid", "std_centroid", "Spectral Centroid"),
        ("avg_bandwidth", "std_bandwidth", "Spectral Bandwidth"),
        ("avg_zcr", "std_zcr", "Zero Crossing Rate"),
    ]
    for _avg_key, std_key, label in feature_labels:
        std_val = audio_features.get(std_key)
        if std_val is not None:
            variance_items.append({"label": label, "value": round(std_val, 4)})

    if variance_items:
        variance_items.sort(key=lambda x: x.get("value", 0) or 0, reverse=True)
        widgets.append(
            {
                "id": "ml_feature_variance",
                "type": "horizontal_bar_chart",
                "title": "Feature Variance (Std Dev)",
                "priority": priority,
                "data": {
                    "type": "horizontal_bar_chart",
                    "title": "Feature Variance (Std Dev)",
                    "labels": [item["label"] for item in variance_items],
                    "data": [item["value"] for item in variance_items],
                    "description": "Standard deviation of audio features — higher variance means more spread",
                },
                "config": {},
            }
        )
        priority += 1

    # Decibel variance
    if noise_analysis:
        db_std = noise_analysis.get("std_mean_db")
        if db_std is not None:
            db_items = [{"label": "Decibel (mean_db)", "value": round(db_std, 2)}]
            widgets.append(
                {
                    "id": "ml_noise_variance",
                    "type": "horizontal_bar_chart",
                    "title": "Noise Analysis Variance",
                    "priority": priority,
                    "data": {
                        "type": "horizontal_bar_chart",
                        "title": "Noise Analysis Variance",
                        "labels": ["Decibel (mean_db)"],
                        "data": [round(db_std, 2)],
                        "description": "Standard deviation of decibel measurements",
                    },
                    "config": {},
                }
            )
            priority += 1

    # Table: all stats
    all_stats_rows: List[Dict[str, Any]] = []
    for key, value in {**audio_features, **noise_analysis}.items():
        if value is not None:
            all_stats_rows.append(
                {
                    "metric": key,
                    "value": round(value, 4) if isinstance(value, float) else value,
                }
            )
    if all_stats_rows:
        widgets.append(
            _build_table_widget(
                "ml_feature_stats_table", "Full Statistics", all_stats_rows, priority
            )
        )

    return {
        "widgets": widgets,
        "layout_template": "grid" if len(widgets) > 2 else "two_column",
        "version": 1,
        "follow_ups": _get_follow_ups("ml_feature_stats"),
    }
