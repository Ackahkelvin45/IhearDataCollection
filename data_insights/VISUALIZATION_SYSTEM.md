# Visualization System — Data Insights App

## Overview

The visualization pipeline turns natural language queries about audio data into rendered charts, tables, stat cards, and dashboards. It spans four layers: **tools → chart builder → widget composer → frontend renderer**. Simple queries produce a single chart; complex queries (like "give me an overview") automatically decompose into multi-widget dashboards.

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│  TOOLS (tools.py)                                            │
│  AudioAnalysisTool / DataAnalysisTool / ML tools             │
│  Returns structured dicts with rows, columns, chart_hint     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  CHART BUILDER (chart_builder.py)                            │
│  Pure-Python decision tree. No LLM.                          │
│  auto_detect_axes() → select_chart_type() → build_chart()    │
│  Fallback: _build_table_config() when axes can't be resolved │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  WIDGET COMPOSER (widget_composer.py)                        │
│  Wraps single charts into artifact format.                   │
│  Decomposes multi-view tool outputs into widget arrays.      │
│  Output: {widgets: [...], layout_template, version}          │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  SSE STREAMING (views.py)                                    │
│  Emits "visualization" (single chart) + "dashboard" (multi)  │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  FRONTEND RENDERER (unified_chat.html)                       │
│  renderArtifact() — multi-widget dashboards                  │
│  addVisualizationToMessage() — single charts (legacy)        │
│  createChart() — Chart.js config builder                     │
│  buildTableHtml() — table fallback                           │
└──────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Tools (data producers)

**File:** `data_insights/workflows/tools.py`

Each tool method returns a structured dictionary. The key convention is:

| Key | Purpose |
|---|---|
| `rows` | Array of row dicts — primary data |
| `columns` | Column names in order |
| `chart_hint` | `{x, y, group_by, type}` — guides chart selection |
| `analysis_type` | String identifier used by widget composer |
| `skip_visualization` | If `true`, no chart is produced |

### Single-view tools (1 chart per result)

| Tool | analysis_type | Shape |
|---|---|---|
| `_recent_datasets` | `recent_datasets` | `{rows, columns, chart_hint}` |
| `_top_collectors_monthly` | `top_collectors_monthly` | `{rows, columns, chart_hint}` |
| `_category_class_count` | `category_class_count` | `{rows, columns, chart_hint}` |
| `_group_count` | `group_count` | `{rows, columns, chart_hint}` |
| `_dataset_count` | `dataset_count` | `{total_count}` — scalar only, `skip_visualization: true` |
| `_decibel_ranked` | `highest_decibel` / `lowest_decibel` | `{rows, columns, chart_hint}` |
| `_decibel_grouped` | `avg_decibel_by_{group}` | `{rows, columns, chart_hint}` |
| `_energy_analysis` | `energy_analysis` | `{results, analysis_type}` |
| `_spectral_analysis` | `spectral_analysis` | `{results, analysis_type}` |
| `_frequency_analysis` | `frequency_analysis` | `{results, analysis_type}` |
| `_correlation_analysis` | `correlation_analysis` | `{correlations}` — scatter data |
| `_statistical_analysis` | `statistical_analysis` / `statistical_distribution` | `{results / distribution_data}` |

### Multi-view tools (auto-decomposed into dashboards)

| Tool | analysis_type | Natural widgets |
|---|---|---|
| `_overview_analysis` | `overview_analysis` | 3 — stat_card (totals), bar_chart (regional), pie_chart (categories) |
| `_temporal_analysis` | `temporal_analysis` | 2 — line_chart (monthly), line_chart (daily) |
| `MLDatasetProfileTool` | `ml_dataset_profile` | 4 — stat_card (totals), progress_bar (coverage), progress_bar (completeness), bar_chart (label distribution) |
| `MLFeatureStatsTool` | `ml_feature_stats` | 2 — stat_card (audio features), stat_card (noise analysis) |

---

## Layer 2: Chart Builder (deterministic chart selection)

**File:** `data_insights/workflows/chart_builder.py`

A pure-Python decision tree with zero LLM calls. Deterministic, instant, debuggable.

### Axis detection: `auto_detect_axes(rows, columns)`

Priority order:
1. **Temporal columns** first: any column containing `date`, `time`, `month`, `year`, `day`, `week` in its name
2. **Categorical columns**: non-numeric, non-temporal
3. **Two numeric columns**: used for scatter plots

### Chart type selection: `select_chart_type(rows, columns, hint)`

| Condition | Chart type |
|---|---|
| Explicit `hint.type` set | Uses that type directly |
| Temporal X + numeric Y, ≤ 7 rows | `bar_chart` |
| Temporal X + numeric Y, 8+ rows | `line_chart` |
| Two numeric axes (X and Y both numeric) | `scatter_plot` |
| Ratio data (0–1 or %), ≤ 6 categories | `pie_chart` |
| Ratio data, 7+ categories | `donut_chart` |
| Numeric Y + categorical X, ≤ 5 categories | `bar_chart` |
| Numeric Y + categorical X, 6–12 categories | `horizontal_bar_chart` |
| Numeric Y + categorical X, 13+ categories | Falls back to **table** |
| No axes detected but rows exist | Falls back to **table** |

### Fallback system

When chart axes can't be resolved, `_build_table_config()` produces a guaranteed-renderable table with columns, rows, and pagination metadata. Every query produces *something* the frontend can render.

### Output format (single chart)

```json
{
    "visualization_type": "bar_chart",
    "visualization_name": "Bar Chart",
    "frontend_data": {
        "type": "bar_chart",
        "title": "Top Collectors",
        "labels": ["Alice", "Bob", "Carol"],
        "data": [42, 35, 28],
        "colors": null,
        "description": "Bar Chart for Dataset Count by Collector",
        "table": {
            "columns": ["collector", "dataset_count"],
            "rows": [...],
            "pagination": {"limit": 20, "offset": 0, "has_more": false}
        }
    }
}
```

---

## Layer 3: Widget Composer (multi-widget dashboards)

**File:** `data_insights/workflows/widget_composer.py`

Two entry points:

### `wrap_as_artifact(chart_config)` — Single widget wrapper

Wraps any single chart into the standard artifact format. Used for all non-multi-view tools.

### `decompose(result)` — Multi-widget decomposition

Routes by `analysis_type` and splits rich tool outputs into independent widgets. Each widget reuses `chart_builder.select_chart_type()` and `build_chart_config()`.

### Artifact format

```json
{
    "widgets": [
        {
            "id": "overview_stats",
            "type": "stat_card",
            "title": "Overview Statistics",
            "data": {"stats": {"total_datasets": 1542, "avg_decibel": 72.3, ...}},
            "config": {},
            "priority": 0
        },
        {
            "id": "regional_breakdown",
            "type": "bar_chart",
            "title": "Regional Breakdown",
            "data": {
                "type": "bar_chart",
                "title": "Regional Breakdown",
                "labels": ["Accra", "Kumasi", "Tamale"],
                "data": [520, 340, 210],
                "colors": null,
                "description": "..."
            },
            "config": {},
            "priority": 1
        }
    ],
    "layout_template": "grid",
    "version": 1
}
```

### Layout templates

| Template | CSS | Used when |
|---|---|---|
| `single` | block display | 1 widget |
| `two_column` | `grid-template-columns: 1fr 1fr` | 2 widgets |
| `grid` | `repeat(auto-fill, minmax(280px, 1fr))` | 3+ widgets |

---

## Layer 4: Frontend Renderer

**File:** `data_insights/templates/data_insights/unified_chat.html`

All visualization logic lives in the `UnifiedAudioDataChat` class (inline script, ~3500 line template).

### SSE event handling

| Event | Handler | Purpose |
|---|---|---|
| `visualization` | `addVisualizationToMessage()` | Single chart (legacy path, still emitted for backward compat) |
| `dashboard` | `renderArtifact()` | Multi-widget artifact (new path, Phase 1+) |

### `renderArtifact(messageId, artifact, toolResponses)`

Iterates `artifact.widgets[]` and renders each widget type:

| Widget type | Rendering |
|---|---|
| `stat_card` | Blue gradient card with key-value stats in a responsive grid |
| `progress_bar` | Labeled progress bars with color-coded fill (green > 80%, yellow > 50%, red < 50%) |
| `table` | Delegates to `buildTableHtml()` — scrollable table with header |
| `bar_chart`, `line_chart`, `pie_chart`, `scatter_plot`, etc. | Canvas element + deferred `createChart()` call with Chart.js |

### `createChart(visualizationData, chartId, toolResponses)`

Switch statement mapping chart type strings to Chart.js config objects:

| Type | Chart.js type | Notes |
|---|---|---|
| `pie_chart` | `pie` | Background color array, legend at bottom |
| `bar_chart` | `bar` | Supports multi-dataset (stacked) or single-dataset |
| `line_chart` | `line` | `tension: 0.4`, `fill: false` |
| `scatter_plot` | `scatter` | Uses `{x, y}` point objects |
| `box_plot` | `boxplot` (plugin) | Requires `chartjs-chart-boxplot` CDN plugin |
| `area_chart` | `line` with `fill: true` | Same as line but filled |
| `heatmap` | `bar` (fallback) | No native Chart.js heatmap |

### `extractChartData(visualizationData, toolResponses)`

Heavy data extraction logic (~275 lines) that normalizes diverse tool output shapes into `{labels, data}` arrays. Handles:
- Direct `frontend_data.labels` / `frontend_data.data`
- `statistical_distribution` → box plot quartile computation
- `category_class_count` → stacked bar datasets
- `correlation_analysis` → scatter `{x, y}` points
- `energy_analysis`, `spectral_analysis`, `temporal_analysis` → label/value extraction
- String response regex parsing (last resort)

### `buildTableHtml(tableData)`

Renders a styled HTML table with header, scrollable body (max 260px), and proper cell escaping. Supports pagination buttons for large result sets.

### Deduplication

- Legacy path: `data-viz-name` attribute on `.viz-card` elements
- New artifact path: `data-widget-id` attribute on `.widget-card` elements
- Both prevent double-rendering of the same visualization

---

## Widget Types Reference

| Type | Visual | Used for |
|---|---|---|
| `stat_card` | Blue gradient card, large numbers with labels | Aggregate values (counts, averages, totals) |
| `progress_bar` | Horizontal bar with percentage | Coverage metrics, completeness, ratios |
| `bar_chart` | Vertical bars | Rankings, categorical comparisons |
| `horizontal_bar_chart` | Horizontal bars | Categories with long labels |
| `line_chart` | Connected line with tension | Time series, trends |
| `scatter_plot` | Point cloud | Correlations, relationships |
| `pie_chart` | Segmented circle | Ratios, proportions (≤ 6 categories) |
| `donut_chart` | Ring chart | Ratios, proportions (7+ categories) |
| `area_chart` | Filled line | Cumulative trends |
| `box_plot` | Box-and-whisker | Distributions, quartiles, outliers |
| `table` | Scrollable data table | Raw data, fallback when chart can't be built |

---

## Dashboard Save

**Model:** `Dashboard` (in `data_insights/models.py`)

| Field | Type | Purpose |
|---|---|---|
| `id` | UUID | Primary key |
| `user` | FK → User | Owner |
| `session` | FK → ChatSession (nullable) | Source conversation |
| `message` | FK → ChatMessage (nullable) | Source message |
| `title` | CharField(255) | User-given name |
| `slug` | SlugField(255, unique) | URL-safe identifier |
| `artifact_spec` | JSONField | Full `{widgets, layout_template, version}` spec |
| `thumbnail` | TextField (nullable) | Base64 JPEG from `canvas.toDataURL()` |
| `is_public` | BooleanField | Shareable toggle |
| `created_at` / `updated_at` | DateTime | Auto timestamps |

**Endpoint:** `POST /insights/sessions/{session_id}/messages/{message_id}/save-dashboard/`

**Thumbnail:** Generated client-side via `canvas.toDataURL('image/jpeg', 0.5)` — no external dependencies.

---

## Backward Compatibility

The system maintains two parallel rendering paths:

1. **Legacy path** (`pending_chart` + `"visualization"` SSE event + `addVisualizationToMessage()`): Handles all single-chart tool outputs. Untouched by the new system.

2. **Artifact path** (`pending_artifact` + `"dashboard"` SSE event + `renderArtifact()`): Handles multi-widget dashboards and wraps single charts into consistent format.

Old-format messages in the database (saved before this update) continue to render via the legacy path. The typo-bug (`visulization` → `visualization`) in message loading was fixed so saved visualizations now restore correctly.

---

## Key Design Decisions

- **Chart type selection is deterministic.** No LLM calls for visualization — `select_chart_type()` is a pure Python function.
- **Fallback is guaranteed.** Every query produces at least a data table, even when chart axes can't be resolved.
- **Multi-widget is deterministic too.** Known multi-view tool outputs are decomposed by `widget_composer.decompose()` without LLM involvement. Phase 4 (LLM-driven composition) is deferred.
- **No Redis.** The system doesn't need caching — tool queries are fast (< 100ms for ORM-based tools), and LangGraph's PostgresSaver handles conversation persistence.
- **No html2canvas.** Thumbnails use Chart.js's built-in `canvas.toDataURL()`.
- **Inline frontend.** All JS is in a single template file — no bundler, no npm, no build step.
