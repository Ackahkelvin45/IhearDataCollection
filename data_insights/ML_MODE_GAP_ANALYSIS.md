# ML Chatbot Mode: Current State & Improvement Plan

## Purpose

This doc describes what the `ml` mode actually does today, how it compares to `analysis` mode, and a concrete plan to close the gap. It was reviewed by an LLM, whose criticisms are incorporated inline.

---

## 1. Architecture overview (both modes)

The chatbot is a LangGraph agent:

| Component | File |
|---|---|
| `DataAgent` | `workflows/agent_workflow.py` — LangGraph state machine: cleanup → clarification_gate → agent ↔ tools → finalize_dashboard → format |
| System prompts | `workflows/prompt.py` — `SYSTEM_TEMPLATE` (analysis, 82 lines) and `ML_SYSTEM_TEMPLATE` (ml, 22 lines) |
| Tools | `workflows/tools.py` — LangChain BaseTool impls querying Django ORM + a few numpy/scipy computations |
| Chart builder | `workflows/chart_builder.py` — deterministic chart-type selection from data shape |
| Widget composer | `workflows/widget_composer.py` — tool results → multi-widget dashboard artifacts |
| Clarification gate | `workflows/clarification_gate.py` — ambiguity detection → structured questions |
| SQL agent | `workflows/sql_agent.py` — Text-to-SQL fallback |
| Views | `views.py` — REST endpoints, agent instantiation per mode, SSE streaming |

The LangGraph graph is identical for both modes. Only the system prompt and tool list differ.

---

## 2. What `analysis` mode has (baseline)

### Tools (6)

| Tool | What it does |
|---|---|
| `data_analysis` | Dataset counts, decibel rankings, groupings, recent datasets, top collectors, SQL fallback |
| `analyze_audio_data` | 7 branches: energy, spectral, frequency, correlation, statistical, temporal, overview — with auto-detection and V2 ambiguity routing |
| `search_noise_datasets` | Filtered search by name/region/community/category/date with pagination |
| `search_audio_features` | Numeric threshold filtering (RMS, ZCR, spectral centroid, bandwidth, duration) |
| `get_noise_dataset_details` | Single-record deep dive: metadata, audio features, noise analysis |
| `WebFetchTool` | External URL fetching |

### System prompt: 82 lines

Tool selection guide, audio term glossary (decibel vs frequency vs spectral vs energy), 11 few-shot examples, data context, explicit rules.

### Dashboard / visualization

Multi-widget decomposition for overview, temporal, grouped, ranked, and statistical results. `finalize_dashboard` runs a structured LLM call for summary + 3 observations. Chart types: bar, horizontal bar, line, scatter, pie, donut — auto-selected by decision tree.

---

## 3. What `ml` mode has (current state)

### Tools (5)

| Tool | What it actually does |
|---|---|
| `ml_dataset_profile` | `SELECT COUNT(*)` variants: totals, null counts per field, grouped label counts |
| `ml_feature_stats` | `AVG`/`STDDEV` on 5 audio features + optional decibel aggregates |
| `data_analysis` | Same as analysis mode |
| `search_noise_datasets` | Same as analysis mode |
| `search_audio_features` | Same as analysis mode |

**Missing from ml mode**: `analyze_audio_data`, `get_noise_dataset_details`, `WebFetchTool`

### System prompt: 22 lines

No glossary, no few-shot examples, no parameter guidance, no tool selection guide. The prompt tells the LLM to "recommend train/val/test splits" but gives it no tool to compute them.

### Visualization

`_decompose_ml_profile` renders stat card + progress bars + bar chart. `ml_feature_stats` has no decompose function — falls through to single-chart via `wrap_as_artifact`.

### Bugs in production

- **`vague_ranking_metric` offers "Complaint rate" as an option** ([clarification_gate.py:194](data_insights/workflows/clarification_gate.py#L194)). There is no complaint model anywhere in this schema. If a user picks it, the downstream tool call will fail or return nothing.
- **`ml_feature_stats` has no decomposition** — returns a JSON blob with no dashboard widget.

---

## 4. The gap

### 4.1 No actual ML computation

Both ML tools are pure SQL aggregates. There is zero of:

| Capability | Why it matters |
|---|---|
| Train/val/test split calculation | The prompt tells the LLM to recommend splits but gives it no tool |
| Class imbalance severity | Raw counts but no minority ratio, entropy, or severity flag |
| Feature correlation | No pairwise correlation between numeric audio features |
| Feature importance ranking | No scoring of which features best separate classes |
| Statistical significance tests | Group comparisons return means, no t-test/ANOVA |
| Outlier detection | No Z-score or IQR flagging |

### 4.2 Correlation analysis is brittle and not in ML mode

`_correlation_analysis` in `AudioAnalysisTool` (excluded from ML mode) handles only two hardcoded keyword pairs. No general correlation matrix exists anywhere.

### 4.3 No ML-native visualizations

Missing: correlation heatmap, feature importance bar, class distribution chart, box plot (in labels dict but never returned by `select_chart_type`).

### 4.4 Clarification gate knows nothing about ML

All 5 ambiguity signals are analysis-specific. Missing ML signals: which label column to use, what split strategy.

### 4.5 Dashboard finalizer prompt is analysis-only

`_DASHBOARD_SUMMARY_PROMPT`: "You are a data analyst reviewing a dashboard." ML mode needs its own summary framing.

### 4.6 ML mode drops tools the ML engineer needs

`AudioAnalysisTool` (7 analysis branches for feature exploration) and `NoiseDetailTool` are excluded. Starting ML investigation means you can't explore spectral/energy/frequency distributions.

### 4.7 System prompt gap

22 lines vs 82 lines. No few-shot, no glossary, no tool selection guidance.

---

## 5. Design decisions (resolved before implementation)

### 5.1 Scope boundary: what "ML mode" means

The original doc said "do not train models" but then proposed PCA and K-means — which are models. The correct boundary is:

> **In scope**: deterministic statistical computation on CPU, including unsupervised dimensionality reduction and clustering used for exploration. Tools compute and surface information; the engineer decides what to do with it.
>
> **Out of scope**: supervised model fitting (training a classifier/regressor), inference, hyperparameter tuning, AutoML. Those belong in notebooks and training pipelines.

This makes PCA and K-means explicitly in scope (unsupervised, used for exploration) while keeping the line clear against predictive modeling.

### 5.2 Why a separate mode exists

ML mode is not "analysis mode plus more tools." It exists because three things differ:

1. **Framing**: results are interpreted through an ML lens (class balance, feature quality, data leakage risk, modeling readiness)
2. **Defaults**: different clarification questions, different chart priorities, different summary tone
3. **Audience**: assumes ML vocabulary (entropy, stratification, effect size, explained variance)

Tools are shared where they serve both audiences (`data_analysis`, `search_noise_datasets`, `AudioAnalysisTool`). ML-specific tools (`ml_class_balance`, `ml_train_test_split`, etc.) are the delta. The tool list is maintained once; mode determines which tools are bound and what system prompt is used.

**Decision**: unify the tool list. Both modes get all tools. Mode controls prompt + finalizer + clarification defaults only. Delete `get_agent_tools(mode="ml")` as a separate list; instead, filter by tool tags or just bind everything and let the prompt route.

### 5.3 Statistical correctness policy

Statistical tools must not silently return misleading numbers. Every tool that computes a statistic must also return diagnostics sufficient to judge whether the statistic is trustworthy:

- `ml_correlation_matrix`: return **both Spearman ρ and Pearson r** side by side, with p-values for each. Spearman is the default sort order (robust to outliers and monotone non-linearity — audio features routinely show log-normal or heavy-tailed distributions where Pearson misleads). Accept a `method` parameter (`"spearman"` / `"pearson"` / `"both"`, default `"both"`). The tool explicitly labels which coefficient it's showing and what each measures ("Spearman captures monotonic relationships; Pearson captures linear relationships").

- `ml_feature_importance`: use sklearn's **`mutual_info_classif`** with the default k-NN estimator (Kraskov, `n_neighbors=3`). Return MI score as primary sort order + ANOVA F-value as secondary cross-check. Include in output: estimator used, `n_neighbors` value, and a **small-sample caveat** when any class has fewer than ~500 samples (MI estimates are biased upward on small data with the k-NN estimator). The decompose function renders this caveat visibly in the insight widget, not buried in JSON.

- `ml_statistical_test`: **always run Welch's t-test**. Do not auto-switch between t-test and Mann-Whitney based on a normality pretest (known statistical anti-pattern — inflates Type I error). Welch's is robust to unequal variance and reasonably robust to non-normality at moderate n. Report both t-test and Mann-Whitney results side by side; let the engineer decide. Decompose surfaces the most important diagnostic prominently (sample size warning if either group < 10).

- `ml_train_test_split`: flag minority classes with **fewer than 30 samples in val/test combined** (heuristic, documented). Warn when stratification is impossible because a class has 1 sample.

### 5.4 Determinism

Every tool with randomness (e.g., shuffle in train/test split) must expose an explicit `seed` parameter with a fixed default (e.g., `seed=42`). The seed value used is returned in the tool output so the user can reproduce results. Two calls with the same seed and same data produce identical output.

---

## 6. Improvement plan

### Phase 0: Bug fixes (ship before anything else)

These are bugs in production. Phase 0 lands **before** the Phase 1 prompt rewrite, because the prompt rewrite will reference fixed behavior that doesn't exist until these are deployed.

1. **Remove "Complaint rate" from `vague_ranking_metric` options** in [clarification_gate.py:194](data_insights/workflows/clarification_gate.py#L194). Replace with "Dominant frequency".

2. **Add `_decompose_ml_feature_stats` to `widget_composer.py`** — stat card (feature count, avg values) + horizontal bar (feature variance ranking) + data table.

### Phase 1: Foundation — unification + prompt + restructure (2-3 days)

**Goal**: eliminate the two-tool-list architecture, give the LLM proper guidance, restore dropped tools.

3. **Unify the tool list**: both modes get all tools. Delete the `mode` parameter from `get_agent_tools()`. Mode controls prompt + finalizer + clarification defaults only. This means `AudioAnalysisTool`, `NoiseDetailTool`, and `WebFetchTool` are available in ML mode — which is correct, since an ML engineer needs feature exploration.

4. **Add `list_ml_schema` tool** — returns available label columns, feature columns, metadata columns, and current row counts. The LLM calls this once when it needs to reference columns by name. Eliminates hallucinated column names on schema changes.

5. **Rewrite `ML_SYSTEM_TEMPLATE`** to match analysis template depth (80+ lines, not 22):
   - Tool selection guide: which tool for which ML question (including the analysis tools now available)
   - Instruction to call `list_ml_schema` before any tool that references columns by name
   - ML glossary: coverage, missingness, class balance/imbalance, entropy, feature importance, stratification, train/val/test split, effect size, Spearman ρ (monotonic) vs Pearson r (linear), mutual information
   - 10-12 few-shot examples showing exact parameter patterns for every ML tool
   - Data context: what AudioFeature, NoiseAnalysis, and Dataset models contain
   - Rules: never guess splits, always compute from real data, flag imbalance explicitly, warn about data leakage when stats are computed before splitting

6. **ML-aware dashboard finalizer**: in `finalize_dashboard`, detect session mode and use an ML-specific summary prompt. Done alongside the template rewrite — doesn't deserve its own phase.

### Phase 2: Class balance + train/test split + label clarification (2-3 days)

**Goal**: the highest-value, simplest-math ML tools — shipped as a complete vertical slice with decompose + chart types + the clarification signals those tools consume.

7. **`ml_class_balance` tool** — given a label column (`category`, `region`, `class`, `subclass`):
   - Per-class counts, percentages
   - Minority/majority ratio
   - Imbalance severity: severe (<5%), moderate (5-15%), mild (15-30%), balanced (>30%)
   - Shannon entropy
   - Stratified split recommendation: exact row counts for train/val/test per class (default 70/15/15)
   - Warn if any class has <30 samples in val+test combined
   - **Insufficient data contract**: if total rows < 10 or < 2 classes present, return `{"error": "insufficient_data", "reason": "..."}` instead of useless numbers

8. **`ml_train_test_split` tool** — given total count, label column, and split ratios:
   - Exact row counts for train/val/test
   - If stratified, per-class counts in each split
   - Seed parameter (default 42), returned in output
   - **Data leakage warning**: checks `_full_dataset_aggregates_computed` flag in agent state. If set, appends warning that stats were computed on full dataset before splitting (see Section 7.6)

9. **`ml_missing_label` clarification signal** — when the user says "classify", "predict", "which features separate X", or "balance of X" but doesn't specify which label column. Options built from available label columns (`category`, `class`, `subclass`, `region`). Consumed by `ml_class_balance`, `ml_train_test_split`, and (in Phase 4) `ml_feature_importance`. Ships with Phase 2 because Phase 2 tools are the first consumers.

10. **Add `class_distribution_bar` chart type to `chart_builder.py`** — grouped bar per class showing counts and percentages.

11. **Add decompose functions** in `widget_composer.py`:
    - `_decompose_class_balance` — stat card (entropy, minority ratio, severity badge colored red/yellow/green) + class distribution bar + per-class table. Severity and per-class sample warnings are rendered visibly, not buried in JSON.
    - `_decompose_train_test_split` — stat card (total rows per split) + split composition bar (stacked by class) + per-split table. Data leakage warning renders as a yellow banner at the top if the flag was set.

### Phase 3: Correlation matrix + statistical tests + export (3-4 days)

**Goal**: tools that compute relationships between features and between groups — with proper statistical care. Plus the export bridge to notebooks.

12. **`ml_correlation_matrix` tool**:
    - Pairwise **Spearman ρ and Pearson r** between all numeric audio feature columns. Both returned side by side with p-values. Spearman is the default sort order. Accept `method` parameter (`"spearman"` / `"pearson"` / `"both"`, default `"both"`).
    - Top-10 strongest pairs (by absolute value of the default sort method)
    - **Row cap**: 5,000 rows stratified by label if one is provided, else random with seed=42. Cap is documented in tool description; actual sample size and sampling method returned in output.
    - **Column cap**: if >12 numeric feature columns exist, the tool returns the full matrix but the decompose function triggers `ml_ambiguous_features` clarification to let the user narrow down the heatmap. The stat card still shows the strongest pair.
    - **Insufficient data**: if < 5 rows or < 2 features with non-null values, return error contract

13. **`ml_statistical_test` tool** — given two groups and a numeric feature:
    - Group means, stds, sample sizes
    - Welch's t-test: t-statistic, p-value, degrees of freedom
    - Mann-Whitney U: U-statistic, p-value
    - Cohen's d (effect size) with interpretation: negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)
    - Plain-English summary
    - **Small sample warning**: if either group has < 10 samples, surfaced visibly in decompose

14. **`ml_export_features` tool** — export X (feature matrix) + y (label vector) as CSV:
    - Selectable features, label column, date/region/category filters
    - Format: flat table (X and y in one CSV) or split (two CSVs)
    - **Row cap**: 100,000 rows (I/O-bound, not compute-bound)
    - This is the bridge from chatbot to notebook — the single feature that lets an ML engineer say "good enough, I'll take it from here"
    - Pulled forward from the deferred bucket because it has higher leverage and lower complexity than PCA/clustering

15. **Add `correlation_heatmap` chart type** — N×N matrix with color scale from -1 to +1, annotated with coefficient values for significant cells.

16. **Add decompose functions**:
    - `_decompose_correlation` — stat card (strongest pair shown as "Spearman ρ = X, Pearson r = Y") + correlation heatmap + table of top-10 pairs with both coefficients and p-values. Stat card shows sample size and sampling method. If >12 features, stat card suggests narrowing down.
    - `_decompose_statistical_test` — stat card (effect size, p-value, plain-English interpretation, sample size badge) + group comparison bar + full test output table
    - `_decompose_export` — stat card (row count, column count, file size estimate) + download button widget

### Phase 4: Feature importance (2-3 days)

**Goal**: help ML engineers understand which features matter for a given label.

17. **`ml_feature_importance` tool** — given a label column and optional feature list:
    - Primary: sklearn `mutual_info_classif` with k-NN estimator (`n_neighbors=3`, documented in output). MI score is the sort order.
    - Secondary: ANOVA F-value as cross-check, with normality flag per feature
    - **Sample cap**: 5,000 rows, stratified by label
    - **Caveats surfaced in decompose**: "k-NN MI estimator (n_neighbors=3). Scores may be biased upward for classes with <500 samples." If any class is below threshold, affected classes are listed visibly.
    - Note that MI scores are univariate (don't capture feature interactions) and are computed on the full dataset (no train/test separation)

18. **Add `feature_importance_bar` chart type** — horizontal bar sorted by importance score, with MI and F-value shown side by side.

19. **Add `_decompose_feature_importance`** — horizontal bar (all features with MI + F-value) + table with scores, normality flags, and the visible caveat about estimator and sample size.

### Phase 5: Remaining ML-specific clarification (1 day)

**Goal**: the remaining load-bearing ambiguity signals that weren't consumed by earlier phases.

Only add signals that directly change tool parameters:

20. **`ml_split_strategy`** — when the user asks for a split but doesn't specify how. Options: "Stratified (preserves class proportions)", "Random (uniform shuffle)", "Time-based (split by recording date)". Consumed by `ml_train_test_split` (Phase 2). Deferred to Phase 5 rather than shipping with Phase 2 because `ml_train_test_split` works with a sensible default (stratified) — the signal adds UX polish, not correctness. Ship it when the split tool has been used enough to know what users actually ask for.

21. **`ml_ambiguous_features`** (reconsidered from v1) — when `ml_correlation_matrix` or `ml_feature_importance` has >12 numeric feature columns available and the user hasn't specified a subset. Options built from `list_ml_schema` feature columns. Triggers when the decompose function detects too many features for a readable heatmap. Consumed by Phase 3 and Phase 4 tools. Deferred to Phase 5 because the tools default to "all features" correctly — the signal improves UX for high-dimensional datasets.

22. **Add ML normalization in `clarification_resolver_node`** for the new dimensions.

Do NOT add:
- `ml_evaluation_metric` (accuracy vs F1 vs precision/recall) — the chatbot doesn't train or evaluate models, so nothing consumes this answer. It would be a question with no downstream effect.
- `ml_missing_label` was shipped in Phase 2 — it's consumed by Phase 2 tools and needed to ship with its first consumer.

### Phase 6: Advanced analysis (deferred until demand exists)

These are specified but NOT scheduled. Build them only after Phases 0-4 are in users' hands and there's evidence they're asked for. `ml_export_features` was pulled forward to Phase 3 because it has higher leverage and lower complexity.

23. **`ml_pca_analysis`** — PCA on selected audio features. Explained variance per component, component loadings, 2D projection.

24. **`ml_clustering`** — K-means or DBSCAN on audio features. Cluster assignments, sizes, per-cluster feature means.

25. **`ml_outlier_detection`** — Z-score and IQR-based outlier flagging per feature.

---

## 7. Operational concerns (cross-cutting)

### 7.1 Cost and latency — per-tool row caps

A single 10,000-row cap across all tools is wrong. Correlation stabilizes well before 10k; MI estimation with k-NN is O(n log n) and 10k is slow. Per-tool caps with justification:

| Tool | Cap | Reason |
|---|---|---|
| `ml_correlation_matrix` | 5,000 rows | Correlation stabilizes by n≈1,000; 5k gives margin |
| `ml_feature_importance` (MI) | 5,000 rows | k-NN MI estimation is O(n log n) per feature; 5k balances speed and stability |
| `ml_statistical_test` | No cap (uses all filtered rows) | Statistical power scales with n; capping a t-test is counterproductive |
| `ml_class_balance` | No cap (uses all filtered rows) | Counts are cheap and exact counts matter |
| `ml_train_test_split` | No cap | Needs exact totals to compute split counts |
| `ml_pca_analysis` | 10,000 rows | PCA is O(min(n,d)² × max(n,d)); 10k is fine |
| `ml_clustering` | 10,000 rows | K-means and DBSCAN scale acceptably at 10k |
| `ml_export_features` | 100,000 rows | Export is I/O-bound; 100k CSV rows is a reasonable HTTP response |

**Sampling strategy**: when a cap is applied, rows are sampled randomly using a fixed seed (42, overridable). If a label column is provided to the tool, sampling is **stratified by label** to preserve class proportions in the sample. The actual sample size and sampling method are always returned in the tool output.

### 7.2 Failure modes

Every new tool must implement a consistent insufficient-data contract:

```json
{
  "error": "insufficient_data",
  "reason": "Only 3 rows found after filtering. Minimum is 10.",
  "rows_available": 3
}
```

This is non-negotiable. A tool that returns a correlation of 1.0 on 2 data points is worse than no tool.

### 7.3 Diagnostics must surface in rendered output

"Every tool returns diagnostics" is worthless if diagnostics live only in the JSON blob. Each decompose function must surface the single most important caveat **visibly** in the rendered widget. Examples:

- `_decompose_correlation`: stat card shows strongest pair AND sample size used. Heatmap is annotated. If rows were sampled, the stat card says "Computed on a stratified sample of 5,000 rows (full dataset: 142,000 rows)."
- `_decompose_class_balance`: stat card shows entropy AND imbalance severity badge (red/yellow/green). If any class has <30 samples in recommended val+test, that warning is rendered in the stat card, not buried.
- `_decompose_feature_importance`: stat card shows top feature AND a caveat line: "k-NN MI estimator (n_neighbors=3). Scores may be biased upward for classes with <500 samples." If any class is below threshold, the affected classes are listed.
- `_decompose_statistical_test`: stat card headline is the plain-English interpretation ("Significant, medium effect"). Sample sizes shown prominently. If n<10 for either group, a warning badge renders: "Low sample size — results are underpowered."

### 7.4 Testing — golden-output tests defined

Each new tool needs a checked-in test fixture and expected output. The fixture:

- **100 synthetic rows** of audio features + labels with known properties:
  - 3 classes: `urban` (60 rows), `rural` (30 rows), `industrial` (10 rows) — deliberate imbalance
  - Known correlations: RMS energy and mean_db set to r≈0.85; spectral centroid and ZCR set to r≈0.10
  - 5 rows with extreme outlier values (10σ) on RMS energy
  - 5 rows with null spectral_bandwidth
- **Checked-in expected output JSON** per tool, generated once from a reference run against the fixture and verified by hand (not auto-generated from code)

Tests verify:
1. Computed values match expected within floating-point tolerance
2. Insufficient-data contract fires correctly for edge cases (<10 rows, single class)
3. Seed determinism: two calls with same seed produce identical output
4. Stratified sampling preserves class proportions (±2 percentage points)
5. Output schema matches what chart builder and widget composer expect (all keys present)
6. Diagnostics are present in output when applicable (sample size, caveats, normality flags)

The fixture lives at `data_insights/tests/ml_fixture.csv` and expected outputs at `data_insights/tests/expected/`. Tests run as part of the Django test suite via `python manage.py test data_insights.tests.test_ml_tools`.

### 7.5 Schema discoverability — tool-based

The schema will evolve (new audio features will be added). Enumerating columns in the prompt is brittle — a schema migration silently desyncs the prompt and the LLM hallucinates old column names. Use a lightweight `list_ml_schema` tool:

```python
class ListMLSchemaTool(BaseTool):
    name: str = "list_ml_schema"
    description: str = "Returns available label columns, feature columns, and metadata columns for ML analysis."
```

Returns:
```json
{
  "label_columns": ["category", "class", "subclass", "region"],
  "feature_columns": ["rms_energy", "zero_crossing_rate", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_flatness", "duration", "harmonic_ratio", "percussive_ratio", "mean_db", "max_db", "min_db", "dominant_frequency"],
  "metadata_columns": ["name", "collector", "recording_date", "recording_device", "microphone_type", "time_of_day", "community"],
  "row_counts": {"noise_dataset": 142000, "audio_feature": 98000, "noise_analysis": 95000}
}
```

The LLM calls this once at session start (or when it needs to verify a column name). It adds one turn but eliminates an entire class of hallucination bugs. The prompt tells the LLM to call this tool before any ML tool that references columns by name, and to verify column names against the returned catalog.

### 7.6 Data leakage warning — session-state tracking

The warning mechanism: each tool that computes a full-dataset aggregate (`ml_feature_stats`, `ml_correlation_matrix`) records a flag in the agent state when it runs. `ml_train_test_split` checks that flag and, if set, appends a warning to its output:

```python
# In ml_train_test_split._run():
if state.get("_full_dataset_aggregates_computed"):
    output["warnings"] = output.get("warnings", []) + [
        "ml_feature_stats and/or ml_correlation_matrix were computed on the full "
        "dataset before this split. Any statistics derived from those tools include "
        "data that will become your test set. For unbiased evaluation, re-run those "
        "tools on the training set only after splitting."
    ]
```

The decompose function renders this warning as a yellow banner at the top of the dashboard, not buried in JSON. The flag is stored in `AgentState` (the TypedDict already supports arbitrary keys) and persists via the existing PostgresSaver checkpointer, so it survives across turns in the same session.

---

## 8. What NOT to do

- **Do not fit supervised models** (no classifier/regressor training, no inference, no AutoML).
- **Do not add GPU-dependent computation** — everything runs on CPU.
- **Do not persist ML artifacts** without a purpose-built model registry (separate system).
- **Do not auto-select features or hyperparameters** — surface information, let the engineer decide.
- **Do not add clarification questions that don't change tool behavior** — every signal must be load-bearing.

---

## 9. Success criteria (with output format)

After Phase 2, an ML engineer can ask and get:

| Question | Output |
|---|---|
| "Is this dataset ready for training?" | Stat card: total count, feature coverage %, class count. Progress bars: field completeness. Severity badge: imbalance level. Follow-ups: "Show me class balance", "Profile features" |
| "How balanced are the classes in category?" | Stat card: entropy, minority ratio, severity. Grouped bar: per-class counts. Table: per-class %, recommended stratified split counts. Warning if any class <30 in val+test |
| "Give me a stratified 70/15/15 split by region" | Stat card: train/val/test row counts. Stacked bar: per-class counts in each split. Table: per-class per-split breakdown. Seed: 42. Warning if any class has <2 samples |
| "Is region A significantly different from region B in spectral centroid?" | Stat card: Cohen's d with interpretation. Group comparison bar. Table: Welch's t (t, p, df), Mann-Whitney (U, p), group means/stds/ns |

After Phase 3:

| "What features are correlated?" | Stat card: strongest pair ("Spearman ρ = 0.85, Pearson r = 0.82", sample size). Heatmap: N×N matrix. Table: top-10 pairs with both coefficients and p-values. |
| "Are RMS energy and spectral centroid correlated?" | Same as above, filtered to the pair. |
| "Is region A significantly different from region B in spectral centroid?" | Stat card: Cohen's d with interpretation, sample size badge. Group comparison bar. Table: Welch's t (t, p, df), Mann-Whitney (U, p), group means/stds/ns. |
| "Export features and labels for category classification" | Stat card: row count, column count, file size estimate. Download button. |

After Phase 4:

| "Which features best separate the categories?" | Horizontal bar: MI score per feature (with F-value secondary). Table: scores + normality flags. Visible caveat: "k-NN MI estimator (n_neighbors=3). Scores may be biased upward for classes with <500 samples." |
|---|---|

---

## 10. Phasing rationale

1. **Phase 0 (bugs)**: "Complaint rate" is a live bug that returns bad data. `ml_feature_stats` without decomposition is a broken UX. These ship **before** the prompt rewrite (Phase 1), because the rewritten prompt will reference behavior that doesn't exist until the bugs are fixed.

2. **Phase 1 (foundation)**: unification removes the two-tool-list maintenance burden and makes ML mode strictly more capable. The prompt rewrite gives the LLM guidance it currently lacks. `list_ml_schema` eliminates hallucinated column names — a correctness prerequisite for every tool in Phases 2-5.

3. **Phase 2 (class balance + splits + label clarification)**: highest user value, simplest math. Delivers "is this dataset ready?" and "how should I split?" end-to-end. `ml_missing_label` clarification ships here because it's consumed by Phase 2 tools — shipping it later would mean those tools have no disambiguation when the user doesn't specify which of the four label columns to analyze.

4. **Phase 3 (correlation + tests + export)**: higher statistical complexity and the bridge to notebooks. `ml_export_features` was pulled forward from Phase 6 because it has higher leverage (lets engineers move from chatbot to notebook in one click) and lower implementation complexity than PCA/clustering.

5. **Phase 4 (feature importance)**: builds on Phase 3 infrastructure (correlation computation, sampling patterns, MI estimation). Edge cases (small samples, many classes, non-normal distributions) benefit from lessons learned in earlier phases.

6. **Phase 5 (remaining clarification)**: `ml_split_strategy` and `ml_ambiguous_features`. Both are UX polish — the tools they serve already work with sensible defaults (stratified split, all features). Deferred so they can be informed by actual usage patterns rather than designed speculatively.

7. **Phase 6 (PCA/clustering/outliers)**: unscheduled. Build when users ask for them. The design is here so it's clear where they'd fit, but the commitment stops there.
