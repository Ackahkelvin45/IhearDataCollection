# Phase 5 — UX & Insight Quality (+ F18 XSS)

> The workflow's implementation, critique (approved), and refactor stages completed;
> its test/validate/document stages crashed on transient API errors, so this phase
> was **validated manually by the orchestrator** (results below).

## Issues fixed

| ID | Issue | Fix |
|----|-------|-----|
| **F18** | Streamed assistant/DB markdown rendered via `innerHTML` without escaping (stored-XSS: a dataset name with `<script>`/`<img onerror>` executes in another user's browser) | New `escapeMarkdownSource()` HTML-escapes `& < >` of the raw text **before** any markdown→HTML replacement (`unified_chat.html`); table helpers no longer double-escape; blockquote regex updated for escaped `&gt;`. Injected markup now renders inert; real markdown still formats. |
| P5-2 | Fractional 0–1 metrics (RMS energy, entropy, correlation, MI, scores) misclassified as "ratio" → pie/donut whose slices don't sum to a whole | Classifier only picks pie/donut for true part-of-whole data (composition keyword + non-negative values summing to ~1.0 or ~100); explicitly blocks 0–1 magnitude metrics (`chart_builder.py`) |
| P5-3 | Units (dB/Hz/s) stripped from charts & stat cards | `unit_for_column`/`label_with_unit` re-attach known units to axis labels, titles, descriptions, stat-card labels/values; unknown columns get none |
| P5-4 | Top-N truncation invisible — users couldn't tell data was cut | Both `ChartDecision` top-N and the hard 12/20-row caps now surface a visible "Showing N of M" caption + `truncated`/`total_count`/`shown_count` flags, rendered in template |
| P5-5 | Box-plot widgets silently degraded to a mislabeled bar chart | Emit a real five-number summary (`box_plot_data`); when raw distributions are unavailable, fall back to an **honestly labeled** "Mean Decibel" bar chart |
| P5-6 | `_to_float` coerced null/non-numeric to `0.0`, fabricating zero-height bars | `_coerce_numeric` returns `None` for null/non-numeric/boolean; those rows are dropped (label+value stay aligned); genuine zeros preserved; drop count surfaced in caption |

## Files modified
- `data_insights/templates/data_insights/unified_chat.html` (F18 XSS escape, truncation/box-plot/unit rendering)
- `data_insights/workflows/chart_builder.py` (classification, units, numeric coercion, truncation)
- `data_insights/workflows/widget_composer.py` (box stats, truncation metadata, captions)
- `data_insights/tests/test_phase5_insight_quality.py` (35 new DB-free tests)

## Manual validation (orchestrator)
- **Full `data_insights` suite: 199 tests pass** (35 new + 164 prior), 1 pre-existing skip — no regressions.
- `manage.py check`: clean (only the pre-existing unrelated `urls.W005`).
- `black --check`: all changed files clean.
- **F18 verified**: `escapeMarkdownSource` is invoked before `formatTables` and all HTML-producing replacements; output still assigned to `innerHTML` but content is escaped-then-formatted. Confirmed the markdown renderer has **no** link/`href`/`src` rule that places user text into an HTML attribute, so `&<>` escaping fully covers the injection surface.

## Risks / residual
- The XSS escape only neutralizes `& < >` (sufficient here because user content is only ever inserted as element text, never into an attribute). If a future markdown rule adds user-controlled attributes (e.g. links), it must additionally quote-escape.
- `box_plot_data` requires the backend to provide a five-number summary; where unavailable, the honest bar-chart fallback is used.
