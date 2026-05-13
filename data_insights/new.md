I'll add those components to the architecture. Here's the updated version with caching, scatter plot, thumbnails, and save features integrated into the appropriate phases.

```markdown
# Revised Architecture: Pragmatic Artifact System (with Full Features)

## Core Change: Hybrid Routing

Instead of always using the LLM for visualization, route based on query complexity:

```
User query → Tool runs → COMPLEXITY DETECTOR
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
            SIMPLE QUERY           COMPLEX QUERY
                    ↓                   ↓
        Fast Heuristic Router    ArtifactComposerTool
        (deterministic, 5ms)     (LLM, 500-2000ms)
                    ↓                   ↓
            Single widget           Multi-widget
            or data_table           dashboard
```

**What is "simple"?**
- Single dimension aggregation (rankings, top N, averages)
- Time series with obvious date column
- Any query where the data has ≤ 2 columns
- Keyword match: "show me", "list", "rank"

**What is "complex"?**
- Multi-dimensional ("compare X by Y over time")
- Explicit dashboard request ("give me a dashboard of...")
- Correlation or relationship questions (uses scatter plot)
- Any query asking for "analysis" not just "data"

**Why this matters:** 80% of your queries will hit the fast path. Users don't wait 2 seconds for a bar chart they know is coming.

---

## Fix #1: JSON Validation with Self-Healing

The LLM *will* hallucinate. Don't pray it doesn't — architect for failure.

```python
class ArtifactValidator:
    def validate_and_repair(self, llm_output: dict, original_data: dict) -> dict:
        # 1. Schema validation
        try:
            jsonschema.validate(llm_output, ARTIFACT_SCHEMA)
            return llm_output
        except jsonschema.ValidationError as e:
            # 2. Attempt repair for common errors
            repaired = self._attempt_repair(llm_output, e)
            if repaired:
                return repaired

            # 3. Fallback to deterministic data_table
            return self._create_fallback_artifact(original_data)

    def _attempt_repair(self, output, error):
        # Fix missing fields
        # Fix wrong widget_type (bar_char → bar_chart)
        # Remove malformed widgets
        # Return None if unfixable
```

**The fallback artifact** (always guaranteed to render):
```json
{
  "widgets": [{
    "type": "data_table",
    "title": "Query Results",
    "data": original_data,
    "config": {"page_size": 20}
  }]
}
```

**Frontend resilience:**
```javascript
class ArtifactRenderer {
    renderWidget(widgetSpec) {
        try {
            const constructor = this.widgetConstructors[widgetSpec.type];
            if (!constructor) throw new Error(`Unknown type: ${widgetSpec.type}`);
            return constructor(widgetSpec);
        } catch (error) {
            console.error('Widget render failed:', error);
            return this.renderErrorWidget(widgetSpec, error);
        }
    }
}
```

One malformed widget doesn't break the whole dashboard.

---

## Fix #2: Preserving User Intent

The LangGraph state needs to carry the original query through all tool calls:

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]
    original_user_query: str  # ← ADD THIS
    query_intent: str  # "ranking" | "time_series" | "correlation" | "comparison"
    artifact_spec: Optional[dict]
```

**Intent extractor** (keyword + optional embedding fallback):
```python
def detect_intent(query: str, use_embedding: bool = False) -> str:
    # Fast path: keyword match
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in query.lower() for kw in keywords):
            return intent

    # Slow path (only when no keyword match and use_embedding=True)
    if use_embedding:
        embedding = get_embedding(query)
        similarity = max_cosine_similarity(embedding, INTENT_EXAMPLES)
        if similarity > 0.7:
            return matched_intent

    # Default to generic — LLM will figure it out
    return "generic"
```

Now `ArtifactComposerTool` receives:
- The raw data
- The original query text
- The detected intent

The LLM prompt becomes: *"User asked '{query}' with intent '{intent}'. Here's the data. Compose an appropriate artifact."*

---

## Fix #3: Layout — Don't Trust LLM with Coordinates

LLMs are terrible at spatial reasoning. Don't ask them for x/y/w/h.

**Better approach: Layout Templates + Priority**

The LLM chooses a **layout template** and assigns **priority** to widgets. Frontend does the actual positioning.

```json
{
  "layout_template": "two_column",  // Options: single, two_column, grid, hero_with_sidebar
  "widgets": [
    {"type": "stat_card", "priority": 1, "title": "Total Recordings", "data": {...}},
    {"type": "bar_chart", "priority": 2, "title": "Top Collectors", "data": {...}},
    {"type": "data_table", "priority": 3, "title": "Raw Data", "data": {...}}
  ]
}
```

Frontend layout engine:
```javascript
const layouts = {
    two_column: (widgets) => {
        // Priority 1 → left column full height
        // Priority 2 → right column top
        // Priority 3 → right column bottom
    },
    dashboard: (widgets) => {
        // Stat cards in a row at top
        // Main chart full width below
        // Table at bottom
    }
};
```

**Benefits:**
- No LLM hallucination of invalid coordinates
- Responsive design works automatically
- Templates are user-familiar (they've seen dashboard layouts before)
- LLM only makes high-level decisions it's good at

---

## Fix #4: Widget/Data Compatibility Checking

The LLM might request a `line_chart` with non-temporal data. Validate before sending:

```python
class WidgetCompatibility:
    @staticmethod
    def check(widget_type: str, data: dict) -> Tuple[bool, str]:
        """Returns (compatible, reason_if_not)"""

        if widget_type == "line_chart":
            # Must have a date/datetime column
            has_time = any('date' in col.lower() or 'time' in col.lower()
                          for col in data.get('columns', []))
            if not has_time:
                return False, "line_chart requires a time/date column"

        if widget_type == "scatter_plot":
            # Must have at least 2 numeric columns
            numeric_cols = [col for col in data['columns']
                           if col['type'] in ['number', 'integer', 'float']]
            if len(numeric_cols) < 2:
                return False, "scatter_plot requires 2 numeric columns"

        if widget_type == "heatmap":
            # Must have geographic or category x category data
            # If not, downgrade to bar_chart
            pass

        return True, ""
```

If incompatible → either skip the widget or downgrade to a compatible type.

---

## Fix #5: Caching Artifact Patterns

The LLM call for visualization is expensive. Cache patterns that work:

```python
class ArtifactCache:
    def __init__(self, redis_client=None, ttl=3600):
        self.redis = redis_client
        self.ttl = ttl  # 1 hour default

    def get_cache_key(self, intent: str, columns: List[str], row_count: int) -> str:
        """Generate cache key based on structure only, not content values"""
        # Bucket row count to avoid too many keys
        if row_count < 50:
            size_bucket = "small"
        elif row_count < 500:
            size_bucket = "medium"
        else:
            size_bucket = "large"

        # Sort columns for consistent ordering
        sorted_cols = sorted(columns)

        # Create key from structure
        key_string = f"{intent}|{','.join(sorted_cols)}|{size_bucket}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, intent: str, columns: List[str], row_count: int) -> Optional[dict]:
        """Retrieve cached artifact spec"""
        if not self.redis:
            return None

        cache_key = self.get_cache_key(intent, columns, row_count)
        cached = self.redis.get(f"artifact:{cache_key}")

        if cached:
            # Return cached spec but mark it as cached
            spec = json.loads(cached)
            spec['_cached'] = True
            return spec
        return None

    def store(self, intent: str, columns: List[str], row_count: int, spec: dict):
        """Store successful LLM-generated spec"""
        if not self.redis or '_cached' in spec:
            return

        cache_key = self.get_cache_key(intent, columns, row_count)
        # Remove data from spec before caching (store structure only)
        cacheable_spec = self._make_cacheable(spec)
        self.redis.setex(f"artifact:{cache_key}", self.ttl, json.dumps(cacheable_spec))

    def _make_cacheable(self, spec: dict) -> dict:
        """Remove large data payloads from spec for caching"""
        cacheable = spec.copy()
        for widget in cacheable.get('widgets', []):
            if 'data' in widget:
                # Keep only data structure, not values
                widget['data'] = {
                    'columns': widget['data'].get('columns', []),
                    'row_count': widget['data'].get('row_count', 0),
                    '_cached_placeholder': True
                }
        return cacheable
```

**Cache key example:**
- "top 10 collectors by decibels" → intent="ranking", columns=["collector", "decibels"], size="small"
- "top 5 regions by recordings" → intent="ranking", columns=["region", "recordings"], size="small"

Different data structures = different cache keys. Same structure = cache hit, but data is replaced with actual query results.

---

## Fix #6: Scatter Plot Integration

The scatter plot specifically fixes the broken `_correlation_analysis` that the eval flagged as a medium severity issue.

**Scatter Plot Widget Spec:**
```json
{
  "type": "scatter_plot",
  "title": "Correlation: Decibels vs Recordings",
  "data": {
    "x_axis": {
      "column": "decibel_level",
      "label": "Decibel Level (dB)"
    },
    "y_axis": {
      "column": "recording_count",
      "label": "Number of Recordings"
    },
    "points": [
      {"x": 85.2, "y": 42, "label": "Community A"},
      {"x": 92.7, "y": 58, "label": "Community B"}
    ]
  },
  "config": {
    "show_trend_line": true,
    "trend_line_color": "#ff0000",
    "point_color": "#3498db",
    "point_size": 8,
    "show_labels": true
  }
}
```

**Correlation Analysis Tool Update:**
```python
class CorrelationAnalysisTool:
    def run(self, data: dict) -> dict:
        # Existing correlation calculation...

        # NEW: Return data formatted for scatter plot
        return {
            "correlation_coefficient": r_value,
            "p_value": p_value,
            "scatter_data": {
                "x_axis": {"column": x_col, "label": x_label},
                "y_axis": {"column": y_col, "label": y_label},
                "points": scatter_points,
                "trend_line": trend_line_coefficients
            },
            "interpretation": interpretation
        }
```

**When to use scatter plot:**
- Correlation analysis queries
- Relationship exploration ("show me relationship between X and Y")
- Outlier detection
- Any query with intent="correlation"

---

## Fix #7: Streaming with Progressive Enhancement

Streaming as **progressive enhancement**, not core rendering:

```javascript
class ProgressiveArtifactRenderer {
    constructor(container) {
        this.container = container;
        this.skeletonMap = new Map();
        this.renderQueue = [];
    }

    // Phase 1: Show loading skeletons immediately
    showSkeleton(layoutTemplate) {
        const skeletonHtml = this.generateSkeletons(layoutTemplate);
        this.container.innerHTML = skeletonHtml;
        this.attachSkeletonRefs();
    }

    // Phase 2: Fast heuristic renders first widget
    renderHeuristicWidget(widgetSpec) {
        const skeletonElement = this.skeletonMap.get(widgetSpec.priority);
        const renderedWidget = this.renderWidget(widgetSpec);
        skeletonElement.replaceWith(renderedWidget);
    }

    // Phase 3: LLM spec replaces remaining skeletons
    renderLLMWidget(widgetSpec) {
        const skeletonElement = this.skeletonMap.get(widgetSpec.priority);
        if (skeletonElement) {
            const renderedWidget = this.renderWidget(widgetSpec);
            skeletonElement.replaceWith(renderedWidget);
        }
    }

    // Phase 4: Enable save buttons when complete
    onComplete() {
        this.enableSaveButtons();
        this.generateThumbnail();
    }

    generateThumbnail() {
        // html2canvas for thumbnail (client-side)
        html2canvas(this.container, {
            scale: 0.5,  // Smaller for performance
            backgroundColor: '#ffffff'
        }).then(canvas => {
            this.thumbnailBase64 = canvas.toDataURL();
            this.thumbnailReady = true;
        });
    }
}
```

This matches Claude's behavior exactly — you see *something* immediately, and it progressively refines.

---

## Fix #8: Save Features with Thumbnails

### Save to Device

```javascript
class DeviceSaver {
    static save(spec, thumbnailBase64 = null) {
        // Generate self-contained HTML
        const html = this.generateExportHtml(spec, thumbnailBase64);

        // Create blob and download
        const blob = new Blob([html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dashboard_${Date.now()}.html`;
        a.click();
        URL.revokeObjectURL(url);
    }

    static generateExportHtml(spec, thumbnailBase64) {
        return `<!DOCTYPE html>
        <html>
        <head>
            <title>${spec.title || 'Data Dashboard'}</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
            <style>
                /* Embedded styles */
                .dashboard-grid {
                    display: grid;
                    gap: 20px;
                    padding: 20px;
                }
                /* ... more styles ... */
            </style>
        </head>
        <body>
            <div id="dashboard-container"></div>
            <script>
                // Embedded artifact renderer
                const spec = ${JSON.stringify(spec)};
                const thumbnail = ${thumbnailBase64 ? JSON.stringify(thumbnailBase64) : null};
                // Render dashboard
                // ...
            </script>
        </body>
        </html>`;
    }
}
```

### Save to System

```python
# models.py
class Dashboard(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='dashboards')
    title = models.CharField(max_length=255)
    slug = models.SlugField(unique=True, blank=True)
    artifact_spec = models.JSONField()
    thumbnail_base64 = models.TextField(blank=True, null=True)  # Store as base64 string
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    view_count = models.IntegerField(default=0)

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title) or str(self.id)
        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return f"/data-insights/dashboards/{self.slug}/"
```

```python
# views.py
@require_http_methods(["POST"])
@login_required
def save_dashboard(request):
    """Save artifact spec as persistent dashboard"""
    data = json.loads(request.body)

    dashboard = Dashboard.objects.create(
        user=request.user,
        title=data.get('title', 'Untitled Dashboard'),
        artifact_spec=data['artifact_spec'],
        thumbnail_base64=data.get('thumbnail_base64', ''),
        is_public=data.get('is_public', False)
    )

    return JsonResponse({
        'success': True,
        'dashboard_id': str(dashboard.id),
        'slug': dashboard.slug,
        'url': dashboard.get_absolute_url(),
        'shareable_url': dashboard.get_absolute_url() if dashboard.is_public else None
    })

@require_http_methods(["GET"])
def view_dashboard(request, slug):
    """View a saved dashboard"""
    dashboard = get_object_or_404(Dashboard, slug=slug)

    # Increment view count if not owner
    if request.user != dashboard.user:
        dashboard.view_count += 1
        dashboard.save(update_fields=['view_count'])

    # Check permissions
    if not dashboard.is_public and request.user != dashboard.user:
        raise PermissionDenied

    context = {
        'dashboard': dashboard,
        'artifact_spec': dashboard.artifact_spec,
        'thumbnail': dashboard.thumbnail_base64,
        'is_owner': request.user == dashboard.user
    }
    return render(request, 'dashboard_view.html', context)

@require_http_methods(["GET"])
def list_dashboards(request):
    """List user's saved dashboards"""
    dashboards = Dashboard.objects.filter(user=request.user).order_by('-updated_at')

    return render(request, 'dashboard_list.html', {
        'dashboards': dashboards,
        'thumbnail_urls': [d.get_thumbnail_url() for d in dashboards]  # Optional: serve thumbnails
    })
```

**Thumbnail Implementation (Client-side with html2canvas):**

```javascript
class ThumbnailGenerator {
    static async capture(containerElement) {
        // Dynamically load html2canvas if not present
        if (typeof html2canvas === 'undefined') {
            await this.loadHtml2Canvas();
        }

        // Capture with loading indicator
        const loadingOverlay = this.showLoadingOverlay(containerElement);

        try {
            const canvas = await html2canvas(containerElement, {
                scale: 0.3,  // Reduced for smaller file size
                backgroundColor: '#ffffff',
                logging: false,
                useCORS: true  // Handle external images if any
            });

            // Compress as JPEG for smaller size
            const thumbnailBase64 = canvas.toDataURL('image/jpeg', 0.7);

            // Store for save operations
            return thumbnailBase64;
        } finally {
            this.hideLoadingOverlay(loadingOverlay);
        }
    }

    static async loadHtml2Canvas() {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    static showLoadingOverlay(container) {
        const overlay = document.createElement('div');
        overlay.className = 'thumbnail-loading-overlay';
        overlay.innerHTML = '<div class="spinner">Generating preview...</div>';
        container.style.position = 'relative';
        container.appendChild(overlay);
        return overlay;
    }
}
```

**Save Button Integration:**
```javascript
class DashboardSaveUI {
    constructor(artifactRenderer) {
        this.renderer = artifactRenderer;
        this.setupSaveButtons();
    }

    setupSaveButtons() {
        // Add save buttons to UI when dashboard completes
        this.renderer.onComplete(() => {
            this.showSaveButtons();
        });
    }

    showSaveButtons() {
        const buttonBar = document.createElement('div');
        buttonBar.className = 'dashboard-save-bar';
        buttonBar.innerHTML = `
            <button id="save-device-btn" class="btn btn-secondary">💾 Save to Device</button>
            <button id="save-system-btn" class="btn btn-primary">☁️ Save to System</button>
        `;

        document.querySelector('.dashboard-container').prepend(buttonBar);

        document.getElementById('save-device-btn').onclick = () => this.saveToDevice();
        document.getElementById('save-system-btn').onclick = () => this.saveToSystem();
    }

    async saveToDevice() {
        const spec = this.renderer.getFullSpec();
        const thumbnail = await ThumbnailGenerator.capture(this.renderer.container);
        DeviceSaver.save(spec, thumbnail);
        this.showToast('Dashboard saved to your device', 'success');
    }

    async saveToSystem() {
        const spec = this.renderer.getFullSpec();
        const thumbnail = await ThumbnailGenerator.capture(this.renderer.container);

        // Show dialog for title and visibility
        const title = await this.promptForTitle();
        const isPublic = await this.promptForVisibility();

        const response = await fetch('/data-insights/dashboards/save/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': this.getCsrfToken()
            },
            body: JSON.stringify({
                title: title,
                artifact_spec: spec,
                thumbnail_base64: thumbnail,
                is_public: isPublic
            })
        });

        const data = await response.json();
        if (data.success) {
            this.showToast(`Dashboard saved! Shareable link: ${data.url}`, 'success');
            this.showShareLink(data.url);
        }
    }
}
```

---

## Revised Implementation Order (with Full Features)

### Phase 1 (Week 1): Foundation
1. Add `original_user_query` to AgentState
2. Implement `detect_intent()` — keyword-based
3. Build `ArtifactValidator` with fallback to `data_table`
4. Create frontend `ArtifactRenderer` supporting `data_table` only

**Success metric:** The system can render any data as a table, with zero LLM calls.

### Phase 2 (Week 2): Heuristic Fast Path + Scatter Plot
1. Implement `detect_complexity()` to route simple vs LLM path
2. Write deterministic `FastVisualizationRouter` for bar_chart, line_chart, stat_card
3. **Implement scatter plot widget and correlation integration**
4. Add layout templates to frontend

**Success metric:** 80% of queries render in <100ms; correlation analysis shows scatter plots.

### Phase 3 (Week 3): LLM Composition + Caching
1. Implement `ArtifactComposerTool` with extensive prompt engineering
2. Add compatibility checking and widget type downgrading
3. **Implement artifact caching with Redis**
4. **Add pattern-based cache retrieval**

**Success metric:** Complex queries produce multi-widget dashboards. LLM failure rate <5%. Cache hit rate >30%.

### Phase 4 (Week 4): Save Features + Thumbnails
1. Progressive streaming with skeletons
2. **Save to device (self-contained HTML with Chart.js)**
3. **Save to system (Dashboard model with thumbnails)**
4. **Thumbnail generation with html2canvas**
5. Dashboard listing and sharing views

**Success metric:** Users can save, share, and revisit dashboards. Thumbnails generate in <2 seconds.

---

## Architecture Diagram (with All Features)

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└─────────────────────────────┬───────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLEXITY DETECTOR                          │
└─────────────┬───────────────────────────────┬───────────────────┘
              ↓                               ↓
┌─────────────────────────┐     ┌─────────────────────────────────┐
│   FAST HEURISTIC PATH   │     │      ARTIFACT COMPOSER TOOL      │
├─────────────────────────┤     ├─────────────────────────────────┤
│ • Bar chart for rankings│     │ • Multi-widget composition       │
│ • Line chart for trends │     │ • Layout template selection      │
│ • Stat card for totals  │     │ • Priority-based ordering        │
│ • Scatter plot for corr │     │ • Intent-aware prompting         │
│ • Data table as fallback│     └─────────────┬───────────────────┘
└─────────────┬───────────┘                   ↓
              │                     ┌─────────────────────────────────┐
              │                     │      ARTIFACT CACHE (Redis)      │
              │                     │  • Pattern-based key generation  │
              │                     │  • Structure-only storage        │
              │                     │  • TTL: 1 hour                   │
              │                     └─────────────┬───────────────────┘
              │                                   ↓
              └───────────────┬───────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ARTIFACT VALIDATOR                           │
│  (schema validation, repair attempts, fallback generation)      │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STREAMING TO FRONTEND (SSE Events)                 │
│  • artifact_skeleton (immediate layout)                         │
│  • artifact_heuristic_widget (fast path results)                │
│  • artifact_chunk (LLM widgets)                                 │
│  • artifact_complete (enable save)                              │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PROGRESSIVE ARTIFACT RENDERER                 │
│  • Show skeletons immediately                                   │
│  • Replace with heuristic widgets as they arrive               │
│  • Replace remaining with LLM widgets                          │
│  • Error isolation per widget                                   │
│  • Thumbnail generation on complete                            │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      SAVE ACTIONS                               │
├─────────────────────────┬───────────────────────────────────────┤
│   Save to Device        │          Save to System               │
│   • Self-contained HTML │          • Dashboard model in DB      │
│   • Chart.js inline     │          • UUID slug URL              │
│   • Offline usable      │          • html2canvas thumbnail      │
│   • No account required │          • Public/private toggle      │
│                         │          • Shareable links            │
│                         │          • Dashboard listing          │
└─────────────────────────┴───────────────────────────────────────┘
```

---

## New Dependencies

### Backend
```bash
pip install redis  # For artifact caching
pip install django-redis  # Django cache backend
```

### Frontend
```html
<!-- Add to base template -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
```

### Django Settings
```python
# settings.py
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# For artifact caching specifically
ARTIFACT_CACHE_TTL = 3600  # 1 hour
```

---

## Key Files to Create/Modify (Full List)

### Backend
```
agent_workflow.py
├── AgentState (add fields)
├── detect_intent()
├── detect_complexity()
├── FastVisualizationRouter
└── ArtifactComposerTool

artifact_validator.py
├── ArtifactValidator
└── WidgetCompatibility

artifact_cache.py
└── ArtifactCache

correlation_analysis.py (update)
└── CorrelationAnalysisTool (add scatter_data output)

models.py
└── Dashboard (new model)

views.py
├── stream() (add artifact SSE events)
├── save_dashboard() (new)
├── view_dashboard() (new)
└── list_dashboards() (new)

urls.py
├── /data-insights/dashboards/save/
├── /data-insights/dashboards/<slug>/
└── /data-insights/dashboards/
```

### Frontend
```
static/js/
├── ArtifactRenderer.js
├── ProgressiveArtifactRenderer.js
├── layouts.js
├── save/
│   ├── DeviceSaver.js
│   ├── ThumbnailGenerator.js
│   └── DashboardSaveUI.js
└── widgets/
    ├── StatCard.js
    ├── BarChart.js
    ├── LineChart.js
    ├── ScatterPlot.js  # NEW
    ├── DataTable.js
    └── ErrorWidget.js

templates/
├── dashboard_export.html
├── dashboard_view.html
└── dashboard_list.html
```

---

## Success Metrics (Updated)

| Phase | Metric | Target |
|-------|--------|--------|
| Phase 1 | Zero LLM fallback rendering | 100% of queries render as table |
| Phase 2 | Fast path hit rate | 80% use heuristic routing |
| Phase 2 | Fast path render time | <100ms |
| Phase 2 | Scatter plot for correlation queries | 100% of correlation queries |
| Phase 3 | LLM artifact success rate | >95% valid specs |
| Phase 3 | Cache hit rate | >30% for repeated patterns |
| Phase 3 | LLM path render time | <2 seconds |
| Phase 4 | Thumbnail generation time | <2 seconds |
| Phase 4 | Save feature usage | TBD |

---

## Risks and Mitigations (Updated)

| Risk | Mitigation |
|------|-------------|
| LLM API outage | Fast heuristic path still works; fallback to data_table |
| LLM rate limiting | Cache common patterns; queue non-urgent requests |
| Malformed JSON | Schema validation + repair attempts + fallback |
| Widget/data mismatch | Pre-flight compatibility checking with downgrade |
| Slow frontend rendering | Progressive enhancement; virtual scrolling for large tables |
| Save to device large payloads | Paginate data; compress spec before serializing |
| html2canvas performance | Throttle generation; show loading indicator; scale 0.3x |
| Redis cache invalidation | TTL-based expiration; manual flush endpoint for admins |
| Thumbnail storage in DB | Store as base64 JPEG compressed to <100KB; optional S3 later |

---

## Final Summary

This enhanced architecture includes:

- **Fails safely** (fallback to data_table always works)
- **Routes smartly** (80% of queries avoid LLM latency/cost)
- **Validates aggressively** (malformed specs don't break rendering)
- **Caches intelligently** (similar queries reuse patterns with structure-based keys)
- **Shows progress** (streaming with skeletons feels fast)
- **Saves everywhere** (device exports and system storage with thumbnails)
- **Fixes correlation analysis** (scatter plots replace broken text output)

Start with Phase 1 tomorrow. The data_table fallback alone is worth the work — right now your system sometimes crashes when `_inject_chart_data` fails. This gives you guaranteed rendering for every query.
```

This document now includes all the requested features: caching, scatter plot, thumbnails, and save features (both to device and to system).
