# Technical Evaluation: Natural Language Query Engineering in `data_insights`

## Overall Architecture

Your NL→data pipeline is a **dual-path system**: a structured-tool path (`DataAnalysisTool` with hardcoded query types) and a free-form SQL generation path (`TextToSQLAgent`). The structured path acts as a router that classifies user intent into predefined query types, while the fallback SQL path handles everything else. This is a solid pattern — it's essentially a lightweight semantic router.

---

## Prompt Engineering — [prompt.py](workflows/prompt.py)

### What Works

- **Role grounding is strong.** Each prompt template establishes a clear persona (audio data analyst, ML assistant) with explicit domain knowledge. The `SYSTEM_TEMPLATE` is well-structured with clear capability boundaries.
- **The "never reveal SQL" constraint** in `SQL_SYSTEM_TEMPLATE` (line 28) is a good security practice — prevents leaking schema internals.
- **The `ML_SYSTEM_TEMPLATE`** is refreshingly concise compared to the others. It's focused and task-specific.

### What's Broken or Weak

**1. The `SQL_SYSTEM_TEMPLATE` is contradictory and confusing for an LLM.**

Look at lines 16-17:
```
1. Generate and run **only safe, read-only SQL queries** (`SELECT` statements).
1. **Always call the PostgresSQLInput tool to execute your SQL query before answering.**
```
Both are numbered "1." — this is a formatting error that confuses the model. But worse, the tool name `PostgresSQLInput` doesn't exist as a user-facing tool. The actual tool is bound as a schema via `self.llm.bind_tools([PostgresSQLInput])` in `sql_agent.py:470`. The prompt references a name the model can't map to. This disconnect between prompt and actual tool binding creates confusion.

**2. `{table_info}` is a black-box injection with zero guidance.**

The `SQL_SYSTEM_TEMPLATE` says:
```
Here is the schema information you can use to write queries:
  ```
  {table_info}
  ```
```
But `table_info` is injected as raw `CREATE TABLE` DDL strings (from `SQLDatabaseWrapper.get_table_info()`). There's no semantic description of what columns mean, no relationship explanation, no example queries. The model has to reverse-engineer table purpose from column names alone. For a domain-specific audio database with fields like `mfccs`, `chroma_stft`, `spectral_rolloff`, this is asking the model to guess.

**3. `SYSTEM_TEMPLATE` is bloated and self-contradictory.**

At ~120 lines, the `SYSTEM_TEMPLATE` is too long. Long system prompts cause attention dilution — the model ignores the middle sections. Specific problems:

- Line 41: "For numeric/DB questions, always call a data tool" — but then it lists 5 different tools. How does the model choose between `data_analysis`, `search_noise_datasets`, `search_audio_features`, etc? The prompt gives no decision tree.
- Lines 46-47: "Call `visualization_analysis` only when..." but then lines 72-77 are a full section about visualization capabilities. Redundant and contradictory.
- Line 134: "Do not mention documents/uploads unless the user explicitly asks" — this is a negative constraint buried at the end where attention is lowest. Negative constraints should be prominent and paired with positive alternatives.

**4. The `analysis_type` enum in `AudioAnalysisInput` forces the LLM to pre-classify.**

The `AudioAnalysisInput` schema (tools.py:652) requires the model to pick from `"energy", "spectral", "frequency", "correlation", "statistical", "temporal", "overview"`. This is a **double-classification problem**: the model must both understand the user's question AND map it to one of these buckets. A question like "show me the relationship between decibel levels and frequency over the past month" is simultaneously correlation AND temporal. The forced single choice loses information.

**5. No few-shot examples in any prompt.**

There is not a single concrete example of `[User Question → Correct Tool Call]` in any prompt. LLMs perform dramatically better with 2-3 examples. The "Example Interactions" section in `SYSTEM_TEMPLATE` (lines 110-118) describes the flow in prose but never shows the actual tool call JSON the model should emit.

---

## Query Intent Classification — [tools.py](workflows/tools.py#L1338-L1375)

### What Works

- The `query_type` Literal enum is comprehensive and covers common query patterns well (recent, top collectors, counts, decibel ranking, etc.).
- Entity extraction via `_match_entity_name` is pragmatic — matching user query text against actual DB values for region, community, category, etc.

### What's Broken or Weak

**1. The classification is entirely delegated to the LLM via enum — no preprocessing.**

The `DataAnalysisInput.query_type` is a `Literal` type. This means the LLM has to read the user's natural language, map it to one of 9 query types, and output the correct enum value. This is fragile. A better approach:

- Use keyword/pattern-based pre-classification (e.g., if "recent" or "latest" appears → `recent_datasets`)
- Use a lightweight embedding similarity against canonical examples of each query type
- Fall back to LLM classification only when the heuristic is uncertain

**2. The `sql` catch-all is a black hole.**

When the model picks `query_type="sql"`, it goes to `_invoke_sql_agent` which invokes a completely separate LangGraph agent. The return value is just `{"message": msg}` — a raw string. This breaks the structured output contract that every other query type maintains. The downstream visualization injection code has no structured data to work with. The user gets a text blob back, and charts never render for SQL-path queries.

**3. Entity extraction has a substring matching footgun.**

`_match_entity_name` (line 1407) does `if name_lower and name_lower in query_lower`. If a region is named "North" and the user asks "northern region recordings", the match succeeds. But if a category is named "Urban" and the user asks "suburban noise", it also matches. No word-boundary check. This will produce false positives on short entity names.

---

## SQL Generation & Safety — [sql_agent.py](workflows/sql_agent.py)

### What Works

- **SQL validation is thorough.** The `_validate_sql_query` method has injection detection, unsafe keyword blocking, table authorization checks, and recursive subquery validation. This is the strongest security component in the entire system.
- **Limit enforcement is robust.** Auto-appending `LIMIT` and capping existing limits prevents accidental full-table scans.
- **The `UNSAFE_KEYWORDS` tuple** in `__init__.py` is comprehensive.

### What's Broken or Weak

**1. SQL extraction regex is fragile and greedy.**

The `extract_sql` method (line 748) tries 6 different regex patterns in sequence. The `r"\bSELECT\b .*?;"` pattern is non-greedy on `;` but greedy on everything else. A multi-statement LLM response with explanatory text between SQL fragments will capture garbage. And the fallback pattern `r"```(.*?)```"` captures ANY code block, not just SQL.

**2. No query result interpretation layer.**

The SQL agent returns raw rows. The `SYSTEM_TEMPLATE` prohibits the model from revealing SQL, but there's no middleware that converts raw column names like `audio_features__rms_energy` into human-readable labels before the LLM sees them. The LLM has to do this mapping in its head, which is error-prone.

**3. `_filter_messages` strips context silently.**

Line 613-626: This method removes AIMessage/ToolMessage pairs where the parser finds tool calls. The intent is to keep the conversation window small, but it removes the *history* of what was tried. If the SQL agent retries a failed query, it loses the context of what failed before. This is actively harmful for the retry loop.

**4. No streaming for SQL queries.**

The `TextToSQLAgent` has no streaming support — it's all-or-nothing `invoke()`. For complex SQL queries that might take seconds, the user sees a spinner with no feedback. Contrast this with the structured tools that stream through the agent workflow.

**5. `ai_answer` parameter is confusing and underused.**

Throughout the code, `ai_answer` is threaded from the view through to the SQL agent, changing behavior in `should_continue` (line 785-787). But the semantic meaning is unclear — when False, the agent stops after one tool call; when True, it can loop. This means by default, the SQL agent gets ONE attempt at a query. If the SQL has a syntax error, the user gets the raw error and the conversation ends. No retry, no correction.

---

## Tool Design — [tools.py](workflows/tools.py)

### What Works

- **Pydantic models for tool inputs** is the right approach — type safety and schema validation.
- **Query caching** via `QueryCacheModel` for large result sets (>100 rows) is good for performance.
- **The `skip_visualization` flag** is a clean mechanism for suppressing charts on count/list results.

### What's Broken or Weak

**1. `AudioAnalysisTool` and `DataAnalysisTool` overlap massively.**

Both tools query the same models (`NoiseDataset`, `AudioFeature`, `NoiseAnalysis`). Both do aggregations by region/category. But they have different input schemas, different output formats, and different analysis type enums. The LLM has to choose between `analyze_audio_data` with `analysis_type="energy"` and `data_analysis` with `query_type="decibel_ranked"`. These should be a single unified tool with clearer routing.

**2. `VisualizationAnalysisTool` uses an LLM-inside-a-tool anti-pattern.**

Line 1856: the visualization tool creates its own `ChatOpenAI` instance and makes a separate LLM call. This means:
- Every viz recommendation costs an extra LLM round-trip (latency + cost)
- The inner LLM call has no access to the conversation context
- The prompt is ~60 lines of chart selection rules that could be a 30-line function
- The `_analyze_query_characteristics` + `_suggest_chart_type` methods (lines 1920-2049) are pure heuristics that already determine the chart type. Then the LLM is called, and then `_validate_recommendation` overrides the LLM output back to what the heuristics said. **The LLM call adds zero value** — the heuristics both precede and override it.

**3. The `_correlation_analysis` method doesn't compute correlations.**

Lines 924-987: Despite the name, this method doesn't calculate any correlation coefficient (Pearson, Spearman). It extracts sample data arrays and returns them as "rms_vs_spectral_centroid" with raw samples. The user asks "what's the correlation?" and gets back two lists of 20 numbers. That's not analysis — it's data dumping.

**4. Tool descriptions are inconsistent with actual capabilities.**

`AudioAnalysisTool.description` says "Comprehensive audio analysis tool for energy, spectral, frequency, and statistical analysis." But the `_frequency_analysis` method only does dominant frequency + ZCR — no FFT, no frequency bands, no spectral power distribution. The description promises more than the tool delivers.

**5. `_get_data_insights_db_uri()` duplicates logic from `sql_agent.py`.**

Both files independently determine database connection parameters with slightly different logic. This is a configuration source-of-truth problem waiting to cause a production incident.

---

## Workflow Graph — [agent_workflow.py](workflows/agent_workflow.py)

### What Works

- The LangGraph state machine is clean and well-structured.
- The node separation (cleanup, agent, tools, post_process, format) is logical.
- Checkpoint-based conversation persistence via `PostgresSaver` is production-grade.

### What's Broken or Weak

**1. The graph always starts with `cleanup` → every message triggers a DB query.**

`graph.set_entry_point("cleanup")` (line 210) means `cleanup_expired_handles` runs on every message, hitting `QueryCacheModel.objects.get()` per handle. For a user with 0 active handles, this is wasted work. For a user with many handles, it's N+1 queries before the agent even starts thinking.

**2. `post_process_tools` has a silent failure mode.**

Lines 136-153: If `json.loads` fails or the result doesn't have `query_id`, the method silently passes. This means query handles can be lost without any logging or error propagation. The comment says "Content might not be JSON, that's okay" — but if content isn't JSON, it means the tool didn't follow the expected contract, which is a bug worth logging.

**3. `AgentState` inherits from `Dict`.**

Line 24: `class AgentState(Dict)` is unusual. LangGraph expects `TypedDict` or `BaseModel` for state. Inheriting from raw `Dict` means no type checking on state updates, no default values, and potential serialization issues. The `messages` field uses `Annotated[List[AnyMessage], add]` which is the LangGraph reducer pattern, but wrapping it in a plain Dict subclass is fragile.

**4. Single agent instance cached globally.**

In `views.py:738-765`, `_create_ai_agent` caches one agent per mode in `_agents: dict = {}`. This means:
- All users share the same agent instance
- The system prompt is static per mode — no user-specific context
- If the agent's internal state is mutated, it affects all users
- The `_current_user_id` ContextVar is a workaround for this architectural choice

---

## Serialization & Streaming — [views.py](views.py)

### What Works

- The `_sanitize_data` recursive sanitizer is thorough and handles edge cases.
- SSE-style streaming with newline-delimited JSON is the right protocol.
- The `_extract_rows` helper consolidates row extraction across all output shapes.

### What's Broken or Weak

**1. `_inject_chart_data` is 150 lines of heuristic spaghetti.**

Lines 970-1114: This method tries every possible data shape, searches for numeric/label columns by keyword matching, and applies chart type heuristics. It's a fragile chain of if/elif/for/break logic that's tightly coupled to the exact output shapes of every tool. Add a new tool with a slightly different output format, and this breaks silently.

**2. The `_nonempty_stream` wrapper (line 526) swallows empty chunks but not errors.**

If the generator yields `None` or an empty string, it's filtered. But if it raises an exception, the entire streaming response crashes because Django's `StreamingHttpResponse` doesn't handle generator exceptions gracefully.

**3. Visualization is computed 3 separate times for the same message.**

In the `stream()` function, visualization data is computed:
1. On ToolMessage receipt (lines 288-296 for table viz, lines 330-365 for auto viz)
2. After the stream loop ends (lines 466-481, another auto viz attempt)
3. In `_inject_chart_data` which can override the chart type again

This means 3 potentially different visualization objects are generated for one message, with the last one winning. The LLM costs from the `VisualizationAnalysisTool` are being paid but the results may be discarded.

---

## Summary of Critical Issues (Ranked)

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 1 | **High** | SQL path returns unstructured text, breaking visualization pipeline | [tools.py:1746-1757](workflows/tools.py#L1746-L1757) |
| 2 | **High** | Visualization LLM call is wasted — heuristics precede and override it | [tools.py:1856-1876](workflows/tools.py#L1856-L1876) |
| 3 | **High** | `SYSTEM_TEMPLATE` is ~120 lines — attention dilution, tool selection ambiguity | [prompt.py:38-154](workflows/prompt.py#L38-L154) |
| 4 | **Medium** | No few-shot examples in any prompt — LLM has to guess tool call format | All prompts |
| 5 | **Medium** | `_filter_messages` strips retry context, breaking SQL error correction | [sql_agent.py:613-626](workflows/sql_agent.py#L613-L626) |
| 6 | **Medium** | `_match_entity_name` substring matching causes false positives | [tools.py:1407-1414](workflows/tools.py#L1407-L1414) |
| 7 | **Medium** | `AudioAnalysisTool` and `DataAnalysisTool` have overlapping responsibilities | [tools.py:682-1198](workflows/tools.py#L682-L1198) and [tools.py:1376-1757](workflows/tools.py#L1376-L1757) |
| 8 | **Medium** | Global agent caching — all users share one instance | [views.py:738-765](views.py#L738-L765) |
| 9 | **Low** | `AgentState(Dict)` instead of TypedDict | [agent_workflow.py:24](workflows/agent_workflow.py#L24) |
| 10 | **Low** | Visualization computed 3 times per message | [views.py:288-481](views.py#L288-L481) |

---

## What You're Doing Well

1. **SQL injection prevention is genuinely strong.** The multi-layered validation in `sql_agent.py` (regex patterns → keyword blocklist → table authorization → subquery recursion → limit enforcement) is more thorough than most production systems I've reviewed.

2. **The structured-tool + SQL-fallback dual-path architecture** is the right design for an NL→data system. It gives you fast, reliable responses for common queries while maintaining flexibility for ad-hoc questions.

3. **Streaming response handling** with proper cleanup on disconnect (`GeneratorExit`) is well-implemented. Many teams skip this.

4. **Pagination preservation** through `QueryCacheModel` and the `paginate_message` endpoint is a thoughtful UX touch that most similar systems lack.

The foundation is solid. The main work needed is prompt consolidation, eliminating the wasted LLM call in visualization, and making the SQL path return structured data.
