# Data Insights App — Complete Improvement Plan

## Priority Order Summary

| Priority | Task | Impact | Effort |
|---|---|---|---|
| 1 | Strict `ToolResult` Contract | Eliminates fragility | Low |
| 2 | GPT-4 as the Brain (Tool Calling) | Replaces custom skill router | Medium |
| 3 | Natural Language Insight Layer | Biggest UX jump | Medium |
| 4 | Session Memory | Enables conversational flow | Medium |
| 5 | Agentic Tool Orchestration | Real reasoning power | High |
| 6 | Anomaly Detection | Proactive intelligence | Medium |
| 7 | Follow-up Chips | Discoverability | Low |
| 8 | JS Modularization | Maintainability | Low |

---

## Priority 1 — Strict `ToolResult` Contract
*(Low effort, eliminates fragility)*

Kills the 275-line `extractChartData()` and removes the regex last-resort path entirely. Do this first — everything else builds on clean data contracts.

```python
@dataclass
class ToolResult:
    analysis_type: str
    rows: list[dict]
    columns: list[str]
    chart_hint: dict          # {x, y, group_by, type}
    metadata: dict            # anything extra (totals, units, etc.)
    skip_visualization: bool = False
```

---

## Priority 2 — GPT-4 as the Brain
*(Replaces custom skill router)*

**Key conclusion from skills discussion:** Don't build a custom skill registry or router. Wrap existing tools as OpenAI function schemas and let GPT-4 decide which to call. The `description` field *is* your router.

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=conversation_history,
    tool_choice="auto",
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_decibel_data",
                "description": "Fetch decibel readings. Use when user asks about loudness, sound levels, or decibel comparisons.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "group_by": {
                            "type": "string",
                            "enum": ["region", "collector", "category"]
                        },
                        "order": {
                            "type": "string",
                            "enum": ["highest", "lowest"]
                        }
                    }
                }
            }
        },
        # ... all other existing tools wrapped the same way
    ]
)
```

### Handle the Tool Loop Inside Existing LangGraph

```python
while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=conversation_history,
        tools=tools,
        tool_choice="auto"
    )
    message = response.choices[0].message

    if not message.tool_calls:
        break  # GPT-4 has a final answer

    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        # Dispatch to existing tool methods
        result = dispatch_tool(tool_name, tool_args)

        conversation_history.append(message)
        conversation_history.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })
```

### Multi-Tool Chaining (Unlocked Automatically)

```
"Why is Accra louder than Kumasi?"
    → GPT-4 calls _decibel_grouped      (sees the gap)
    → GPT-4 calls _temporal_analysis    (checks time-of-day)
    → GPT-4 calls _correlation_analysis (checks contributing factors)
    → synthesizes all three into one explanation
```

### Note on DeepSeek

DeepSeek V3 is significantly cheaper and API-compatible, but tool calling is less battle-tested. Build and stabilize with GPT-4o first. Swapping later is one line:

```python
# DeepSeek drop-in replacement (when ready)
client = OpenAI(
    api_key="your-deepseek-key",
    base_url="https://api.deepseek.com"
)
model = "deepseek-chat"
```

---

## Priority 3 — Natural Language Insight Layer
*(Biggest visible UX jump)*

Add one GPT-4 call after every chart renders. Touches nothing in the existing pipeline.

```python
insight = client.chat.completions.create(
    model="gpt-4o",
    max_tokens=200,
    messages=[{
        "role": "user",
        "content": f"""
            User asked: '{query}'
            Data returned: {tool_result_summary}

            Give 2 sharp observations. Flag any anomalies.
            Suggest one follow-up question.
        """
    }]
)
```

### Before vs After

```
# Before
User: "Why are Accra recordings louder?"
App:  [bar chart]

# After
User: "Why are Accra recordings louder?"
App:  [bar chart]
      "Accra's average is 78.3dB — 12% above the national
       average. The gap is strongest in Q1, suggesting
       seasonal or event-driven factors."

      [Follow-up chips]
      "Show Accra trend over time"
      "Compare Accra collectors"
      "What's driving Q1 specifically?"
```

---

## Priority 4 — Session Memory
*(Enables conversational flow)*

```python
session_memory = {
    "active_dataset": "GESMA_v2",
    "last_group_by": "region",
    "last_metric": "avg_decibel",
    "last_entities": ["Accra", "Kumasi"],
    "user_filters": {"date_range": "2024-01"},
    "chart_history": ["bar_chart:region", "line_chart:monthly"]
}
```

Pass into every GPT-4 call as part of the system prompt. Store in existing LangGraph `PostgresSaver` — no new infrastructure needed.

### What This Enables

- *"Show me the same but for last month"* — knows what "the same" means
- *"Drill into Accra"* — knows Accra was in the previous chart
- *"Why did it spike?"* — knows which metric and timeframe to investigate

---

## Priority 5 — Agentic Tool Orchestration
*(Real reasoning power — multi-step, self-directed)*

Right now the pipeline is linear: one query → one tool → one chart. Claude-like power comes from multi-step reasoning over tools — the agent picks a tool, inspects the result, decides if it needs more data, and only concludes when it has enough to give a complete answer.

### Current vs Agentic

```
User: "Why are decibel levels higher in Accra than Kumasi?"

Current:  query → _decibel_grouped → bar chart  ✓ (shows the gap)

Agentic:  query → _decibel_grouped (sees gap)
               → _temporal_analysis (checks if time-of-day explains it)
               → _correlation_analysis (checks against traffic/population)
               → synthesizes: "Accra's peak is 7–9am, correlates with..."
```

### How to Build It

Replace the single tool call with a LangGraph agent loop. You already use LangGraph's `PostgresSaver` — you're halfway there.

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 1. Define your tools as a list
tools = [
    get_decibel_data,
    get_temporal_analysis,
    get_correlation_analysis,
    get_overview_analysis,
    # ... all your existing tools
]

# 2. Bind tools to GPT-4
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)

# 3. Agent node — decides what to do next
def agent_node(state):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# 4. Router — did GPT-4 call a tool or is it done?
def should_continue(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 5. Wire the graph
builder = StateGraph(dict)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")  # loop back after each tool call

graph = builder.compile(checkpointer=postgres_saver)  # your existing saver
```

### What the Loop Looks Like at Runtime

```
Turn 1: GPT-4 calls _decibel_grouped
        → sees Accra is 12% louder
        → decides it needs more context

Turn 2: GPT-4 calls _temporal_analysis
        → sees the gap peaks at 7–9am
        → decides to check if it's traffic-related

Turn 3: GPT-4 calls _correlation_analysis
        → confirms correlation with urban density
        → has enough to answer

Final:  "Accra's peak decibel readings occur between 7–9am,
         correlating with morning traffic in high-density areas.
         Kumasi shows a flatter daily profile, suggesting a
         more distributed sound source pattern."
```

The agent decides how many tool calls it needs. You don't hardcode the chain — GPT-4 figures it out from the data.

---

## Priority 6 — Proactive Anomaly Detection
*(Proactive intelligence)*

Run automatically after every tool result, before rendering.

```python
def check_anomalies(rows: list[dict], metric: str) -> list[dict]:
    import statistics
    values = [r[metric] for r in rows]
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    return [
        r for r in rows
        if abs(r[metric] - mean) > 2 * stdev
    ]
```

Surface findings automatically in the UI:

```
⚠️ 3 anomalies detected — Tamale's August reading
   is 2.4σ above baseline. Investigate?
```

---

## Priority 7 — Follow-up Chips
*(Discoverability, low effort)*

```python
follow_ups = client.chat.completions.create(
    model="gpt-4o",
    max_tokens=100,
    messages=[{
        "role": "user",
        "content": f"""
            Data shown: {summary}
            Generate 3 specific follow-up questions a data analyst would ask.
            Return JSON array of strings only. No markdown.
        """
    }]
)
```

```javascript
// Frontend
follow_ups.forEach(q => {
    const chip = document.createElement('button');
    chip.className = 'followup-chip';
    chip.textContent = q;
    chip.onclick = () => sendPrompt(q);
    chipsContainer.appendChild(chip);
});
```

---

## Priority 8 — JS Modularization
*(Maintainability)*

Break the 3500-line template into static files. No bundler, no npm required.

```
templates/
  unified_chat.html              # ~200 lines, orchestration only

static/data_insights/js/
  chart_renderer.js              # createChart() + extractChartData()
  artifact_renderer.js           # renderArtifact() + all widget types
  table_renderer.js              # buildTableHtml()
```

```html
<!-- Load in unified_chat.html -->
<script src="{% static 'data_insights/js/chart_renderer.js' %}"></script>
<script src="{% static 'data_insights/js/artifact_renderer.js' %}"></script>
<script src="{% static 'data_insights/js/table_renderer.js' %}"></script>
```

---

## Full Architecture Picture

```
User Query
    │
    ▼
GPT-4 (reads tool descriptions, picks tool(s))
    │
    ▼
Tool Loop (LangGraph — existing)
    │
    ├── Tool 1 → ToolResult (strict contract)
    ├── Tool 2 → ToolResult
    └── Tool N → ToolResult
    │
    ▼
Anomaly Check (automatic, before render)
    │
    ▼
chart_builder → widget_composer (unchanged)
    │
    ▼
GPT-4 Insight Generation (2-3 sentences)
    │
    ▼
Frontend (chart + insight + follow-up chips)
```

> The entire visualization pipeline stays intact. GPT-4 sits at the top as router
> and at the bottom as explainer. Everything in the middle is already built and working.
