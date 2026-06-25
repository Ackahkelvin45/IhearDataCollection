import json
import re
import time
from typing import Any, Dict, List, Optional, Annotated, TypedDict
from operator import add

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from loguru import logger
from django.conf import settings
from django.utils import timezone

from .tools import AGENT_TOOLS
from .chart_builder import resolve_chart
from .clarification_gate import (
    clarification_gate_node,
    clarification_resolver_node,
    emit_clarification_node,
    should_clarify,
)
from data_insights.models import QueryCacheModel


AGENT_CONFIG = getattr(settings, "AI_INSIGHT", {}).get("AGENT", {})

# Bounds the worst-case LangGraph tool-loop depth per run so a runaway
# agent loop cannot rack up unbounded LLM spend. Configurable via
# AI_INSIGHT["AGENT"]["RECURSION_LIMIT"] (env: AI_INSIGHT_RECURSION_LIMIT),
# default 15.
RECURSION_LIMIT = AGENT_CONFIG.get("RECURSION_LIMIT", 15)


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add]
    user_id: int
    session_id: int
    mode: str
    current_query_handles: List[str]
    context_data: Dict[str, Any]
    # Clarification fields
    clarification_pending: bool
    clarification_question: Optional[str]
    clarification_options: Optional[List[str]]
    clarification_answer: Optional[str]
    clarification_is_custom: bool
    clarification_dimension: Optional[str]
    original_query: Optional[str]
    confirmation_needed: bool
    confirmation_resolved_value: Optional[str]
    # Visualization — computed once in post_process_tools, emitted once by views.py
    pending_chart: Optional[Dict[str, Any]]
    # Multi-widget artifact — superset of pending_chart for dashboard rendering
    pending_artifact: Optional[Dict[str, Any]]
    # Clarification payload — written by emit_clarification_node, read by views.py
    # via the checkpointer (same pattern as pending_chart/pending_artifact).
    # Avoids the per-worker module dict that breaks under multi-worker deployments.
    pending_clarification_payload: Optional[Dict[str, Any]]
    # Data leakage tracking — set True when a full-dataset aggregate tool runs.
    # ml_train_test_split checks this to emit a leakage warning.
    _full_dataset_aggregates_computed: bool


# ── System prompt for the finalize_dashboard structured LLM call ──────────

_DASHBOARD_SUMMARY_PROMPT = """You are a data analyst reviewing a dashboard that was just built.
The user asked a question about audio data, and we queried the database.
Now we have a dashboard with charts, stats, and tables.

Given a data summary, produce exactly:
1. A 2-4 sentence narrative summary highlighting the single most important finding, one notable pattern or outlier. Do NOT list data row by row. End with an invitation to drill deeper.
2. Exactly 3 short, sharp observations. One surprising finding. One pattern. One thing worth investigating further. Each observation: 1 sentence.

IMPORTANT: If the data in the summary does NOT appear to answer the user's question
(e.g. user asked about frequency but the data shows decibel levels, or user asked
about energy but the data shows counts), say so in the summary instead of describing
the data. For example: "The data returned is about decibel levels rather than
frequency — the system may have misinterpreted the query."

Respond ONLY with valid JSON in this exact format:
{"summary": "...", "observations": ["...", "...", "..."]}"""

_DASHBOARD_SUMMARY_PROMPT_ML = """You are a machine learning data assistant reviewing a dataset analysis dashboard.
The user asked a question about ML readiness, and we analyzed the database.

Given a data summary, produce exactly:
1. A 2-4 sentence narrative summary focused on modeling implications. What is the single
   most important finding for someone preparing to train a model — class balance,
   feature coverage, data quality, or a notable distribution? End with a concrete
   next step (e.g. "Consider stratifying by category during splitting" or
   "The low feature coverage suggests prioritizing feature extraction for 30% of
   the dataset before training").

2. Exactly 3 short, sharp observations. One about data quality or coverage.
   One about class balance or distribution. One about feature characteristics or
   a specific modeling concern. Each observation: 1 sentence, actionable.

Use ML terminology naturally: class distribution, feature variance, stratification,
coverage, missingness, train/val/test split, minority class, imbalance severity.

IMPORTANT: If the data does NOT appear to answer the user's question, say so instead
of describing irrelevant data. For example: "The data returned shows audio feature
statistics rather than class balance — the system may have misinterpreted the query."

Respond ONLY with valid JSON in this exact format:
{"summary": "...", "observations": ["...", "...", "..."]}"""


# ── Merge helpers ──────────────────────────────────────────────────────────


def _best_layout(count: int) -> str:
    if count <= 1:
        return "single"
    if count <= 3:
        return "two_column"
    return "grid"


def _merge_artifacts(existing: Dict, new: Dict) -> Dict:
    """Merge two artifacts. All widgets re-sorted globally by priority.

    Colliding widget IDs from 'new' are renamed with a '_2' suffix.
    """
    existing_widgets = existing.get("widgets", [])
    new_widgets = new.get("widgets", [])

    # Dedup: rename colliding IDs
    existing_ids = {w["id"] for w in existing_widgets}
    for w in new_widgets:
        if w["id"] in existing_ids:
            w["id"] = f"{w['id']}_2"
        existing_ids.add(w["id"])

    # Re-sort ALL widgets globally by priority
    all_widgets = existing_widgets + new_widgets
    all_widgets.sort(key=lambda w: w.get("priority", 0))

    merged = {
        **existing,
        "widgets": all_widgets,
        "layout_template": _best_layout(len(all_widgets)),
        "follow_ups": existing.get("follow_ups", []) + new.get("follow_ups", []),
    }
    # Preserve merge metadata from the most recent caller
    if "_merge_group" in new:
        merged["_merge_group"] = new["_merge_group"]
    if "_analysis_type" in new:
        merged["_analysis_type"] = new["_analysis_type"]
    return merged


def _build_data_summary(
    artifact: Dict[str, Any], user_question: str = ""
) -> Dict[str, Any]:
    """Build a compact data summary from the artifact for the insight LLM."""
    widgets = artifact.get("widgets", [])
    chart_widgets = [
        w
        for w in widgets
        if isinstance(w.get("data"), dict) and w["data"].get("labels")
    ]
    stat_widgets = [w for w in widgets if w.get("type") == "stat_card"]

    summary: Dict[str, Any] = {
        "widget_count": len(widgets),
    }
    if user_question:
        summary["user_question"] = user_question

    # Extract data from first chart widget
    total_rows = 0
    if chart_widgets:
        c = chart_widgets[0]
        cd = c.get("data", {})
        labels = cd.get("labels", [])
        data_values = cd.get("data", [])
        stats_data = {
            "title": c.get("title", ""),
            "labels_count": len(labels),
            "max_value": round(max(data_values), 2) if data_values else None,
            "min_value": round(min(data_values), 2) if data_values else None,
            "sample_labels": labels[:5],
        }
        # Add avg if values exist
        if data_values:
            stats_data["avg_value"] = round(sum(data_values) / len(data_values), 2)
        summary["primary_chart"] = stats_data
        total_rows = len(labels)

    # Extract stats from stat widgets
    if stat_widgets:
        summary["statistics"] = stat_widgets[0].get("data", {}).get("stats", {})

    summary["total_rows"] = total_rows
    return summary


class DataAgent:

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        max_retries: int = 3,
        enable_caching: bool = True,
        session_timeout_hours: int = 24,
        tools: Optional[List[Any]] = None,
        dashboard_llm: Optional[BaseChatModel] = None,
    ):
        self.llm = llm
        self.dashboard_llm = dashboard_llm or llm
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.enable_caching = enable_caching
        self.session_timeout_hours = session_timeout_hours
        self.tools = tools or list(AGENT_TOOLS)

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create tool node for executing tools
        self.tool_node = ToolNode(self.tools)

        self._compiled_graph: Optional[Any] = None
        self._compiled_checkpointer: Optional[BaseCheckpointSaver] = None

        logger.info(f"CRM Agent initialized with {len(self.tools)} tools")

    def _create_system_message(
        self, user_context: Dict[str, Any] | None = None
    ) -> SystemMessage:
        context_info = ""
        if user_context:
            context_info = f"""

## Current Context
- User ID: {user_context.get("user_id", "Unknown")}
- Session ID: {user_context.get("session_id", "Unknown")}
- Active Query Handles: {", ".join(user_context.get("current_query_handles", []))}
"""

        return SystemMessage(content=self.system_prompt + context_info)

    @staticmethod
    def _looks_like_clarification(content: str) -> bool:
        """Heuristic check: does the LLM response look like it's asking the
        user to clarify rather than providing an answer?

        Phase 2 observation only — we log matches but don't intercept.
        Phase 3 would route matches through the structured gate.
        """
        if "?" not in content:
            return False
        patterns = [
            r"\bdid you mean\b",
            r"\bwhich (one|of these)\b",
            r"\bcould you (specify|clarify)\b",
            r"\bwould you like\b",
            r"\bdo you (want|mean|prefer)\b",
            r"\bare you (asking|referring|looking)\b",
            r"\bI (need|want) to (clarify|confirm|understand)\b",
            r"\bcan you (specify|clarify|tell me more)\b",
        ]
        content_lower = content.lower()
        return any(re.search(p, content_lower) for p in patterns)

    def should_continue(self, state: AgentState) -> str:
        """Determine next step in the workflow"""
        last_message = state["messages"][-1]

        # If the last message has tool calls, execute them
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        # Otherwise, end the conversation
        return "__end__"

    def call_model(self, state: AgentState) -> Dict[str, Any]:
        try:
            user_context = {
                "user_id": state.get("user_id"),
                "session_id": state.get("session_id"),
                "current_query_handles": state.get("current_query_handles", []),
            }

            system_msg = self._create_system_message(user_context)

            # Inject clarification context when present
            if state.get("clarification_answer"):
                from .prompt import CLARIFICATION_CONTEXT_TEMPLATE

                clarification_context = CLARIFICATION_CONTEXT_TEMPLATE.format(
                    dimension=state.get("clarification_dimension", ""),
                    resolved_value=state.get("clarification_answer", ""),
                )
                system_msg = SystemMessage(
                    content=system_msg.content + clarification_context
                )

            messages = [system_msg] + state["messages"]

            response = self.llm_with_tools.invoke(messages)

            # Ensure response is an AIMessage, not HumanMessage
            if hasattr(response, "tool_calls") and response.tool_calls:
                self._update_query_handles(state, response.tool_calls)

            # Phase 2 observability: detect LLM free-text clarifications.
            # When the LLM asks a clarification question instead of calling a tool,
            # log it so we can measure how often the prompt-based deferral is working.
            content = getattr(response, "content", "") or ""
            if (
                isinstance(content, str)
                and not getattr(response, "tool_calls", None)
                and self._looks_like_clarification(content)
            ):
                logger.info(
                    f"llm_freetext_clarification: content={content[:200]} "
                    f"session={state.get('session_id')}"
                )

            # Only return AIMessage objects to avoid serialization issues
            if hasattr(response, "content") and hasattr(response, "tool_calls"):
                return {"messages": [response]}
            else:
                # If response is not an AIMessage, create one
                safe_response = AIMessage(
                    content=str(response) if response else "No response"
                )
                return {"messages": [safe_response]}

        except Exception as e:
            # Ensure error message is JSON serializable
            error_str = str(e) if e else "Unknown error occurred"
            logger.error(f"Error calling model: {error_str}")
            error_msg = AIMessage(content="I encountered an error. Please try again.")
            return {"messages": [error_msg]}

    def _update_query_handles(self, state: AgentState, tool_calls: List[Any]) -> None:
        current_handles = state.get("current_query_handles", [])

        for tool_call in tool_calls:
            # Check if this is a search tool that might create handles
            if tool_call["name"] in [
                "search_customers",
                "search_dormant_accounts",
                "search_by_segmentation",
            ]:
                # We'll update this after tool execution in post_process_tools
                pass

    def post_process_tools(self, state: AgentState) -> Dict[str, Any]:
        """Post-process tool results to extract query handles and compute visualization."""
        last_message = state["messages"][-1]
        current_handles = state.get("current_query_handles", [])
        pending_chart = None
        pending_artifact = None
        analysis_type = None
        merge_group = None

        if isinstance(last_message, ToolMessage):
            result = None
            try:
                result = json.loads(last_message.content)
                if isinstance(result, dict) and "query_id" in result:
                    query_id = result["query_id"]
                    if query_id not in current_handles:
                        current_handles.append(query_id)
                        logger.info(f"Added query handle: {query_id}")

                # Compute visualization once here — the single source of truth.
                # Views.py reads pending_chart / pending_artifact from the final state.
                if isinstance(result, dict):
                    skip_viz = result.get("skip_visualization", False)
                    analysis_type = result.get("analysis_type", "")
                    merge_group = result.get("merge_group")
                    if not skip_viz:
                        # Route to decompose() for all multi-widget types
                        from .widget_composer import decompose

                        pending_artifact = decompose(result)
                        # If decompose returned None, fall through to single-chart path
                        if not pending_artifact:
                            pending_chart = resolve_chart(result)
            except (json.JSONDecodeError, TypeError):
                # Tool returned non-JSON content — log it since the tool contract expects JSON
                logger.warning(
                    f"Tool returned non-JSON content (length={len(str(last_message.content))}): "
                    f"{str(last_message.content)[:200]}"
                )

                # Check for tool ambiguity sentinel even when JSON parsing failed.
                # The _tool_ambiguity sentinel is valid JSON so this only fires if
                # the tool returned something unexpected.
                content_str = str(last_message.content) if last_message.content else ""
                if "_tool_ambiguity" in content_str:
                    logger.info("Tool ambiguity sentinel detected in non-JSON response")
                    return self._handle_tool_ambiguity(state, result={})

        # Check for tool ambiguity sentinel in successfully parsed tool result.
        # When a tool cannot determine the right parameters, it returns
        # {_tool_ambiguity: True, dimension: "analysis_type", query: "..."}
        # instead of silently guessing. Route back to the clarification gate.
        if isinstance(result, dict) and result.get("_tool_ambiguity"):
            return self._handle_tool_ambiguity(state, result)

        # Wrap single chart into artifact format (for non-multi-view tools)
        if pending_chart and not pending_artifact:
            from .widget_composer import wrap_as_artifact

            pending_artifact = wrap_as_artifact(
                pending_chart, analysis_type=analysis_type or None
            )

        # Attach merge_group to the new artifact for merge detection
        if pending_artifact is not None and merge_group:
            pending_artifact["_merge_group"] = merge_group
            pending_artifact["_analysis_type"] = analysis_type

        # Merge sibling artifacts (same merge_group from a prior tool call)
        if pending_artifact is not None:
            existing = state.get("pending_artifact")
            if existing:
                existing_mg = existing.get("_merge_group")
                new_mg = pending_artifact.get("_merge_group")
                if existing_mg and existing_mg == new_mg:
                    pending_artifact = _merge_artifacts(existing, pending_artifact)
                    logger.info(
                        "Merged sibling artifacts (merge_group=%s): %d+%d widgets",
                        existing_mg,
                        len(existing.get("widgets", [])),
                        len(pending_artifact.get("widgets", [])),
                    )

        update: Dict[str, Any] = {
            "current_query_handles": current_handles,
        }
        # Only overwrite state when a valid chart/artifact was computed.
        # A subsequent tool call that produces no chart (e.g. a scalar
        # count or a defunct visualization_analysis call) must not clobber
        # the chart from an earlier data tool call.
        if pending_chart is not None:
            update["pending_chart"] = pending_chart
        if pending_artifact is not None:
            update["pending_artifact"] = pending_artifact
        return update

    def _handle_tool_ambiguity(
        self, state: AgentState, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a tool's _tool_ambiguity sentinel by routing back to the gate."""
        dimension = result.get("dimension", "analysis_type")
        query = result.get("query", "")
        logger.info(
            f"Tool ambiguity detected: dimension={dimension} query={query[:100]}"
        )

        from .clarification_gate import ANALYSIS_TYPE_LABELS

        payload = {
            "question": "What kind of analysis do you want?",
            "options": [v for v in ANALYSIS_TYPE_LABELS.values()]
            + ["All of the above"],
            "dimension": dimension,
        }

        return {
            "clarification_pending": True,
            "clarification_question": payload["question"],
            "clarification_options": payload["options"],
            "clarification_dimension": dimension,
            "clarification_answer": None,
            "clarification_is_custom": False,
            "original_query": query,
            "pending_clarification_payload": payload,
            "current_query_handles": state.get("current_query_handles", []),
        }

    def should_continue_from_post_process(self, state: AgentState) -> str:
        """Route after post_process: emit_clarification if tool ambiguity, else agent."""
        if state.get("clarification_pending") and not state.get("clarification_answer"):
            return "emit_clarification"
        return "agent"

    def cleanup_expired_handles(self, state: AgentState) -> Dict[str, Any]:
        """Clean up expired query handles"""
        current_handles = state.get("current_query_handles", [])
        valid_handles = []

        for handle in current_handles:
            try:
                cache_entry = QueryCacheModel.objects.get(query_id=handle)
                if not cache_entry.is_expired:
                    valid_handles.append(handle)
                else:
                    logger.info(f"Removing expired query handle: {handle}")
            except QueryCacheModel.DoesNotExist:
                logger.info(f"Removing non-existent query handle: {handle}")

        return {"current_query_handles": valid_handles}

    def format_response(self, state: AgentState) -> Dict[str, Any]:
        """Format the final response with helpful context.

        Adds a dashboard pointer when widgets exist and the LLM text
        doesn't already reference the dashboard. Also appends active
        query handle info for debugging.
        """
        last_message = state["messages"][-1]
        pending_artifact = state.get("pending_artifact")
        content = (
            last_message.content or "" if isinstance(last_message, AIMessage) else ""
        )

        additions = []

        # Dashboard pointer (only if not redundant with LLM summary)
        if pending_artifact and isinstance(last_message, AIMessage):
            lower = content.lower()
            if (
                "dashboard" not in lower
                and "chart" not in lower
                and "below" not in lower
            ):
                additions.append("*See the dashboard below for the full breakdown.*")

        # Active query handles context
        current_handles = state.get("current_query_handles", [])
        if current_handles:
            handle_info = "\n\n**Active Query Handles:**\n"
            for handle in current_handles[-3:]:
                try:
                    cache_entry = QueryCacheModel.objects.get(query_id=handle)
                    handle_info += f"- `{handle}`: {cache_entry.query_type} ({cache_entry.result_count} records)\n"
                except QueryCacheModel.DoesNotExist:
                    continue
            if len(handle_info) > len("\n\n**Active Query Handles:**\n"):
                additions.append(handle_info)

        if additions:
            enhanced_content = content + "\n\n" + "\n".join(additions)
            return {"messages": [AIMessage(content=enhanced_content)]}

        return {}

    def finalize_dashboard(self, state: AgentState) -> Dict[str, Any]:
        """Generate a dashboard-aware summary + observations.

        Runs ONCE after the agent loop terminates, when the complete artifact is
        available. Makes one structured LLM call returning {summary, observations}.

        Guards:
        - Only replaces the agent's last message when it has no tool_calls and has
          string content.
        - Skips entirely for trivial data (single row, no chart).
        - If the LLM call fails, keeps the original message and skips the insight
          widget (degraded but not broken).
        """
        artifact = state.get("pending_artifact")
        if not artifact or not artifact.get("widgets"):
            return {}

        # Extract the user's original question for mismatch detection
        user_question = ""
        for msg in state.get("messages", []):
            if (
                isinstance(msg, HumanMessage)
                and isinstance(msg.content, str)
                and msg.content.strip()
            ):
                user_question = msg.content[:500]
                break

        data_summary = _build_data_summary(artifact, user_question=user_question)

        # Guard: skip for trivial data (single row, no chart)
        if (
            data_summary.get("widget_count", 0) <= 1
            or data_summary.get("total_rows", 0) <= 1
        ):
            return {}

        is_ml_mode = state.get("mode", "analysis") == "ml"
        summary_prompt = (
            _DASHBOARD_SUMMARY_PROMPT_ML if is_ml_mode else _DASHBOARD_SUMMARY_PROMPT
        )

        start = time.monotonic()
        try:
            response = self.dashboard_llm.invoke(
                [
                    SystemMessage(content=summary_prompt),
                    HumanMessage(
                        content=f"Data summary:\n{json.dumps(data_summary, indent=2)}"
                    ),
                ]
            )
            elapsed = int((time.monotonic() - start) * 1000)
            logger.info(
                "finalize_dashboard llm_ms=%d widget_count=%d rows=%d",
                elapsed,
                len(artifact.get("widgets", [])),
                data_summary.get("total_rows", 0),
            )

            parsed = json.loads(response.content)

            # Append observations as insight widget
            observations = (parsed.get("observations") or [])[:3]
            if observations:
                artifact.setdefault("widgets", []).append(
                    {
                        "id": "insights",
                        "type": "insight",
                        "title": "Key Observations",
                        "data": {"observations": observations},
                        "priority": 999,
                    }
                )

            # Replace agent's last message content with dashboard-aware summary
            summary_text = (parsed.get("summary") or "").strip()
            if summary_text:
                last_msg = state["messages"][-1]
                # Only replace if terminal text message (no tool_calls, string content)
                if (
                    isinstance(last_msg, AIMessage)
                    and not getattr(last_msg, "tool_calls", None)
                    and isinstance(last_msg.content, str)
                ):
                    state["messages"][-1] = AIMessage(
                        content=summary_text, id=last_msg.id
                    )

            return {"pending_artifact": artifact}

        except Exception as e:
            logger.exception("finalize_dashboard LLM call failed: %s", e)
            return {}  # degraded: keep original message, skip insight widget

    def compile_workflow(
        self, checkpointer: Optional[BaseCheckpointSaver] = None
    ) -> Any:
        if (
            self._compiled_graph is not None
            and self._compiled_checkpointer is checkpointer
        ):
            return self._compiled_graph

        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("agent", self.call_model)
        graph.add_node("tools", self.tool_node)
        graph.add_node("post_process", self.post_process_tools)
        graph.add_node("cleanup", self.cleanup_expired_handles)
        graph.add_node("finalize_dashboard", self.finalize_dashboard)
        graph.add_node("format", self.format_response)
        graph.add_node("clarification_gate", clarification_gate_node)
        graph.add_node("clarification_resolver", clarification_resolver_node)
        graph.add_node("emit_clarification", emit_clarification_node)

        # Add edges
        graph.set_entry_point("cleanup")
        graph.add_edge("cleanup", "clarification_gate")

        # Conditional: clear → agent, ambiguous → emit_clarification, answer → resolver
        graph.add_conditional_edges(
            "clarification_gate",
            should_clarify,
            {
                "agent": "agent",
                "emit_clarification": "emit_clarification",
                "clarification_resolver": "clarification_resolver",
            },
        )

        # After emit, the frontend sends back the answer → resolver → agent
        graph.add_edge("emit_clarification", "__end__")
        graph.add_edge("clarification_resolver", "agent")

        # Conditional edge from agent
        graph.add_conditional_edges(
            "agent",
            self.should_continue,
            {"tools": "tools", "__end__": "finalize_dashboard"},
        )

        # After tools, post-process. Under V2, tool ambiguity routes to
        # emit_clarification; normally routes back to agent.
        graph.add_edge("tools", "post_process")
        graph.add_conditional_edges(
            "post_process",
            self.should_continue_from_post_process,
            {
                "agent": "agent",
                "emit_clarification": "emit_clarification",
            },
        )

        # finalize_dashboard runs once after the agent loop, then format
        graph.add_edge("finalize_dashboard", "format")

        # Final formatting
        graph.add_edge("format", "__end__")

        # Compile with checkpointer for conversation memory
        compiled = graph.compile(checkpointer=checkpointer)

        self._compiled_graph = compiled
        self._compiled_checkpointer = checkpointer

        logger.info("CRM Agent workflow compiled successfully")
        return compiled

    def reuse_compiled_graph(
        self, compiled_graph: Any, checkpointer: Optional[BaseCheckpointSaver] = None
    ) -> None:
        """Reuse a pre-compiled graph from the cache so we don't rebuild the state
        machine on every request. The compiled graph is immutable and safe to share
        across agent instances — conversation state lives in the checkpointer, keyed
        by thread_id = session_id."""
        self._compiled_graph = compiled_graph
        self._compiled_checkpointer = checkpointer

    def create_initial_state(
        self,
        user_input: str,
        user_id: int,
        session_id: int,
        context_data: Optional[Dict[str, Any]] = None,
        mode: str = "analysis",
    ) -> AgentState:
        """Create initial state for the agent"""

        return {
            "messages": [HumanMessage(content=user_input)],
            "user_id": user_id,
            "session_id": session_id,
            "mode": mode,
            "current_query_handles": [],
            "context_data": context_data or {},
            "clarification_pending": False,
            "clarification_question": None,
            "clarification_options": None,
            "clarification_answer": None,
            "clarification_is_custom": False,
            "clarification_dimension": None,
            "original_query": None,
            "confirmation_needed": False,
            "confirmation_resolved_value": None,
            "pending_chart": None,
            "pending_artifact": None,
            "pending_clarification_payload": None,
            "_full_dataset_aggregates_computed": False,
        }

    def process_message(
        self,
        user_input: str,
        user_id: int,
        session_id: int,
        context_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        mode: str = "analysis",
    ) -> Any:
        initial_state = self.create_initial_state(
            user_input,
            user_id,
            session_id,
            context_data,
            mode=mode,
        )

        workflow = self.compile_workflow(checkpointer)

        config = {
            "configurable": {"thread_id": str(session_id)},
            "recursion_limit": RECURSION_LIMIT,
        }

        if stream:
            return workflow.stream(initial_state, config=config, stream_mode="messages")
        else:
            result = workflow.invoke(initial_state, config=config)
            return result

    async def process_message_async(
        self,
        user_input: str,
        user_id: int,
        session_id: int,
        context_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        mode: str = "analysis",
    ) -> Any:
        initial_state = self.create_initial_state(
            user_input,
            user_id,
            session_id,
            context_data,
            mode=mode,
        )

        workflow = self.compile_workflow(checkpointer)

        config = {
            "configurable": {"thread_id": str(session_id)},
            "recursion_limit": RECURSION_LIMIT,
        }

        if stream:
            return workflow.astream(
                initial_state, config=config, stream_mode="messages"
            )
        else:
            result = await workflow.ainvoke(initial_state, config=config)
            return result

    def process_clarification_response(
        self,
        user_id: int,
        session_id: int,
        clarification_answer: str,
        clarification_is_custom: bool,
        stream: bool = False,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Any:
        """Resume the graph after the user answered a clarification question.

        Creates an initial state with the clarification answer pre-set so the
        gate routes to the resolver node instead of re-detecting ambiguity.

        Only sets fields that changed — fields NOT in this dict (dimension,
        original_query, etc.) persist from the checkpointer's stored state.
        """
        state: AgentState = {
            "messages": [HumanMessage(content="[Clarification response]")],
            "user_id": user_id,
            "session_id": session_id,
            "current_query_handles": [],
            "context_data": {},
            "clarification_pending": True,
            "clarification_answer": clarification_answer,
            "clarification_is_custom": clarification_is_custom,
            "pending_chart": None,
            "pending_artifact": None,
            "pending_clarification_payload": None,
        }

        workflow = self.compile_workflow(checkpointer)
        config = {
            "configurable": {"thread_id": str(session_id)},
            "recursion_limit": RECURSION_LIMIT,
        }

        if stream:
            return workflow.stream(state, config=config, stream_mode="messages")
        else:
            return workflow.invoke(state, config=config)

    async def process_clarification_response_async(
        self,
        user_id: int,
        session_id: int,
        clarification_answer: str,
        clarification_is_custom: bool,
        stream: bool = False,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Any:
        """Async variant of process_clarification_response — uses astream for native
        async iteration, avoiding the sync-to-async bridge penalty on every chunk."""
        state: AgentState = {
            "messages": [HumanMessage(content="[Clarification response]")],
            "user_id": user_id,
            "session_id": session_id,
            "current_query_handles": [],
            "context_data": {},
            "clarification_pending": True,
            "clarification_answer": clarification_answer,
            "clarification_is_custom": clarification_is_custom,
            "pending_chart": None,
            "pending_artifact": None,
            "pending_clarification_payload": None,
        }

        workflow = self.compile_workflow(checkpointer)
        config = {
            "configurable": {"thread_id": str(session_id)},
            "recursion_limit": RECURSION_LIMIT,
        }

        if stream:
            return workflow.astream(state, config=config, stream_mode="messages")
        else:
            return await workflow.ainvoke(state, config=config)

    def get_session_summary(self, session_id: int) -> Dict[str, Any]:
        try:
            user_handles = QueryCacheModel.objects.filter(
                created_by_id=session_id,
                expires_at__gt=timezone.now(),
            ).values("query_id", "query_type", "result_count", "created_at")

            return {
                "session_id": session_id,
                "active_query_handles": list(user_handles),
                "handle_count": len(user_handles),
            }

        except Exception as e:
            logger.error(f"Error getting session summary: {str(e)}")
            return {"error": str(e)}


def create_data_insights_agent(
    llm: BaseChatModel, system_prompt: str, **kwargs
) -> DataAgent:
    return DataAgent(llm=llm, system_prompt=system_prompt, **kwargs)
