import json
from typing import Any, Dict, List, Optional, Annotated
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
from django.utils import timezone

from .tools import AGENT_TOOLS
from data_insights.models import QueryCacheModel


class AgentState(Dict):

    messages: Annotated[List[AnyMessage], add]
    user_id: int
    session_id: int
    current_query_handles: List[str]
    context_data: Dict[str, Any]


class DataAgent:
  

    def __init__(
        self,
        llm: BaseChatModel,
        system_prompt: str,
        max_retries: int = 3,
        enable_caching: bool = True,
        session_timeout_hours: int = 24,
        tools: Optional[List[Any]] = None,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.enable_caching = enable_caching
        self.session_timeout_hours = session_timeout_hours
        self.tools = tools or list(AGENT_TOOLS)

        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Create tool node for executing tools
        self.tool_node = ToolNode(self.tools)

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

            messages = [system_msg] + state["messages"]

            response = self.llm_with_tools.invoke(messages)

            # Ensure response is an AIMessage, not HumanMessage
            if hasattr(response, "tool_calls") and response.tool_calls:
                self._update_query_handles(state, response.tool_calls)

            # Only return AIMessage objects to avoid serialization issues
            if hasattr(response, 'content') and hasattr(response, 'tool_calls'):
                return {"messages": [response]}
            else:
                # If response is not an AIMessage, create one
                safe_response = AIMessage(content=str(response) if response else "No response")
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
        """Post-process tool results to extract query handles"""
        last_message = state["messages"][-1]
        current_handles = state.get("current_query_handles", [])

        if isinstance(last_message, ToolMessage):
            try:
                result = json.loads(last_message.content)
                if isinstance(result, dict) and "query_id" in result:
                    query_id = result["query_id"]
                    if query_id not in current_handles:
                        current_handles.append(query_id)
                        logger.info(f"Added query handle: {query_id}")
            except (json.JSONDecodeError, TypeError):
                # Content might not be JSON, that's okay
                pass

        return {"current_query_handles": current_handles}

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
        """Format the final response with helpful context"""
        last_message = state["messages"][-1]

        # If there are active query handles, add helpful context
        current_handles = state.get("current_query_handles", [])
        if current_handles and isinstance(last_message, AIMessage):
            handle_info = "\n\n**Active Query Handles:**\n"
            for handle in current_handles[-3:]:  # Show last 3 handles
                try:
                    cache_entry = QueryCacheModel.objects.get(query_id=handle)
                    handle_info += f"- `{handle}`: {cache_entry.query_type} ({cache_entry.result_count} records)\n"
                except QueryCacheModel.DoesNotExist:
                    continue

            if len(handle_info) > len("\n\n**Active Query Handles:**\n"):
                enhanced_content = last_message.content + handle_info
                enhanced_message = AIMessage(content=enhanced_content)
                return {"messages": [enhanced_message]}

        return {}

    def compile_workflow(
        self, checkpointer: Optional[BaseCheckpointSaver] = None
    ) -> Any:
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("agent", self.call_model)
        graph.add_node("tools", self.tool_node)
        graph.add_node("post_process", self.post_process_tools)
        graph.add_node("cleanup", self.cleanup_expired_handles)
        graph.add_node("format", self.format_response)

        # Add edges
        graph.set_entry_point("cleanup")  # Start by cleaning up expired handles
        graph.add_edge("cleanup", "agent")

        # Conditional edge from agent
        graph.add_conditional_edges(
            "agent", self.should_continue, {"tools": "tools", "__end__": "format"}
        )

        # After tools, post-process then go back to agent
        graph.add_edge("tools", "post_process")
        graph.add_edge("post_process", "agent")

        # Final formatting
        graph.add_edge("format", "__end__")

        # Compile with checkpointer for conversation memory
        compiled = graph.compile(checkpointer=checkpointer)

        logger.info("CRM Agent workflow compiled successfully")
        return compiled

    def create_initial_state(
        self,
        user_input: str,
        user_id: int,
        session_id: int,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> AgentState:
        """Create initial state for the agent"""

        return AgentState(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_id": user_id,
                "session_id": session_id,
                "current_query_handles": [],
                "context_data": context_data or {},
            }
        )

    def process_message(
        self,
        user_input: str,
        user_id: int,
        session_id: int,
        context_data: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        checkpointer: Optional[BaseCheckpointSaver] = None,
    ) -> Any:
        initial_state = self.create_initial_state(
            user_input, user_id, session_id, context_data
        )

        workflow = self.compile_workflow(checkpointer)

        config = {"configurable": {"thread_id": str(session_id)}}

        if stream:
            return workflow.stream(initial_state, config=config, stream_mode="messages")
        else:
            result = workflow.invoke(initial_state, config=config)
            return result

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


def create_data_insights_agent(llm: BaseChatModel, system_prompt: str, **kwargs) -> DataAgent:
    return DataAgent(llm=llm, system_prompt=system_prompt, **kwargs)
