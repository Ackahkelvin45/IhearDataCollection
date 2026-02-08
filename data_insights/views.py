from django.shortcuts import render
from django.db.models import Count, Min, Max
from django.db import transaction
import os
from data.models import NoiseDataset, NoiseAnalysis
from core.models import Region, Community
from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticated
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework import status
from django_filters.rest_framework import DjangoFilterBackend
from .serializers import (
    ChatMessageCreateSerializer,
    ChatMessageDetailSerializer,
    ChatMessageListSerializer,
    ChatSessionArchiveSerializer,
    ChatSessionUpdateSerializer,
    MessageStatusSerializer,
    ChatSessionCreateSerializer,
    ChatSessionDetailSerializer,
    ChatSessionListSerializer,
)

from .models import ChatSession, ChatMessage
from rest_framework.decorators import action
import time
import json
from django.http import StreamingHttpResponse
from langchain_core.messages import (
    AIMessageChunk,
    ToolMessage,
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from langchain_openai import ChatOpenAI
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from loguru import logger
from typing import Any, Dict, cast
from django.conf import settings

from data_insights.workflows.agent_workflow import create_data_insights_agent
from data_insights.workflows.tools import get_tool_by_name
from data_insights.workflows.prompt import SYSTEM_TEMPLATE
from rest_framework.response import Response


AI_CONFIG = getattr(settings, "AI_INSIGHT", {})
DB_CONFIG = AI_CONFIG.get("DATABASE", {})
AGENT_CONFIG = AI_CONFIG.get("AGENT", {})
SECURITY_CONFIG = AI_CONFIG.get("SECURITY", {})

DB_URI = (
    f"postgresql://{DB_CONFIG.get('USER', 'admin')}:"
    f"{DB_CONFIG.get('PASSWORD', 'localhost')}@"
    f"{DB_CONFIG.get('HOST', 'db')}:"
    f"{DB_CONFIG.get('PORT', 5432)}/"
    f"{DB_CONFIG.get('NAME', 'brainbox-crm')}"
)


def home(request):
    """Landing page for the insights chat UI."""
    # Order suggestions so the first one maps to a chart
    suggestions = [
        "Which region has the most data collected?",
        "Show me recent 20 data collected",
        "Which data has the highest decibel level?",
        "Which community has the lowest decibel level?",
    ]
    ml_suggestions = [
        "Show label distribution by category",
        "How many datasets have audio features?",
        "What is the feature coverage percentage?",
        "Recommend a train/val/test split size",
    ]
    return render(
        request,
        "data_insights/home.html",
        {"suggestions": suggestions, "ml_suggestions": ml_suggestions},
    )


def chat(request):
    """Main chat interface."""
    return render(request, "data_insights/chat.html")


def unified_chat(request):
    """Unified chat interface with sessions, suggestions, and chat in one page."""
    suggestions = [
        "Which region has the most data collected?",
        "Show me recent 20 data collected",
        "Which data has the highest decibel level?",
        "Which community has the lowest decibel level?",
    ]
    return render(
        request, "data_insights/unified_chat.html", {"suggestions": suggestions}
    )


class ChatSessionView(ModelViewSet):
    permission_classes = [IsAuthenticated]
    filter_backends = [SearchFilter, OrderingFilter, DjangoFilterBackend]
    filterset_fields = ["title"]
    search_fields = ["title"]
    ordering_fields = ["-created_at", "-updated_at"]

    def get_serializer_class(self):
        if self.action == "list":
            return ChatSessionListSerializer
        elif self.action == "create":
            return ChatSessionCreateSerializer
        elif self.action in ["partial", "partial_update"]:
            return ChatSessionUpdateSerializer
        elif self.action == "create_message":
            return ChatMessageCreateSerializer
        elif self.action == "retrieve":
            return ChatSessionDetailSerializer
        elif self.action == "archive_session":
            return ChatSessionArchiveSerializer
        else:
            return ChatSessionDetailSerializer

    def get_queryset(self):
        return (
            ChatSession.objects.filter(
                user=self.request.user,
                status__in=[ChatSession.Status.ACTIVE, ChatSession.Status.ARCHIVED],
            )
            .prefetch_related("messages")
            .order_by("-updated_at")
        )

    def perform_create(self, serializer):
        """Set the user when creating a new chat session"""
        serializer.save(user=self.request.user)

    @action(detail=True, methods=["post"], url_path="messages")
    def create_message(self, request, pk=None):
        session = self.get_object()

        create_serializer = ChatMessageCreateSerializer(data=request.data)
        create_serializer.is_valid(raise_exception=True)

        validated_data = cast(Dict[str, Any], create_serializer.validated_data)
        user_input = validated_data["user_input"]
        ai_answer = validated_data["ai_answer"]
        mode = validated_data.get("mode")

        if mode and session.mode != mode:
            session.mode = mode
            session.save(update_fields=["mode"])

        with transaction.atomic():
            # Generate title from first message if session doesn't have a title
            if not session.title:
                session.title = session.generate_title_from_message(user_input)
                session.save()

            message = ChatMessage.objects.create(
                session=session,
                user_input=user_input,
                status=ChatMessage.MessageStatus.PENDING,
            )
            session.increment_total_messages()
            return self._process_message_sync(message, session, ai_answer)

    @action(detail=True, methods=["post"], url_path="archive")
    def archive_session(self, request, pk=None):
        """Archive a chat session"""
        session = self.get_object()
        session.archive()
        return Response(
            {"message": "Session archived successfully"}, status=status.HTTP_200_OK
        )

    def _process_message_sync(
        self, message: ChatMessage, session: ChatSession, ai_answer: bool = False
    ):
        try:
            message.mark_processing()
            start_time = time.time()

            agent = self._create_ai_agent(ai_answer, mode=session.mode)

            llm_response = ""
            tool_call = None

            pool = ConnectionPool(
                conninfo=DB_URI,
                max_size=DB_CONFIG.get("MAX_CONNECTIONS", 20),
                kwargs={
                    "autocommit": True,
                    "prepare_threshold": 0,
                },
            )

            checkpointer = PostgresSaver(pool)  # type: ignore
            checkpointer.setup()

            def stream():
                nonlocal llm_response, tool_call
                visualization_data = None
                thinking_sent = False
                reasoning_sent = False

                try:
                    yield self._format_stream_message(
                        "thinking", {"message": "Thinking..."}
                    )
                    thinking_sent = True

                    response_stream = agent.process_message(
                        user_input=message.user_input,
                        user_id=session.user.id,
                        session_id=session.id,
                        stream=True,
                        checkpointer=checkpointer,
                    )

                    for part in response_stream:
                        try:
                            seq = list(part)
                            if not seq:
                                continue

                            msg = seq[0]

                            # Debug logging to see what message types we're getting
                            logger.debug(f"Processing message type: {type(msg)}")

                            # Skip ALL LangChain Message objects except the ones we can handle
                            if isinstance(msg, BaseMessage):
                                if isinstance(msg, (ToolMessage, AIMessageChunk)):
                                    # These are the only message types we process
                                    pass
                                else:
                                    # Skip all other message types (HumanMessage, AIMessage, etc.)
                                    logger.debug(
                                        f"Skipping {msg.__class__.__name__} in stream"
                                    )
                                    continue

                        except Exception as stream_e:
                            logger.warning(f"Error processing stream part: {stream_e}")
                            continue

                        if isinstance(msg, ToolMessage):
                            try:
                                # Safely extract content from ToolMessage
                                content = msg.content
                                if hasattr(content, "content"):
                                    content = str(content.content)
                                else:
                                    content = str(content) if content else ""

                                tool_call = json.loads(content)

                                try:
                                    yield self._format_stream_message(
                                        "querying_db",
                                        {"message": "Tool results received."},
                                    )
                                except Exception:
                                    pass

                                # Check if this is a visualization analysis tool response
                                if (
                                    isinstance(tool_call, dict)
                                    and "recommendation" in tool_call
                                ):
                                    visualization_data = tool_call
                                    try:
                                        yield self._format_stream_message(
                                            "visualization", tool_call
                                        )
                                    except Exception as viz_e:
                                        logger.warning(
                                            f"Error formatting visualization message: {viz_e}"
                                        )
                                        yield self._format_stream_message(
                                            "visualization",
                                            {"error": "Visualization formatting error"},
                                        )
                                else:
                                    try:
                                        yield self._format_stream_message(
                                            "tool_response", tool_call
                                        )
                                        # Stream a visualization as soon as tool data is available
                                        if visualization_data is None and isinstance(
                                            tool_call, dict
                                        ):
                                            try:
                                                viz_tool = get_tool_by_name(
                                                    "visualization_analysis"
                                                )
                                                if viz_tool is not None:
                                                    data_summary = None
                                                    if tool_call.get("analysis_type"):
                                                        data_summary = f"tool:{tool_call.get('analysis_type')}"
                                                    elif tool_call.get("total_count") is not None:
                                                        data_summary = f"total_count:{tool_call.get('total_count')}"

                                                    auto_viz = viz_tool._run(
                                                        query=message.user_input,
                                                        data_summary=data_summary,
                                                    )
                                                    if (
                                                        isinstance(auto_viz, dict)
                                                        and auto_viz.get("recommendation")
                                                    ):
                                                        visualization_data = auto_viz
                                                        yield self._format_stream_message(
                                                            "visualization", auto_viz
                                                        )
                                            except Exception as auto_viz_e:
                                                logger.warning(
                                                    f"Streaming visualization failed: {auto_viz_e}"
                                                )
                                    except Exception as tool_e:
                                        logger.warning(
                                            f"Error formatting tool response: {tool_e}"
                                        )
                                        yield self._format_stream_message(
                                            "tool_response",
                                            {"error": "Tool response formatting error"},
                                        )
                            except json.JSONDecodeError:
                                # Ensure we only store JSON-serializable data
                                try:
                                    content = str(msg.content) if msg.content else ""
                                except Exception:
                                    content = "Error parsing tool response"

                                tool_call = (
                                    content if "error" not in content.lower() else []
                                )
                                try:
                                    yield self._format_stream_message(
                                        "tool_response", tool_call
                                    )
                                except Exception as fallback_e:
                                    logger.warning(
                                        f"Error in fallback tool response: {fallback_e}"
                                    )
                                    yield self._format_stream_message(
                                        "tool_response", "Tool response error"
                                    )

                        elif isinstance(msg, AIMessageChunk):
                            if msg.tool_calls:
                                try:
                                    tool_names = []
                                    for tc in msg.tool_calls or []:
                                        name = tc.get("name") if isinstance(tc, dict) else None
                                        if name:
                                            tool_names.append(name)
                                    yield self._format_stream_message(
                                        "querying_db",
                                        {"tools": tool_names} if tool_names else None,
                                    )
                                except Exception as tc_e:
                                    logger.warning(
                                        f"Error formatting tool_call message: {tc_e}"
                                    )
                            elif msg.content:
                                try:
                                    content = str(msg.content)
                                    if not reasoning_sent:
                                        try:
                                            yield self._format_stream_message(
                                                "reasoning",
                                                {"message": "Summarizing results..."},
                                            )
                                            reasoning_sent = True
                                        except Exception:
                                            pass
                                    llm_response += content
                                    yield self._format_stream_message("llm", content)
                                except Exception as llm_e:
                                    logger.warning(
                                        f"Error formatting LLM message: {llm_e}"
                                    )
                                    # Continue without yielding if there's an error

                except Exception as stream_error:
                    # If there's an error in streaming, log it but continue to save
                    error_str = str(stream_error)
                    logger.warning(f"Error in streaming process: {error_str}")

                    # If it's a HumanMessage serialization error, don't treat it as a failure
                    if "HumanMessage" in error_str and "JSON serializable" in error_str:
                        logger.info(
                            "HumanMessage serialization error detected - treating as successful completion"
                        )
                    else:
                        logger.error(f"Genuine streaming error: {error_str}")

                finally:
                    # Always save the message data, even if streaming failed
                    try:
                        processing_time = int((time.time() - start_time) * 1000)

                        # Determine if we should mark as completed or failed
                        should_mark_completed = True
                        response_to_save = llm_response

                        if not llm_response or llm_response.strip() == "":
                            # No response was generated
                            response_to_save = "I encountered an issue processing your request. Please try again."
                            should_mark_completed = False

                        # If no visualization was produced by tools, auto-recommend one
                        try:
                            if not visualization_data:
                                viz_tool = get_tool_by_name("visualization_analysis")
                                if viz_tool is not None:
                                    auto_viz = viz_tool._run(
                                        query=message.user_input,
                                        data_summary=(
                                            f"tool:{tool_call.get('analysis_type')}"
                                            if isinstance(tool_call, dict)
                                            and tool_call.get("analysis_type")
                                            else None
                                        ),
                                    )
                                    # Only attach if it looks valid
                                    if isinstance(auto_viz, dict) and auto_viz.get(
                                        "recommendation"
                                    ):
                                        visualization_data = auto_viz
                        except Exception as auto_viz_e:
                            logger.warning(
                                f"Auto visualization recommendation failed: {auto_viz_e}"
                            )

                        message.assistant_response = response_to_save

                        # Sanitize data before storing in database
                        message.tool_called = (
                            self._sanitize_data(tool_call) if tool_call else None
                        )
                        message.visulization = (
                            self._sanitize_data(visualization_data)
                            if visualization_data
                            else None
                        )

                        # Mark as completed if we have a response, even if there were serialization issues
                        message.status = (
                            ChatMessage.MessageStatus.COMPLETED
                            if should_mark_completed
                            else ChatMessage.MessageStatus.FAILED
                        )
                        message.save()

                        logger.info(
                            f"Message {message.id} saved as {message.status} with response length: {len(message.assistant_response)}"
                        )

                        # Send completion message with maximum safety
                        try:
                            # Create a completely safe completion message
                            completion_data = {
                                "message_id": str(message.id),
                                "processing_time_ms": processing_time,
                                "status": message.status,
                            }

                            # Only add visualization if it exists and is safe
                            if visualization_data:
                                try:
                                    safe_viz_data = self._sanitize_data(
                                        visualization_data
                                    )
                                    # Test serialization before adding
                                    json.dumps(safe_viz_data)
                                    completion_data["visualization"] = safe_viz_data
                                except Exception as viz_test_e:
                                    logger.warning(
                                        f"Visualization data not serializable, skipping: {viz_test_e}"
                                    )

                            yield self._format_stream_message(
                                "completed", completion_data
                            )

                        except Exception as comp_e:
                            logger.warning(
                                f"Error formatting completion message: {comp_e}"
                            )
                            # Ultra-safe fallback - direct JSON
                            try:
                                safe_completion = {
                                    "action": "completed",
                                    "data": {
                                        "message_id": str(message.id),
                                        "status": "completed",
                                    },
                                }
                                yield json.dumps(safe_completion).encode(
                                    "utf-8"
                                ) + b"\n"
                            except Exception:
                                # Absolute final fallback
                                yield b'{"action": "completed", "data": {"status": "completed"}}\n'

                    except Exception as save_error:
                        # If saving fails, mark as failed
                        logger.error(
                            f"Failed to save message {message.id}: {save_error}"
                        )
                        try:
                            message.status = ChatMessage.MessageStatus.FAILED
                            message.assistant_response = "I encountered an error processing your request. Please try again."
                            message.save()
                            yield json.dumps(
                                {
                                    "action": "error",
                                    "data": {"message": "Failed to process message"},
                                }
                            ).encode("utf-8") + b"\n"
                        except Exception:
                            logger.error(
                                f"Failed to mark message {message.id} as failed"
                            )

            return StreamingHttpResponse(
                stream(),
                content_type="application/json",
                headers={"Cache-Control": "no-cache"},
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize AI processing for message {message.id}: {str(e)}"
            )
            return Response(
                {"error": "Failed to process message", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _create_ai_agent(self, ai_answer: bool = False, mode: str = "analysis"):
        llm = ChatOpenAI(
            model=AGENT_CONFIG.get("MODEL", "o4-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        system_prompt = SYSTEM_TEMPLATE
        try:
            from data_insights.workflows.prompt import ML_SYSTEM_TEMPLATE
            if mode == "ml":
                system_prompt = ML_SYSTEM_TEMPLATE
        except Exception:
            pass

        from data_insights.workflows.tools import get_agent_tools
        tools = get_agent_tools(mode=mode)

        agent = create_data_insights_agent(
            llm=llm,
            system_prompt=system_prompt,
            tools=tools,
            max_retries=AGENT_CONFIG.get("MAX_RETRIES", 3),
            enable_caching=AGENT_CONFIG.get("ENABLE_CACHING", True),
        )

        return agent

    def _sanitize_data(self, data: Any) -> Any:
        """Recursively sanitize data to remove ALL LangChain Message objects and other non-serializable items"""
        if data is None:
            return None

        # Handle all LangChain message types
        if isinstance(data, BaseMessage):
            message_type = data.__class__.__name__
            content = (
                str(data.content) if hasattr(data, "content") and data.content else ""
            )

            # Return a safe dictionary representation
            return {"type": message_type, "content": content}

        # Legacy check for any message-like objects
        if hasattr(data, "__class__") and "Message" in data.__class__.__name__:
            if hasattr(data, "content"):
                return {
                    "type": data.__class__.__name__,
                    "content": str(data.content) if data.content else "",
                }
            else:
                return {"type": data.__class__.__name__, "content": ""}

        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                try:
                    sanitized[key] = self._sanitize_data(value)
                except Exception as e:
                    logger.debug(f"Error sanitizing dict key '{key}': {e}")
                    sanitized[key] = str(value) if value is not None else None
            return sanitized

        if isinstance(data, (list, tuple)):
            sanitized = []
            for i, item in enumerate(data):
                try:
                    sanitized.append(self._sanitize_data(item))
                except Exception as e:
                    logger.debug(f"Error sanitizing list item {i}: {e}")
                    sanitized.append(str(item) if item is not None else None)
            return sanitized

        return data

    def _format_stream_message(self, action: str, data: Any) -> bytes:
        if data is None:
            return json.dumps({"action": action, "data": None}).encode("utf-8") + b"\n"

        try:
            # First, sanitize the data to remove any HumanMessage objects
            sanitized_data = self._sanitize_data(data)

            # Handle specific data processing for certain actions
            if isinstance(sanitized_data, dict):
                if action == "visualization":
                    # Keep visualization data as-is after sanitization
                    pass
                elif "customers" in sanitized_data:
                    sanitized_data = sanitized_data["customers"]
                elif "likely_dormant_accounts" in sanitized_data:
                    sanitized_data = sanitized_data["likely_dormant_accounts"]
                elif "customer_segmentations" in sanitized_data:
                    sanitized_data = sanitized_data["customer_segmentations"]
                elif "dormant_accounts" in sanitized_data:
                    sanitized_data = sanitized_data["dormant_accounts"]
                elif "dormant_reactivation_requests" in sanitized_data:
                    sanitized_data = sanitized_data["dormant_reactivation_requests"]
                elif "success" in sanitized_data:
                    sanitized_data = [sanitized_data]

            # Pre-test serialization to catch any remaining issues
            try:
                json.dumps(sanitized_data)
            except (TypeError, ValueError) as pre_test_e:
                logger.warning(
                    f"Pre-test serialization failed for {action}: {pre_test_e}"
                )
                # Convert to string as ultimate fallback
                sanitized_data = (
                    str(sanitized_data) if sanitized_data is not None else None
                )

            # Try to serialize the sanitized data
            return (
                json.dumps({"action": action, "data": sanitized_data}).encode("utf-8")
                + b"\n"
            )

        except (TypeError, ValueError) as e:
            # If serialization still fails, use a final fallback
            logger.warning(
                f"Failed to serialize data for action {action} even after sanitization: {e}"
            )

            # Check if this is a HumanMessage error and just skip it
            if "HumanMessage" in str(e):
                logger.info(
                    f"Skipping {action} due to HumanMessage serialization issue"
                )
                return b""  # Return empty bytes to skip this message

            try:
                # Final attempt with string conversion
                safe_data = str(data) if data is not None else None
                return (
                    json.dumps({"action": action, "data": safe_data}).encode("utf-8")
                    + b"\n"
                )
            except (TypeError, ValueError):
                # Ultimate fallback - return a safe error message
                return (
                    json.dumps(
                        {"action": action, "data": "Serialization error"}
                    ).encode("utf-8")
                    + b"\n"
                )
