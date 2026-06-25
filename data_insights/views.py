from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django.db.models import Count, Min, Max
from django.db import transaction
from django.utils import timezone
import os
from data.models import NoiseDataset, NoiseAnalysis
from core.models import Region, Community
from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticated
from rest_framework.filters import SearchFilter, OrderingFilter
from rest_framework.throttling import UserRateThrottle
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

from .models import ChatSession, ChatMessage, Dashboard
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
from typing import Any, Dict, cast, List, Optional
import threading
from django.conf import settings

from data_insights.workflows.agent_workflow import create_data_insights_agent
from data_insights.workflows.chart_builder import auto_detect_axes, select_chart_type
from data_insights.workflows.clarification_gate import get_pending_clarification
from data_insights.workflows.tools import (
    get_tool_by_name,
    allowed_tables,
    _current_user_id,
)
from data_insights.workflows.sql_agent import (
    SQLDatabaseWrapper,
    TextToSQLAgent,
    create_readonly_engine,
)
from data_insights.workflows.prompt import SYSTEM_TEMPLATE
from rest_framework.response import Response


AI_CONFIG = getattr(settings, "AI_INSIGHT", {})
DB_CONFIG = AI_CONFIG.get("DATABASE", {})
AGENT_CONFIG = AI_CONFIG.get("AGENT", {})
SECURITY_CONFIG = AI_CONFIG.get("SECURITY", {})

# Per-LLM-call output cap. Bounds worst-case output cost per request without
# truncating normal answers. Configurable via AI_INSIGHT["AGENT"]["MAX_TOKENS"]
# (env: AI_INSIGHT_MAX_TOKENS), default 2000.
AGENT_MAX_TOKENS = AGENT_CONFIG.get("MAX_TOKENS", 2000)


class AIInsightRateThrottle(UserRateThrottle):
    """Per-user throttle for the AI insight chat/message endpoints.

    Wires the previously-dead AI_INSIGHT["SECURITY"]["RATE_LIMIT_PER_MINUTE"]
    setting (env: AI_INSIGHT_RATE_LIMIT_PER_MINUTE) into DRF so LLM-backed
    endpoints have a real ceiling. The "ai_insight" scope resolves its rate
    from REST_FRAMEWORK["DEFAULT_THROTTLE_RATES"] in settings (default
    30 requests/minute per user — generous enough not to break existing
    clients while bounding LLM spend).
    """

    scope = "ai_insight"


# Built from AI_INSIGHT["DATABASE"] in settings: local DB when USE_SQLITE=true, Docker DB when false
DB_URI = (
    f"postgresql://{DB_CONFIG.get('USER', 'postgres')}:"
    f"{DB_CONFIG.get('PASSWORD', '')}@"
    f"{DB_CONFIG.get('HOST', 'localhost')}:"
    f"{DB_CONFIG.get('PORT', 5432)}/"
    f"{DB_CONFIG.get('NAME', 'iheardatadb')}"
)

_pg_pool: Optional[ConnectionPool] = None
_checkpointer: Optional[PostgresSaver] = None
_checkpointer_lock = threading.Lock()


def _get_checkpointer() -> PostgresSaver:
    global _pg_pool, _checkpointer
    if _checkpointer is None:
        with _checkpointer_lock:
            if _checkpointer is None:
                pool = ConnectionPool(
                    conninfo=DB_URI,
                    max_size=DB_CONFIG.get("MAX_CONNECTIONS", 20),
                    kwargs={"autocommit": True, "prepare_threshold": 0},
                )
                saver = PostgresSaver(pool)
                saver.setup()
                _pg_pool = pool
                _checkpointer = saver
    return _checkpointer


# Compiled LangGraph workflows are immutable once built and safe to share across
# requests. We cache them per mode to avoid rebuilding the state machine on every
# message. The actual conversation state lives in PostgresSaver, keyed by
# thread_id = session_id, so there is no cross-user state leakage.
# Tools are also cached per mode since they are stateless.
_compiled_graphs: dict = {}
_cached_tools: dict = {}


@login_required
def unified_chat(request):
    """Unified chat interface with sessions, suggestions, and chat in one page."""
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
        "data_insights/unified_chat.html",
        {"suggestions": suggestions, "ml_suggestions": ml_suggestions},
    )


class ChatSessionView(ModelViewSet):
    permission_classes = [IsAuthenticated]
    throttle_classes = [AIInsightRateThrottle]
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

    @action(
        detail=True,
        methods=["post"],
        url_path="messages/(?P<message_id>[^/.]+)/clarify",
    )
    def clarify_message(self, request, pk=None, message_id=None):
        """Handle a clarification response and resume the agent graph."""
        session = self.get_object()
        message = ChatMessage.objects.filter(session=session, id=message_id).first()
        if not message:
            return Response(
                {"error": "Message not found"}, status=status.HTTP_404_NOT_FOUND
            )

        answer = request.data.get("answer", "")
        is_custom = request.data.get("is_custom", False)

        if not answer:
            return Response(
                {"error": "answer is required"}, status=status.HTTP_400_BAD_REQUEST
            )

        return self._process_clarification_response(message, session, answer, is_custom)

    def _process_clarification_response(
        self,
        message: ChatMessage,
        session: ChatSession,
        answer: str,
        is_custom: bool,
    ):
        """Sync entry point.  The inner stream generator stays sync while a thin
        async adapter wraps it so Django's ASGI handler sees a native async iterator."""
        try:
            message.mark_processing()
            start_time = time.time()

            agent = self._create_ai_agent(False, mode=session.mode)
            llm_response = ""
            tool_call = None
            _current_user_id.set(session.user.id)

            def stream():
                nonlocal llm_response, tool_call
                reasoning_sent = False
                stream_errored = False
                pending_chart = None
                pending_artifact = None

                try:
                    yield self._format_stream_message(
                        "thinking", {"message": "Thinking..."}
                    )

                    response_stream = agent.process_clarification_response(
                        user_id=session.user.id,
                        session_id=session.id,
                        clarification_answer=answer,
                        clarification_is_custom=is_custom,
                        stream=True,
                        checkpointer=_get_checkpointer(),
                    )

                    for part in response_stream:
                        try:
                            seq = list(part)
                            if not seq:
                                continue
                            msg = seq[0]
                            logger.debug(f"Processing message type: {type(msg)}")
                            if isinstance(msg, BaseMessage):
                                if isinstance(msg, (ToolMessage, AIMessageChunk)):
                                    pass
                                else:
                                    logger.debug(
                                        f"Skipping {msg.__class__.__name__} in stream"
                                    )
                                    continue
                        except Exception as stream_e:
                            logger.warning(f"Error processing stream part: {stream_e}")
                            continue

                        if isinstance(msg, ToolMessage):
                            try:
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

                                try:
                                    yield self._format_stream_message(
                                        "tool_response",
                                        self._strip_sensitive_fields(tool_call),
                                    )
                                except Exception:
                                    pass
                            except json.JSONDecodeError:
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
                                except Exception:
                                    pass

                        elif isinstance(msg, AIMessageChunk):
                            if msg.tool_calls:
                                try:
                                    tool_names = []
                                    for tc in msg.tool_calls or []:
                                        name = (
                                            tc.get("name")
                                            if isinstance(tc, dict)
                                            else None
                                        )
                                        if name:
                                            tool_names.append(name)
                                    yield self._format_stream_message(
                                        "querying_db",
                                        {"tools": tool_names} if tool_names else None,
                                    )
                                except Exception:
                                    pass
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
                                except Exception:
                                    pass

                    # After stream: check confirmation state
                    try:
                        checkpointer = _get_checkpointer()
                        config = {"configurable": {"thread_id": str(session.id)}}
                        final_snapshot = agent._compiled_graph.get_state(config)
                        if final_snapshot and final_snapshot.values:
                            if final_snapshot.values.get("confirmation_needed"):
                                resolved_value = final_snapshot.values.get(
                                    "confirmation_resolved_value"
                                )
                                yield self._format_stream_message(
                                    "confirmation",
                                    {
                                        "message": f"Interpreting that as {resolved_value} — does that look right?",
                                        "resolved_value": resolved_value,
                                        "dimension": final_snapshot.values.get(
                                            "clarification_dimension"
                                        ),
                                    },
                                )
                    except Exception:
                        pass

                except GeneratorExit:
                    try:
                        message.status = ChatMessage.MessageStatus.FAILED
                        message.assistant_response = llm_response or ""
                        message.tool_called = (
                            self._sanitize_data(tool_call) if tool_call else None
                        )
                        message.save()
                    except Exception as ge_save_err:
                        logger.error(
                            f"Failed to save message on disconnect: {ge_save_err}"
                        )
                    return

                except Exception as exc:
                    import traceback

                    error_str = str(exc) or repr(exc)
                    if "HumanMessage" in error_str and "JSON serializable" in error_str:
                        logger.info(
                            "HumanMessage serialization warning after stream completion"
                        )
                    else:
                        logger.error(
                            f"Streaming error for message {message.id}: {error_str}"
                        )
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        stream_errored = True

                # --- Read pending_chart / pending_artifact from final state ---
                try:
                    checkpointer = _get_checkpointer()
                    config = {"configurable": {"thread_id": str(session.id)}}
                    final_snapshot = agent._compiled_graph.get_state(config)
                    if final_snapshot and final_snapshot.values:
                        pending_chart = final_snapshot.values.get("pending_chart")
                        pending_artifact = final_snapshot.values.get("pending_artifact")
                except Exception:
                    pass

                # DB save
                completion_data = None
                save_failed = False
                try:
                    processing_time = int((time.time() - start_time) * 1000)
                    should_mark_completed = (
                        bool(llm_response and llm_response.strip())
                        and not stream_errored
                    )
                    response_to_save = (
                        llm_response
                        if should_mark_completed
                        else "I encountered an issue processing your request. Please try again."
                    )

                    if pending_chart:
                        try:
                            yield self._format_stream_message(
                                "visualization", pending_chart
                            )
                        except Exception:
                            pass

                    if pending_artifact:
                        try:
                            yield self._format_stream_message(
                                "dashboard", pending_artifact
                            )
                        except Exception:
                            pass

                    message.assistant_response = response_to_save
                    message.tool_called = (
                        self._sanitize_data(tool_call) if tool_call else None
                    )
                    message.visualization = (
                        self._sanitize_data(pending_chart) if pending_chart else None
                    )
                    message.status = (
                        ChatMessage.MessageStatus.COMPLETED
                        if should_mark_completed
                        else ChatMessage.MessageStatus.FAILED
                    )
                    message.save()
                    logger.info(f"Message {message.id} saved as {message.status}")

                    completion_data = {
                        "message_id": str(message.id),
                        "processing_time_ms": processing_time,
                        "status": message.status,
                    }
                    if pending_chart:
                        try:
                            safe_viz = self._sanitize_data(pending_chart)
                            json.dumps(safe_viz)
                            completion_data["visualization"] = safe_viz
                        except Exception:
                            pass
                    if pending_artifact:
                        try:
                            safe_art = self._sanitize_data(pending_artifact)
                            json.dumps(safe_art)
                            completion_data["artifact"] = safe_art
                        except Exception:
                            pass

                except Exception as save_error:
                    logger.error(f"Failed to save message {message.id}: {save_error}")
                    save_failed = True
                    try:
                        message.status = ChatMessage.MessageStatus.FAILED
                        message.assistant_response = "I encountered an error processing your request. Please try again."
                        message.save()
                    except Exception:
                        logger.error(f"Failed to mark message {message.id} as failed")

                if stream_errored or save_failed:
                    yield json.dumps(
                        {
                            "action": "error",
                            "data": {"message": "Failed to process message"},
                        }
                    ).encode("utf-8") + b"\n"
                    return

                yield self._format_stream_message("completed", completion_data)

            def _safe_stream(gen):
                try:
                    for chunk in gen:
                        if chunk:
                            yield chunk
                except GeneratorExit:
                    return
                except Exception as stream_exc:
                    logger.error(f"Unhandled streaming error: {stream_exc}")
                    yield json.dumps(
                        {
                            "action": "error",
                            "data": {
                                "message": "Stream encountered an unexpected error"
                            },
                        }
                    ).encode("utf-8") + b"\n"

            response = StreamingHttpResponse(
                _safe_stream(stream()),
                content_type="application/json; charset=utf-8",
            )
            response["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response["Pragma"] = "no-cache"
            response["Expires"] = "0"
            response["X-Accel-Buffering"] = "no"
            return response

        except Exception as e:
            logger.error(
                f"Failed to process clarification for message {message.id}: {e}"
            )
            return Response(
                {"error": "Failed to process clarification", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    def _process_message_sync(
        self, message: ChatMessage, session: ChatSession, ai_answer: bool = False
    ):
        """Returns a StreamingHttpResponse backed by an async iterator.  The inner
        stream generator stays sync (avoiding sync_to_async bridges for every
        ORM call) while Django's ASGI handler sees a native async iterator and
        iterates it without the per-chunk ``sync_to_async`` penalty."""
        try:
            message.mark_processing()
            start_time = time.time()

            agent = self._create_ai_agent(ai_answer, mode=session.mode)

            llm_response = ""
            tool_call = None

            _current_user_id.set(session.user.id)

            def stream():
                nonlocal llm_response, tool_call
                reasoning_sent = False
                stream_errored = False
                clarification_data = None
                pending_chart = None
                pending_artifact = None

                try:
                    yield self._format_stream_message(
                        "thinking", {"message": "Thinking..."}
                    )

                    response_stream = agent.process_message(
                        user_input=message.user_input,
                        user_id=session.user.id,
                        session_id=session.id,
                        stream=True,
                        checkpointer=_get_checkpointer(),
                        mode=session.mode,
                    )

                    for part in response_stream:
                        try:
                            seq = list(part)
                            if not seq:
                                continue

                            msg = seq[0]

                            logger.debug(f"Processing message type: {type(msg)}")

                            if isinstance(msg, BaseMessage):
                                if isinstance(msg, (ToolMessage, AIMessageChunk)):
                                    pass
                                else:
                                    logger.debug(
                                        f"Skipping {msg.__class__.__name__} in stream"
                                    )
                                    continue

                        except Exception as stream_e:
                            logger.warning(f"Error processing stream part: {stream_e}")
                            continue

                        if isinstance(msg, ToolMessage):
                            try:
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

                                try:
                                    yield self._format_stream_message(
                                        "tool_response",
                                        self._strip_sensitive_fields(tool_call),
                                    )
                                except Exception as tool_e:
                                    logger.warning(
                                        f"Error formatting tool response: {tool_e}"
                                    )
                            except json.JSONDecodeError:
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
                                        name = (
                                            tc.get("name")
                                            if isinstance(tc, dict)
                                            else None
                                        )
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

                except GeneratorExit:
                    try:
                        message.status = ChatMessage.MessageStatus.FAILED
                        message.assistant_response = llm_response or ""
                        message.tool_called = (
                            self._sanitize_data(tool_call) if tool_call else None
                        )
                        message.save()
                    except Exception as ge_save_err:
                        logger.error(
                            f"Failed to save message {message.id} on disconnect: {ge_save_err}"
                        )
                    return

                except Exception as exc:
                    import traceback

                    error_str = str(exc) or repr(exc)
                    if "HumanMessage" in error_str and "JSON serializable" in error_str:
                        logger.info(
                            "HumanMessage serialization warning after stream completion"
                        )
                    else:
                        logger.error(
                            f"Streaming error for message {message.id}: {error_str}"
                        )
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        stream_errored = True

                # --- Read pending_chart / pending_artifact / clarification from final state ---
                try:
                    checkpointer = _get_checkpointer()
                    config = {"configurable": {"thread_id": str(session.id)}}
                    final_snapshot = agent._compiled_graph.get_state(config)
                    if final_snapshot and final_snapshot.values:
                        pending_chart = final_snapshot.values.get("pending_chart")
                        pending_artifact = final_snapshot.values.get("pending_artifact")
                        clarification_data = final_snapshot.values.get(
                            "pending_clarification_payload"
                        )
                except Exception:
                    pass

                completion_data = None
                save_failed = False
                try:
                    processing_time = int((time.time() - start_time) * 1000)

                    if clarification_data:
                        message.assistant_response = llm_response or ""
                        message.status = ChatMessage.MessageStatus.CLARIFICATION_PENDING
                        message.save()
                        logger.info(
                            f"Message {message.id} saved as clarification_pending"
                        )

                        try:
                            yield self._format_stream_message(
                                "clarification",
                                {
                                    "question": clarification_data["question"],
                                    "options": clarification_data["options"],
                                    "allow_custom": True,
                                    "custom_placeholder": "e.g. type your answer...",
                                    "dimension": clarification_data["dimension"],
                                },
                            )
                        except Exception as clar_yield_err:
                            logger.warning(
                                f"Error yielding clarification chunk: {clar_yield_err}"
                            )

                        yield self._format_stream_message(
                            "completed",
                            {
                                "message_id": str(message.id),
                                "processing_time_ms": processing_time,
                                "status": "clarification_pending",
                            },
                        )
                        return

                    should_mark_completed = (
                        bool(llm_response and llm_response.strip())
                        and not stream_errored
                    )
                    response_to_save = (
                        llm_response
                        if should_mark_completed
                        else "I encountered an issue processing your request. Please try again."
                    )

                    if pending_chart:
                        try:
                            yield self._format_stream_message(
                                "visualization", pending_chart
                            )
                        except Exception:
                            pass

                    if pending_artifact:
                        try:
                            yield self._format_stream_message(
                                "dashboard", pending_artifact
                            )
                        except Exception:
                            pass

                    message.assistant_response = response_to_save
                    message.tool_called = (
                        self._sanitize_data(tool_call) if tool_call else None
                    )
                    message.visualization = (
                        self._sanitize_data(pending_chart) if pending_chart else None
                    )
                    message.status = (
                        ChatMessage.MessageStatus.COMPLETED
                        if should_mark_completed
                        else ChatMessage.MessageStatus.FAILED
                    )
                    message.save()
                    logger.info(f"Message {message.id} saved as {message.status}")

                    completion_data = {
                        "message_id": str(message.id),
                        "processing_time_ms": processing_time,
                        "status": message.status,
                    }
                    if pending_chart:
                        try:
                            safe_viz = self._sanitize_data(pending_chart)
                            json.dumps(safe_viz)
                            completion_data["visualization"] = safe_viz
                        except Exception:
                            pass
                    if pending_artifact:
                        try:
                            safe_art = self._sanitize_data(pending_artifact)
                            json.dumps(safe_art)
                            completion_data["artifact"] = safe_art
                        except Exception:
                            pass

                except Exception as save_error:
                    logger.error(f"Failed to save message {message.id}: {save_error}")
                    save_failed = True
                    try:
                        message.status = ChatMessage.MessageStatus.FAILED
                        message.assistant_response = "I encountered an error processing your request. Please try again."
                        message.save()
                    except Exception:
                        logger.error(f"Failed to mark message {message.id} as failed")

                if stream_errored or save_failed:
                    yield json.dumps(
                        {
                            "action": "error",
                            "data": {"message": "Failed to process message"},
                        }
                    ).encode("utf-8") + b"\n"
                    return

                yield self._format_stream_message("completed", completion_data)

            def _safe_stream(gen):
                """Wrap the generator so that unexpected exceptions don't crash
                Django's StreamingHttpResponse.  A terminal error event is yielded
                and the generator exits cleanly."""
                try:
                    for chunk in gen:
                        if chunk:
                            yield chunk
                except GeneratorExit:
                    return
                except Exception as stream_exc:
                    logger.error(f"Unhandled streaming error: {stream_exc}")
                    yield json.dumps(
                        {
                            "action": "error",
                            "data": {
                                "message": "Stream encountered an unexpected error"
                            },
                        }
                    ).encode("utf-8") + b"\n"

            response = StreamingHttpResponse(
                _safe_stream(stream()),
                content_type="application/json; charset=utf-8",
            )
            response["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response["Pragma"] = "no-cache"
            response["Expires"] = "0"
            response["X-Accel-Buffering"] = "no"
            return response

        except Exception as e:
            logger.error(
                f"Failed to initialize AI processing for message {message.id}: {str(e)}"
            )
            return Response(
                {"error": "Failed to process message", "details": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(
        detail=True,
        methods=["post"],
        url_path="messages/(?P<message_id>[^/.]+)/paginate",
    )
    def paginate_message(self, request, pk=None, message_id=None):
        session = self.get_object()
        message = ChatMessage.objects.filter(session=session, id=message_id).first()
        if not message:
            return Response(
                {"error": "Message not found"}, status=status.HTTP_404_NOT_FOUND
            )

        tool_data = message.tool_called or {}
        pagination_meta = (
            tool_data.get("pagination") if isinstance(tool_data, dict) else {}
        ) or {}

        try:
            limit = int(request.data.get("limit") or pagination_meta.get("limit") or 20)
        except (TypeError, ValueError):
            limit = 20
        try:
            offset = int(request.data.get("offset") or 0)
        except (TypeError, ValueError):
            offset = 0

        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        result = self._paginate_from_tool_data(tool_data, limit, offset)
        if not result:
            return Response(
                {"error": "Pagination not available for this result"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        return Response(result, status=status.HTTP_200_OK)

    def _paginate_from_tool_data(
        self, tool_data: Any, limit: int, offset: int
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(tool_data, dict):
            return None

        analysis_type = tool_data.get("analysis_type")
        query_kind = tool_data.get("query_kind")
        filter_criteria = tool_data.get("filter_criteria") or tool_data.get(
            "pagination", {}
        ).get("filter_criteria")

        if analysis_type == "recent_datasets" or query_kind == "recent_datasets":
            qs = NoiseDataset.objects.select_related(
                "region",
                "community",
                "category",
                "dataset_type",
            ).order_by("-recording_date", "-created_at")
            total_count = qs.count()
            rows = []
            for item in qs[offset : offset + limit]:
                rows.append(
                    {
                        "name": item.name,
                        "region": item.region.name if item.region else None,
                        "community": item.community.name if item.community else None,
                        "category": item.category.name if item.category else None,
                        "recording_date": item.recording_date,
                        "recording_device": item.recording_device,
                    }
                )
            columns = [
                "name",
                "region",
                "community",
                "category",
                "recording_date",
                "recording_device",
            ]
            return {
                "rows": rows,
                "columns": columns,
                "row_count": len(rows),
                "limit": limit,
                "offset": offset,
                "has_more": total_count > (offset + limit),
                "total_count": total_count,
                "title": "Recent Datasets",
            }

        if (
            analysis_type in {"dataset_search_results", "dataset_search_sample"}
            or query_kind == "dataset_search"
        ):
            qs = NoiseDataset.objects.select_related(
                "region",
                "community",
                "category",
                "time_of_day",
                "class_name",
                "subclass",
                "microphone_type",
                "dataset_type",
                "collector",
            )
            filter_criteria = filter_criteria or {}
            if isinstance(filter_criteria, dict):
                if "name" in filter_criteria:
                    qs = qs.filter(name__icontains=filter_criteria["name"])
                if "region" in filter_criteria:
                    qs = qs.filter(region__name__icontains=filter_criteria["region"])
                if "community" in filter_criteria:
                    qs = qs.filter(
                        community__name__icontains=filter_criteria["community"]
                    )
                if "date_from" in filter_criteria:
                    qs = qs.filter(recording_date__gte=filter_criteria["date_from"])
                if "date_to" in filter_criteria:
                    qs = qs.filter(recording_date__lte=filter_criteria["date_to"])

            total_count = qs.count()
            rows = []
            for dataset in qs[offset : offset + limit]:
                rows.append(
                    {
                        "id": dataset.id,
                        "name": dataset.name,
                        "region": dataset.region.name if dataset.region else None,
                        "community": (
                            dataset.community.name if dataset.community else None
                        ),
                        "category": dataset.category.name if dataset.category else None,
                        "recording_date": dataset.recording_date,
                        "recording_device": dataset.recording_device,
                    }
                )
            columns = [
                "id",
                "name",
                "region",
                "community",
                "category",
                "recording_date",
                "recording_device",
            ]
            return {
                "rows": rows,
                "columns": columns,
                "row_count": len(rows),
                "limit": limit,
                "offset": offset,
                "has_more": total_count > (offset + limit),
                "total_count": total_count,
                "title": "Datasets",
            }

        if query_kind == "sql" and tool_data.get("query_sql"):
            query_sql = tool_data.get("query_sql")
            try:
                # Re-executes agent-generated SQL, so it must use the same
                # strictly read-only engine as the NL->SQL agent (P2-1):
                # writes/DDL rejected, statement_timeout enforced.
                engine = create_readonly_engine(
                    DB_CONFIG.get("USER", "postgres"),
                    DB_CONFIG.get("PASSWORD", ""),
                    DB_CONFIG.get("HOST", "localhost"),
                    int(DB_CONFIG.get("PORT", 5432)),
                    DB_CONFIG.get("NAME", "iheardatadb"),
                )
                db = SQLDatabaseWrapper(
                    engine,
                    include_tables=allowed_tables,
                    sample_rows_in_table_info=0,
                    indexes_in_table_info=False,
                    max_string_length=60,
                    lazy_table_reflection=True,
                    enable_cache=True,
                )
                safe_query = TextToSQLAgent._validate_sql_query(
                    query_sql,
                    db.get_usable_table_names(),
                    max_limit=limit,
                    default_offset=offset,
                )
                rows = db.run(safe_query, include_columns=True)
                columns = list(rows[0].keys()) if rows else []
                return {
                    "rows": rows,
                    "columns": columns,
                    "row_count": len(rows),
                    "limit": limit,
                    "offset": offset,
                    "has_more": len(rows) == limit,
                    "total_count": None,
                    "title": "Results",
                }
            except Exception as e:
                logger.warning(f"Pagination SQL error: {e}")
                return None

        return None

    @action(
        detail=True,
        methods=["post"],
        url_path="messages/(?P<message_id>[^/.]+)/save-dashboard",
    )
    def save_dashboard(self, request, pk=None, message_id=None):
        """Save a message's visualization artifact as a named dashboard."""
        from uuid import uuid4
        from django.utils.text import slugify

        session = self.get_object()
        message = ChatMessage.objects.filter(session=session, id=message_id).first()
        if not message:
            return Response(
                {"error": "Message not found"}, status=status.HTTP_404_NOT_FOUND
            )

        viz = message.visualization
        if not viz or not isinstance(viz, dict):
            return Response(
                {"error": "No visualization to save"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        title = request.data.get("title", "").strip()
        if not title:
            title = f"Dashboard - {timezone.now().strftime('%Y-%m-%d %H:%M')}"

        thumbnail = request.data.get("thumbnail", None)

        # If old single-chart format, wrap into artifact format
        artifact = viz
        if "widgets" not in artifact:
            from .workflows.widget_composer import wrap_as_artifact

            wrapped = wrap_as_artifact(artifact)
            if wrapped:
                artifact = wrapped

        dashboard = Dashboard.objects.create(
            user=request.user,
            session=session,
            message=message,
            title=title,
            slug=slugify(title)[:200] + "-" + str(uuid4())[:8],
            artifact_spec=artifact,
            thumbnail=thumbnail,
        )

        return Response(
            {
                "id": str(dashboard.id),
                "title": dashboard.title,
                "slug": dashboard.slug,
            },
            status=status.HTTP_201_CREATED,
        )

    @action(
        detail=True,
        methods=["post"],
        url_path="messages/(?P<message_id>[^/.]+)/insight",
    )
    def generate_insight(self, request, pk=None, message_id=None):
        """Generate a short natural-language insight about the data in a message."""
        session = self.get_object()
        message = ChatMessage.objects.filter(session=session, id=message_id).first()
        if not message:
            return Response(
                {"error": "Message not found"}, status=status.HTTP_404_NOT_FOUND
            )

        query = request.data.get("query", message.user_input or "")
        summary = request.data.get("summary", "")

        if not summary:
            # Build a lightweight summary from the stored visualization
            viz = message.visualization
            if viz and isinstance(viz, dict):
                if "widgets" in viz:
                    parts = [
                        f"{w.get('title', '')}: {w.get('type', '')}"
                        for w in viz["widgets"]
                    ]
                    summary = "; ".join(parts)
                elif viz.get("visualization_name"):
                    summary = f"{viz.get('visualization_name')}: {viz.get('visualization_type', '')}"
            if not summary:
                summary = "Data results"

        try:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=AGENT_CONFIG.get("MODEL", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                max_tokens=200,
                temperature=0.7,
                streaming=True,
            )
            prompt = (
                "You are a data analyst. The user asked a question and we ran a query.\n"
                f"User question: {query}\n"
                f"What the data shows: {summary}\n\n"
                "Give exactly 2 short, sharp observations about this data. "
                "Flag anything surprising or worth investigating. "
                "Use plain language. No bullet points, no markdown. "
                "Keep each observation to one sentence. Separate with a blank line."
            )
            response = llm.invoke(prompt)
            insight_text = (
                str(response.content).strip() if hasattr(response, "content") else ""
            )
        except Exception as e:
            logger.warning(f"Insight generation failed: {e}")
            return Response(
                {"error": "Could not generate insight"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        return Response({"insight": insight_text}, status=status.HTTP_200_OK)

    def _create_ai_agent(self, ai_answer: bool = False, mode: str = "analysis"):
        from data_insights.workflows.tools import get_agent_tools

        system_prompt = SYSTEM_TEMPLATE
        try:
            from data_insights.workflows.prompt import ML_SYSTEM_TEMPLATE

            if mode == "ml":
                system_prompt = ML_SYSTEM_TEMPLATE
        except Exception:
            pass

        # Tools are unified — all modes get all tools. Mode only controls
        # the system prompt and dashboard finalizer. Tools cached once globally.
        if "all" not in _cached_tools:
            _cached_tools["all"] = get_agent_tools()

        llm = ChatOpenAI(
            model=AGENT_CONFIG.get("MODEL", "o4-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=AGENT_MAX_TOKENS,
            streaming=True,
        )
        # Dashboard summary LLM — non-streaming, same model, used for one
        # structured-output call in finalize_dashboard after the agent loop.
        dashboard_llm = ChatOpenAI(
            model=AGENT_CONFIG.get("MODEL", "o4-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=AGENT_MAX_TOKENS,
            streaming=False,
        )
        agent = create_data_insights_agent(
            llm=llm,
            system_prompt=system_prompt,
            tools=_cached_tools["all"],
            max_retries=AGENT_CONFIG.get("MAX_RETRIES", 3),
            enable_caching=AGENT_CONFIG.get("ENABLE_CACHING", True),
            dashboard_llm=dashboard_llm,
        )

        if "__all__" in _compiled_graphs:
            agent.reuse_compiled_graph(_compiled_graphs["__all__"], _get_checkpointer())
        else:
            _compiled_graphs["__all__"] = agent.compile_workflow(_get_checkpointer())

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

    def _strip_sensitive_fields(self, data: Any) -> Any:
        """Remove sensitive fields (e.g., raw SQL) before streaming to the client."""
        if data is None:
            return None
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if key in {"query_sql"}:
                    continue
                cleaned[key] = self._strip_sensitive_fields(value)
            return cleaned
        if isinstance(data, list):
            return [self._strip_sensitive_fields(item) for item in data]
        return data

    def _summarize_tool_data(self, tool_data: Any) -> Optional[str]:
        try:
            if isinstance(tool_data, dict):
                if tool_data.get("skip_visualization"):
                    return None
                if tool_data.get("analysis_type"):
                    return f"tool:{tool_data.get('analysis_type')}"
                # Try to extract rows via the shared helper to get an accurate summary
                rows, cols = self._extract_rows(tool_data)
                if rows:
                    return f"rows:{len(rows)} cols:{cols}"
                if tool_data.get("total_count") is not None:
                    return f"total_count:{tool_data.get('total_count')}"
                if tool_data.get("message") and tool_data.get("total_count") is None:
                    return None
                return "tool:dict"
            if isinstance(tool_data, list):
                rows, cols = self._extract_rows(tool_data)
                return f"rows:{len(rows)} cols:{cols}" if rows else None
            return None
        except Exception:
            return None

    def _extract_rows(self, tool_data: Any) -> tuple:
        """Return (rows, columns) from any tool output shape.

        Handles all known key conventions so callers don't need to repeat
        the same if/elif ladder.  Datetime values are stringified; nested
        lists/dicts are dropped so every row contains only scalars.
        """
        import datetime

        def _scalar(v: Any) -> Any:
            if isinstance(v, (datetime.date, datetime.datetime)):
                return v.isoformat()
            if isinstance(v, (list, dict)):
                return None  # drop non-scalar cells
            return v

        def _clean(row: Dict[str, Any]) -> Dict[str, Any]:
            return {k: _scalar(v) for k, v in row.items()}

        rows: List[Dict[str, Any]] = []

        if isinstance(tool_data, list):
            rows = [_clean(r) for r in tool_data if isinstance(r, dict)]
        elif isinstance(tool_data, dict):
            # Ordered preference: try list-valued keys first
            for key in (
                "rows",
                "datasets",
                "audio_features",
                "sample_data",
                "results",
                "monthly_trends",
                "daily_trends",
                "regional_breakdown",
                "category_breakdown",
            ):
                val = tool_data.get(key)
                if isinstance(val, list) and val:
                    rows = [_clean(r) for r in val if isinstance(r, dict)]
                    if rows:
                        break

            # distribution_data: {group_name: {count, avg, max, min, decibel_values}}
            if not rows and isinstance(tool_data.get("distribution_data"), dict):
                group_key = "region" if "regions" in tool_data else "category"
                for name, stats in tool_data["distribution_data"].items():
                    if isinstance(stats, dict):
                        rows.append(
                            {
                                group_key: name,
                                "count": stats.get("count"),
                                "avg_db": stats.get("avg"),
                                "max_db": stats.get("max"),
                                "min_db": stats.get("min"),
                            }
                        )

            # overview_statistics: single aggregate dict → one-row table
            if not rows and isinstance(tool_data.get("overview_statistics"), dict):
                rows = [_clean(tool_data["overview_statistics"])]

        # Remove columns whose every value is None (keeps table tidy)
        if rows:
            all_keys = list(rows[0].keys())
            non_empty = [k for k in all_keys if any(r.get(k) is not None for r in rows)]
            columns = non_empty or all_keys
            rows = [{k: r.get(k) for k in columns} for r in rows]
        else:
            columns = []

        return rows, columns

    def _build_table_visualization(
        self, tool_data: Any, title: str = "Results"
    ) -> Optional[Dict[str, Any]]:
        try:
            rows, columns = self._extract_rows(tool_data)

            if not rows or not columns:
                return None

            max_rows = 20
            rows = rows[:max_rows]
            table_rows = []
            for row in rows:
                table_rows.append({c: row.get(c) for c in columns})

            pagination = None
            if isinstance(tool_data, dict):
                pagination = tool_data.get("pagination") or {
                    "limit": tool_data.get("limit"),
                    "offset": tool_data.get("offset"),
                    "has_more": tool_data.get("has_more"),
                    "total_count": tool_data.get("total_count"),
                    "query_kind": tool_data.get("analysis_type")
                    or tool_data.get("query_kind"),
                }

            return {
                "visualization_type": "table",
                "visualization_name": "Table",
                "frontend_data": {
                    "type": "none",
                    "title": title,
                    "table": {
                        "columns": columns,
                        "rows": table_rows,
                        "title": title,
                        "pagination": pagination,
                    },
                    "description": "Table results",
                },
            }
        except Exception:
            return None

    # ---- Column detection helpers (shared by _inject_chart_data and viz tools) ----

    @staticmethod
    def _detect_numeric_columns(
        rows: List[Dict[str, Any]], columns: List[str]
    ) -> List[str]:
        """Return columns whose values are all numeric (int, float, or parseable strings)."""
        numeric: List[str] = []
        for col in columns:
            if all(
                v is None
                or isinstance(v, (int, float))
                or (
                    isinstance(v, str) and v.replace(".", "").replace("-", "").isdigit()
                )
                for r in rows
                if (v := r.get(col)) is not None
            ):
                numeric.append(col)
        return numeric

    @staticmethod
    def _detect_time_columns(columns: List[str]) -> bool:
        """True if any column name suggests a date/time dimension."""
        date_keywords = ("date", "time", "month", "year", "day")
        return any(any(k in c.lower() for k in date_keywords) for c in columns)

    @staticmethod
    def _pick_best_column(columns: List[str], preferred: List[str]) -> str:
        """Return the first column whose name contains a preferred keyword, or the
        first column in the list as fallback."""
        if not columns:
            return ""
        for pref in preferred:
            for col in columns:
                if pref in col.lower():
                    return col
        return columns[0]

    # ------------------------------------------------------------------

    def _inject_chart_data(
        self,
        visualization_data: Optional[Dict[str, Any]],
        tool_data: Any,
        question: str,
    ) -> Optional[Dict[str, Any]]:
        """Populate frontend chart payload (labels, data, chart type) from tool output.

        The method uses _extract_rows to normalise any tool output shape, then picks
        the best label and numeric columns, computes labels/data arrays, and selects
        a chart type.  It does NOT depend on hard-coded column names from specific
        tools — it works on any row-shaped dict."""
        if not visualization_data or not tool_data:
            return visualization_data

        rows, columns = self._extract_rows(tool_data)
        if not rows or not columns:
            return visualization_data

        numeric_keys = self._detect_numeric_columns(rows, columns)

        # Read chart_hint from the tool that produced the data.
        # chart_hint is authoritative — the tool knows its own output shape.
        hint = tool_data.get("chart_hint") if isinstance(tool_data, dict) else None
        hint = hint or {}

        label_key = hint.get("x")
        value_key = hint.get("y")

        # Fallback: use auto_detect_axes (temporal-first priority) when chart_hint
        # is missing or incomplete. This fixes the recording_device bug because
        # recording_date is temporal and gets priority over categorical columns.
        if not label_key or not value_key:
            detected = auto_detect_axes(rows, columns)
            if not label_key:
                label_key = detected.get("x")
            if not value_key:
                value_key = detected.get("y")

        # Last-resort fallback: keyword-based scanning (original behavior,
        # with recording_date added to the label preference list).
        if not value_key:
            value_key = self._pick_best_column(
                numeric_keys,
                [
                    "mean_db",
                    "max_db",
                    "min_db",
                    "avg",
                    "average",
                    "count",
                    "total",
                    "value",
                    "level",
                    "decibel",
                ],
            )
        if not label_key:
            label_keys = [c for c in columns if c not in numeric_keys]
            label_key = self._pick_best_column(
                label_keys or columns,
                [
                    "recording_date",
                    "date",
                    "name",
                    "dataset",
                    "region",
                    "community",
                    "category",
                    "class",
                    "subclass",
                ],
            )

        if not label_key or not value_key:
            return visualization_data

        # Build chart payload
        max_rows = 12
        rows = rows[:max_rows]
        labels = [
            str(r.get(label_key) or f"Row {idx + 1}") for idx, r in enumerate(rows)
        ]
        data = [
            (
                float(r.get(value_key))
                if isinstance(r.get(value_key), (int, float))
                else float(str(r.get(value_key) or "0").replace(",", ""))
            )
            for r in rows
        ]

        frontend_data = visualization_data.get("frontend_data") or {}
        frontend_data.update({"labels": labels, "data": data})
        frontend_data.setdefault("colors", frontend_data.get("colors"))

        # Title
        if not frontend_data.get("title"):
            q = question.lower()
            if any(w in q for w in ("highest", "top", "max")):
                frontend_data["title"] = "Top Results"
            elif any(w in q for w in ("lowest", "min", "bottom")):
                frontend_data["title"] = "Lowest Results"
            else:
                frontend_data["title"] = "Results"

        # Inline table for ranking/extreme-value queries
        if any(
            w in question.lower()
            for w in ("highest", "lowest", "top", "bottom", "max", "min")
        ):
            table_cols = columns[:6]
            frontend_data["table"] = {
                "columns": table_cols,
                "rows": [{c: r.get(c) for c in table_cols} for r in rows],
                "title": frontend_data.get("title", "Results"),
            }
            if len(rows) <= 1:
                frontend_data["type"] = "none"

        # Chart type — delegate to the pure-Python decision tree
        hint_for_type = {"x": label_key, "y": value_key}
        if isinstance(tool_data, dict):
            tool_hint = tool_data.get("chart_hint") or {}
            if tool_hint.get("type"):
                hint_for_type["type"] = tool_hint["type"]
        chart_type = select_chart_type(rows, columns, hint_for_type)
        if chart_type and frontend_data.get("type") != "none":
            visualization_data["visualization_type"] = chart_type
            if visualization_data.get("recommendation"):
                visualization_data["recommendation"]["recommended_chart"] = chart_type
            frontend_data["type"] = chart_type

        visualization_data["frontend_data"] = frontend_data
        return visualization_data

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

            # HumanMessage objects should never reach serialization; log and drop the event.
            if "HumanMessage" in str(e):
                logger.info(f"Dropping {action} event: HumanMessage in data")
                return (
                    json.dumps({"action": action, "data": None}).encode("utf-8") + b"\n"
                )

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
