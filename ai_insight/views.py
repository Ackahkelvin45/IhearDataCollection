import json
import time
from typing import Any, Dict, cast

from django.conf import settings
from django.db import transaction
from django.http import StreamingHttpResponse
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_openai import ChatOpenAI
from psycopg_pool import ConnectionPool
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.filters import OrderingFilter, SearchFilter
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.viewsets import ModelViewSet
from langgraph.checkpoint.postgres import PostgresSaver
from django_filters.rest_framework import DjangoFilterBackend
from loguru import logger

from ai_insight.models import ChatMessage, ChatSession
from ai_insight.permissions import CanUseAIInsight
from ai_insight.serializers import (
    ChatMessageCreateSerializer,
    ChatMessageListSerializer,
    ChatSessionCreateSerializer,
    ChatSessionDetailSerializer,
    ChatSessionListSerializer,
    ChatSessionUpdateSerializer,
    ChatSessionArchiveSerializer,
)
from ai_insight.workflow.agent_workflow import create_crm_agent
from ai_insight.workflow.prompt import SYSTEM_TEMPLATE


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


class AIInsightPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


class ChatSessionViewSet(ModelViewSet):
    permission_classes = [IsAuthenticated, CanUseAIInsight]
    pagination_class = AIInsightPagination
    filter_backends = [DjangoFilterBackend, SearchFilter, OrderingFilter]
    search_fields = ["title"]
    filterset_fields = ["status"]
    ordering_fields = ["created_at", "updated_at"]
    ordering = ["-created_at"]

    def get_serializer_class(self):
        if self.action == "list":
            return ChatSessionListSerializer
        elif self.action == "create":
            return ChatSessionCreateSerializer
        elif self.action in ["update", "partial_update"]:
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
        return ChatSession.objects.filter(
            user=self.request.user,
            status__in=[ChatSession.Status.ACTIVE, ChatSession.Status.ARCHIVED],
        ).prefetch_related("messages")

    @action(detail=True, methods=["post"], url_path="archive")
    def archive_session(self, request, pk=None):
        session = self.get_object()
        session.archive()
        serializer = ChatSessionArchiveSerializer(session)
        return Response(serializer.data)

    @action(detail=True, methods=["get"], url_path="messages")
    def list_messages(self, request, pk=None):
        session = self.get_object()
        messages = session.messages.all()

        status_filter = request.query_params.get("status")
        if status_filter:
            messages = messages.filter(status=status_filter)

        page = self.paginate_queryset(messages)
        if page is not None:
            serializer = ChatMessageListSerializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = ChatMessageListSerializer(messages, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=["post"], url_path="messages")
    def create_message(self, request, pk=None):
        session = self.get_object()

        create_serializer = ChatMessageCreateSerializer(data=request.data)
        create_serializer.is_valid(raise_exception=True)

        validated_data = cast(Dict[str, Any], create_serializer.validated_data)
        user_input = validated_data["user_input"]
        ai_answer = validated_data["ai_answer"]

        with transaction.atomic():
            message = ChatMessage.objects.create(
                session=session,
                user_input=user_input,
                status=ChatMessage.MessageStatus.PENDING,
            )
            session.increment_message_count()
            return self._process_message_sync(message, session, ai_answer)

    def _process_message_sync(
        self, message: ChatMessage, session: ChatSession, ai_answer: bool = False
    ):
        try:
            message.mark_processing()
            start_time = time.time()

            agent = self._create_ai_agent(ai_answer)

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

                try:
                    response_stream = agent.process_message(
                        user_input=message.user_input,
                        user_id=session.user.id,
                        session_id=session.id,
                        stream=True,
                        checkpointer=checkpointer,
                    )

                    for part in response_stream:
                        seq = list(part)
                        if not seq:
                            continue

                        msg = seq[0]

                        if isinstance(msg, ToolMessage):
                            try:
                                tool_call = json.loads(msg.content)  # type: ignore
                            except json.JSONDecodeError:
                                tool_call = (
                                    msg.content
                                    if "error" not in str(msg.content).lower()
                                    else []
                                )
                            yield self._format_stream_message(
                                "tool_response", tool_call
                            )

                        elif isinstance(msg, AIMessageChunk):
                            if msg.tool_calls:
                                yield self._format_stream_message("tool_call", None)
                            elif msg.content:
                                content = str(msg.content)
                                llm_response += content
                                yield self._format_stream_message("llm", content)

                    processing_time = int((time.time() - start_time) * 1000)
                    message.assistant_response = llm_response
                    message.tool_call = tool_call
                    message.status = ChatMessage.MessageStatus.COMPLETED
                    message.processing_time_ms = processing_time
                    message.save()

                    yield self._format_stream_message(
                        "completed",
                        {
                            "message_id": str(message.id),
                            "processing_time_ms": processing_time,
                        },
                    )

                except Exception as e:
                    logger.error(f"Error processing message {message.id}: {str(e)}")
                    yield self._format_stream_message("error", {"message": str(e)})

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

    def _create_ai_agent(self, ai_answer: bool = False):
        llm = ChatOpenAI(model=AGENT_CONFIG.get("MODEL", "o4-mini"))

        agent = create_crm_agent(
            llm=llm,
            system_prompt=SYSTEM_TEMPLATE,
            max_retries=AGENT_CONFIG.get("MAX_RETRIES", 3),
            enable_caching=AGENT_CONFIG.get("ENABLE_CACHING", True),
        )

        return agent

    def _format_stream_message(self, action: str, data: Any) -> bytes:
        if data is None:
            return json.dumps({"action": action, "data": None}).encode("utf-8") + b"\n"

        if "customers" in data:
            data = data["customers"]
        elif "likely_dormant_accounts" in data:
            data = data["likely_dormant_accounts"]
        elif "customer_segmentations" in data:
            data = data["customer_segmentations"]
        elif "dormant_accounts" in data:
            data = data["dormant_accounts"]
        elif "likely_dormant_accounts" in data:
            data = data["likely_dormant_accounts"]
        elif "dormant_reactivation_requests" in data:
            data = data["dormant_reactivation_requests"]
        elif "success" in data:
            data = [data]
        return json.dumps({"action": action, "data": data}).encode("utf-8") + b"\n"
