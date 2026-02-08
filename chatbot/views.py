from rest_framework import viewsets, status
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework.pagination import PageNumberPagination
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import StreamingHttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
import asyncio
import time
import logging
from pathlib import Path

from .models import Document, ChatSession, Message, MessageFeedback
from .serializers import (
    DocumentSerializer,
    ChatSessionSerializer,
    ChatSessionDetailSerializer,
    MessageSerializer,
    MessageFeedbackSerializer,
    SendMessageSerializer,
    CreateSessionSerializer,
    DocumentUploadSerializer,
)

# Services are imported lazily; RAG uses FAISS only (no Chroma)
# from .tasks import process_document_task, delete_document_vectors_task

logger = logging.getLogger(__name__)


class StandardResultsSetPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


class DocumentViewSet(viewsets.ModelViewSet):
    """ViewSet for Document management"""

    serializer_class = DocumentSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        return Document.objects.filter(uploaded_by=self.request.user).order_by(
            "-uploaded_at"
        )

    def create(self, request):
        """Upload a new document"""
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        file = serializer.validated_data["file"]
        title = serializer.validated_data.get("title") or file.name

        ext = Path(file.name).suffix.lower().lstrip(".")

        # Create document
        document = Document.objects.create(
            title=title,
            file=file,
            file_type=ext,
            uploaded_by=request.user,
        )

        # Trigger async processing
        # process_document_task.delay(str(document.id))

        return Response(
            DocumentSerializer(document, context={"request": request}).data,
            status=status.HTTP_201_CREATED,
        )

    def destroy(self, request, pk=None):
        """Delete a document"""
        document = self.get_object()

        # Delete vectors in background (lazy import to avoid startup issues)
        from .tasks import delete_document_vectors_task

        delete_document_vectors_task.delay(str(document.id))

        document.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=True, methods=["get"])
    def status(self, request, pk=None):
        """Get processing status of a document"""
        document = self.get_object()
        serializer = DocumentSerializer(document, context={"request": request})
        return Response(serializer.data)

    @action(detail=True, methods=["post"])
    def reprocess(self, request, pk=None):
        """Reprocess a document"""
        from .tasks import reprocess_document_task

        document = self.get_object()
        reprocess_document_task.delay(str(document.id))

        return Response(
            {"message": "Document queued for reprocessing"},
            status=status.HTTP_202_ACCEPTED,
        )


class ChatSessionViewSet(viewsets.ModelViewSet):
    """ViewSet for ChatSession management"""

    permission_classes = [IsAuthenticated]
    pagination_class = StandardResultsSetPagination

    def get_queryset(self):
        return ChatSession.objects.filter(user=self.request.user).order_by(
            "-updated_at"
        )

    def get_serializer_class(self):
        if self.action == "retrieve":
            return ChatSessionDetailSerializer
        return ChatSessionSerializer

    def create(self, request):
        """Create a new chat session"""
        serializer = CreateSessionSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        title = serializer.validated_data.get("title", "New Chat")
        document_ids = serializer.validated_data.get("document_ids", [])

        # Create session
        session = ChatSession.objects.create(user=request.user, title=title)

        # Add documents if provided
        if document_ids:
            documents = Document.objects.filter(
                id__in=document_ids, uploaded_by=request.user
            )
            session.documents.set(documents)

        return Response(
            ChatSessionSerializer(session).data, status=status.HTTP_201_CREATED
        )

    @action(detail=True, methods=["post"])
    def send_message(self, request, pk=None):
        """Send a message in a session (non-streaming)"""
        session = self.get_object()
        serializer = SendMessageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        message_text = serializer.validated_data["message"]

        # Save user message
        user_message = Message.objects.create(
            session=session, role="user", content=message_text
        )

        # Get chat history
        chat_history = self._get_chat_history(session, exclude_last=True)

        # Intelligent routing based on question intent
        start_time = time.time()
        try:
            from .services import IntentClassifier, RAGService, DatasetService

            # Classify the question intent
            intent_classifier = IntentClassifier()
            routing_info = intent_classifier.get_routing_info(message_text)
            intent = routing_info.get("intent", "EXPLANATORY")

            # Route based on intent
            if intent == "NUMERIC":
                # Handle numeric/database queries
                dataset_service = DatasetService()
                result = dataset_service.query_dataset(message_text, {})
            else:
                # Handle explanatory/document queries with RAG
                rag_service = RAGService()
                result = rag_service.query(message_text, chat_history=chat_history)

            response_time = time.time() - start_time

            # Ensure we have valid result data
            answer = result.get("answer", "I couldn't generate a response. Please try again.")
            tokens_used = result.get("tokens_used", 0) or len(answer.split())
            sources = result.get("sources", [])

            # Save assistant message
            assistant_message = Message.objects.create(
                session=session,
                role="assistant",
                content=answer,
                sources=sources,
                tokens_used=tokens_used,
                response_time=response_time,
                metadata={
                    "intent": intent,
                    "method_used": "database_query" if intent == "NUMERIC" else "rag",
                },
            )

            # Update session timestamp
            session.save()

            return Response(
                {
                    "user_message": MessageSerializer(user_message).data,
                    "assistant_message": MessageSerializer(assistant_message).data,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            logger.error(f"Error in send_message: {e}", exc_info=True)
            response_time = time.time() - start_time
            
            # Save error message as assistant response so it's tracked
            error_message = f"I encountered an error while processing your question: {str(e)}. Please try again."
            try:
                assistant_message = Message.objects.create(
                    session=session,
                    role="assistant",
                    content=error_message,
                    sources=[],
                    tokens_used=len(error_message.split()),
                    response_time=response_time,
                    metadata={
                        "intent": "error",
                        "method_used": "error",
                        "error": str(e),
                    },
                )
                session.save()
                
                return Response(
                    {
                        "user_message": MessageSerializer(user_message).data,
                        "assistant_message": MessageSerializer(assistant_message).data,
                        "error": str(e),
                    },
                    status=status.HTTP_200_OK,  # Return 200 so frontend can display the error message
                )
            except Exception as save_error:
                logger.error(f"Failed to save error message: {save_error}")
                return Response(
                    {"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

    @action(detail=True, methods=["post"])
    def send_message_stream(self, request, pk=None):
        """Send a message with streaming response (SSE) - Async version for ASGI"""
        session = self.get_object()
        serializer = SendMessageSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        message_text = serializer.validated_data["message"]

        # Save user message
        user_message = Message.objects.create(
            session=session, role="user", content=message_text
        )

        # Get chat history
        chat_history = self._get_chat_history(session, exclude_last=True)

        # Build comprehensive context for the chatbot
        context = {
            "session_id": str(session.id),
            "user_id": request.user.id if request.user.is_authenticated else None,
            "chat_history": chat_history,
            "session_created": session.created_at.isoformat(),
            "message_count": session.messages.count(),
            "user_authenticated": request.user.is_authenticated,
        }

        # Add user information if authenticated
        if request.user.is_authenticated:
            context.update(
                {
                    "username": request.user.username,
                    "user_email": request.user.email,
                    "is_staff": request.user.is_staff,
                    "date_joined": request.user.date_joined.isoformat(),
                }
            )

        # Create streaming response
        def stream_generator():
            import json as _json
            start_time = time.time()
            assistant_message_saved = False
            
            try:
                from .services import (
                    ChatbotService,
                    StreamingService,
                )

                chatbot_service = ChatbotService()

                # Process question with full context awareness
                result = chatbot_service.process_question(message_text, context)
                intent = result.get("intent", "EXPLANATORY")
                response_time = time.time() - start_time

                if intent == "NUMERIC":
                    # Only show "Querying database..." for numeric/database queries
                    yield "data: " + _json.dumps({"type": "querying"}) + "\n\n"
                    # For numeric questions, return immediate result (no streaming needed)
                    answer = result.get("answer", "I processed your numeric query.")
                    tokens_used = result.get("tokens_used", 0) or len(answer.split())

                    # Save the assistant message immediately for numeric responses
                    Message.objects.create(
                        session=session,
                        role="assistant",
                        content=answer,
                        sources=result.get("sources", []),
                        tokens_used=tokens_used,
                        response_time=response_time,
                        metadata={
                            "intent": result.get("intent"),
                            "method_used": result.get("method_used"),
                            "conversation_context": result.get("conversation_context"),
                            "follow_up_suggestions": result.get(
                                "follow_up_suggestions"
                            ),
                            "processing_time": result.get("processing_time", response_time),
                        },
                    )
                    assistant_message_saved = True

                    # Yield a single complete response (use json.dumps to avoid escaping issues)
                    import json as _json
                    yield "data: " + _json.dumps({"type": "start"}) + "\n\n"
                    yield "data: " + _json.dumps({"type": "token", "content": answer}) + "\n\n"
                    yield "data: " + _json.dumps({
                        "type": "source",
                        "sources": result.get("sources", []),
                        "intent": intent,
                        "method": result.get("method_used", "database"),
                        "follow_up_suggestions": result.get("follow_up_suggestions", []),
                    }) + "\n\n"
                    yield "data: " + _json.dumps({"type": "complete", "tokens_used": tokens_used, "response_time": response_time}) + "\n\n"
                    
                    # Update session timestamp
                    session.save()
                    return

                else:
                    # For explanatory questions, use streaming RAG with context
                    rag_service = chatbot_service.rag_service
                    streaming_service = StreamingService()

                    # Stream the RAG response with enhanced context
                    # Optimized: accumulate data while streaming without blocking
                    accumulated_content = ""
                    accumulated_sources = []
                    tokens_used = 0
                    stream_start_time = time.time()

                    for sse_event in streaming_service.stream_response(
                        rag_service, message_text, chat_history
                    ):
                        # Yield immediately - don't block on parsing
                        yield sse_event

                        # Parse SSE: format is "event: stream_X\ndata: {...}\n\n" - extract data line
                        data_line = None
                        for line in sse_event.split("\n"):
                            if line.startswith("data:"):
                                data_line = line[5:].strip()
                                break
                        if not data_line:
                            continue
                        try:
                            import json
                            data = json.loads(data_line)
                            event_type = data.get("type")
                            if event_type == "token":
                                accumulated_content += data.get("content", "")
                            elif event_type == "source":
                                if isinstance(data, dict) and "sources" in data:
                                    accumulated_sources = data.get("sources", [])
                                else:
                                    accumulated_sources.append(data)
                            elif event_type == "complete":
                                tokens_used = data.get("tokens_used", 0) or len(accumulated_content.split())
                        except Exception as parse_error:
                            logger.debug(f"SSE parse error (non-critical): {parse_error}")

                    # Calculate response time
                    response_time = time.time() - start_time

                    # Always save assistant message, even if content is empty (to track errors)
                    if not accumulated_content.strip():
                        accumulated_content = "I apologize, but I couldn't generate a response. Please try again or rephrase your question."
                        logger.warning(f"Empty response generated for question: {message_text[:50]}")

                    # We are in the explanatory/streaming RAG branch
                    intent = "EXPLANATORY"

                    # Ensure tokens_used is set
                    if not tokens_used:
                        tokens_used = len(accumulated_content.split())

                    Message.objects.create(
                        session=session,
                        role="assistant",
                        content=accumulated_content,
                        sources=accumulated_sources if isinstance(accumulated_sources, list) else [],
                        tokens_used=tokens_used,
                        response_time=response_time,
                        metadata={
                            "intent": intent,
                            "method_used": "rag_streaming",
                            "streaming": True,
                        },
                    )
                    assistant_message_saved = True

                    # Update session timestamp
                    session.save()

            except Exception as e:
                logger.error(f"Error in async streaming: {e}", exc_info=True)
                response_time = time.time() - start_time
                
                # Save error message as assistant response so it's tracked
                if not assistant_message_saved:
                    error_message = f"I encountered an error while processing your question: {str(e)}. Please try again."
                    try:
                        Message.objects.create(
                            session=session,
                            role="assistant",
                            content=error_message,
                            sources=[],
                            tokens_used=len(error_message.split()),
                            response_time=response_time,
                            metadata={
                                "intent": "error",
                                "method_used": "error",
                                "error": str(e),
                                "streaming": True,
                            },
                        )
                        session.save()
                    except Exception as save_error:
                        logger.error(f"Failed to save error message: {save_error}")
                
                from .services import StreamingService
                streaming_service = StreamingService()
                yield streaming_service.format_sse(
                    {"type": "error", "message": str(e)}, event="error"
                )

        # Wrap sync generator in async iterator so ASGI doesn't warn and block
        _sentinel = object()
        def _feed_queue(sync_gen, q):
            for item in sync_gen:
                q.put_nowait(item)
            q.put_nowait(_sentinel)

        async def async_stream():
            q = asyncio.Queue()
            loop = asyncio.get_event_loop()
            loop.run_in_executor(None, lambda: _feed_queue(stream_generator(), q))
            while True:
                item = await q.get()
                if item is _sentinel:
                    break
                yield item

        response = StreamingHttpResponse(
            async_stream(), content_type="text/event-stream"
        )
        # Critical headers for proper SSE streaming
        response["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response["Pragma"] = "no-cache"
        response["Expires"] = "0"
        response["X-Accel-Buffering"] = "no"  # Disable Nginx buffering
        response["Connection"] = "keep-alive"
        return response

    @action(detail=True, methods=["get"])
    def messages(self, request, pk=None):
        """Get all messages in a session"""
        session = self.get_object()
        messages = session.messages.all().order_by("created_at")

        paginator = StandardResultsSetPagination()
        page = paginator.paginate_queryset(messages, request)

        if page is not None:
            serializer = MessageSerializer(page, many=True)
            return paginator.get_paginated_response(serializer.data)

        serializer = MessageSerializer(messages, many=True)
        return Response(serializer.data)

    def _get_chat_history(self, session, exclude_last=False):
        """Extract chat history as list of tuples"""
        messages = list(session.messages.all().order_by("created_at"))

        if exclude_last and len(messages) > 0:
            messages = messages[:-1]

        chat_history = []
        for i in range(0, len(messages) - 1, 2):
            if i + 1 < len(messages):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                if user_msg.role == "user" and assistant_msg.role == "assistant":
                    chat_history.append((user_msg.content, assistant_msg.content))

        return chat_history


class MessageFeedbackViewSet(viewsets.ModelViewSet):
    """ViewSet for MessageFeedback"""

    serializer_class = MessageFeedbackSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Only allow users to see feedback for their own messages
        return MessageFeedback.objects.filter(
            message__session__user=self.request.user
        ).order_by("-created_at")

    def create(self, request):
        """Create feedback for a message"""
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        # Verify message belongs to user
        message_id = serializer.validated_data["message"].id
        message = get_object_or_404(Message, id=message_id, session__user=request.user)

        # Create or update feedback
        feedback, created = MessageFeedback.objects.update_or_create(
            message=message,
            defaults={
                "rating": serializer.validated_data["rating"],
                "comment": serializer.validated_data.get("comment", ""),
                "metadata": serializer.validated_data.get("metadata", {}),
            },
        )

        return Response(
            MessageFeedbackSerializer(feedback).data,
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def chatbot_stats(request):
    """Get chatbot statistics for the user"""
    user = request.user

    stats = {
        "total_sessions": ChatSession.objects.filter(user=user).count(),
        "active_sessions": ChatSession.objects.filter(
            user=user, is_active=True
        ).count(),
        "total_messages": Message.objects.filter(session__user=user).count(),
        "total_documents": Document.objects.filter(uploaded_by=user).count(),
        "processed_documents": Document.objects.filter(
            uploaded_by=user, processed=True
        ).count(),
    }

    # Get vector store stats
    try:
        from .services import RAGService

        rag_service = RAGService()
        vector_stats = rag_service.get_collection_stats()
        stats["vector_store"] = vector_stats
    except Exception as e:
        logger.error(f"Error getting vector store stats: {e}")
        stats["vector_store"] = {}

    return Response(stats)


@login_required
def chatbot_home(request):
    """Render the chatbot UI"""
    return render(request, "chatbot/home.html")
