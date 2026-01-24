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

# Services are imported lazily to avoid ChromaDB import issues
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

        # Delete vectors in background
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
        try:
            start_time = time.time()
            from .services import IntentClassifier, RAGService, DatasetService

            # Classify the question intent
            intent_classifier = IntentClassifier()
            routing_info = intent_classifier.get_routing_info(message_text)
            intent = routing_info["intent"]

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

            # Save assistant message
            assistant_message = Message.objects.create(
                session=session,
                role="assistant",
                content=result["answer"],
                sources=result["sources"],
                tokens_used=result.get("tokens_used", 0),
                response_time=response_time,
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
            logger.error(f"Error in send_message: {e}")
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
            try:
                from .services import (
                    ChatbotService,
                    StreamingService,
                )

                chatbot_service = ChatbotService()

                # Process question with full context awareness
                result = chatbot_service.process_question(message_text, context)
                intent = result.get("intent", "EXPLANATORY")

                if intent == "NUMERIC":
                    # For numeric questions, return immediate result (no streaming needed)
                    answer = result.get("answer", "I processed your numeric query.")

                    # Save the assistant message immediately for numeric responses
                    Message.objects.create(
                        session=session,
                        role="assistant",
                        content=answer,
                        sources=result.get("sources", []),
                        tokens_used=result.get("tokens_used", 0),
                        metadata={
                            "intent": result.get("intent"),
                            "method_used": result.get("method_used"),
                            "conversation_context": result.get("conversation_context"),
                            "follow_up_suggestions": result.get(
                                "follow_up_suggestions"
                            ),
                            "processing_time": result.get("processing_time"),
                        },
                    )

                    # Yield a single complete response
                    yield f'data: {{"type": "start"}}\n\n'
                    yield f"data: {{\"type\": \"token\", \"content\": \"{answer.replace(chr(10), '\\\\n').replace('\"', '\\\\\"')}\"}}\n\n"
                    yield f"data: {{\"type\": \"source\", \"sources\": {result.get('sources', [])}, \"intent\": \"{intent}\", \"method\": \"{result.get('method_used', 'database')}\", \"follow_up_suggestions\": {result.get('follow_up_suggestions', [])}}}\n\n"
                    yield f"data: {{\"type\": \"end\", \"tokens_used\": {result.get('tokens_used', 0)}, \"response_time\": {result.get('processing_time', 0):.2f}, \"conversation_context\": {result.get('conversation_context', {})}}}\n\n"
                    return

                else:
                    # For explanatory questions, use streaming RAG with context
                    rag_service = chatbot_service.rag_service
                    streaming_service = StreamingService()

                    # Stream the RAG response with enhanced context
                    accumulated_content = ""
                    accumulated_sources = []
                    tokens_used = 0

                    for sse_event in streaming_service.stream_response(
                        rag_service, message_text, chat_history
                    ):
                        yield sse_event

                        # Parse the event to accumulate data
                        if '"type": "token"' in sse_event:
                            import json

                            try:
                                data_line = [
                                    line
                                    for line in sse_event.split("\n")
                                    if line.startswith("data:")
                                ][0]
                                data = json.loads(data_line.replace("data: ", ""))
                                accumulated_content += data.get("content", "")
                            except:
                                pass

                        elif '"type": "source"' in sse_event:
                            import json

                            try:
                                data_line = [
                                    line
                                    for line in sse_event.split("\n")
                                    if line.startswith("data:")
                                ][0]
                                data = json.loads(data_line.replace("data: ", ""))
                                accumulated_sources.append(data)
                            except:
                                pass

                        elif '"type": "complete"' in sse_event:
                            import json

                            try:
                                data_line = [
                                    line
                                    for line in sse_event.split("\n")
                                    if line.startswith("data:")
                                ][0]
                                data = json.loads(data_line.replace("data: ", ""))
                                tokens_used = data.get("tokens_used", 0)
                            except:
                                pass

                    # Save assistant message after streaming completes with enhanced context
                    if accumulated_content:
                        # Get the full result from chatbot service for context
                        full_result = chatbot_service.process_question(
                            message_text, context
                        )

                        Message.objects.create(
                            session=session,
                            role="assistant",
                            content=accumulated_content,
                            sources=accumulated_sources,
                            tokens_used=tokens_used,
                            metadata={
                                "intent": full_result.get("intent"),
                                "method_used": full_result.get("method_used"),
                                "conversation_context": full_result.get(
                                    "conversation_context"
                                ),
                                "follow_up_suggestions": full_result.get(
                                    "follow_up_suggestions"
                                ),
                                "processing_time": full_result.get("processing_time"),
                            },
                        )

                        # Update session timestamp
                        session.save()

            except Exception as e:
                logger.error(f"Error in async streaming: {e}")
                from .services import StreamingService

                streaming_service = StreamingService()
                yield streaming_service.format_sse(
                    {"type": "error", "message": str(e)}, event="error"
                )

        # Return ASGI-compatible streaming response
        from django.http import StreamingHttpResponse

        response = StreamingHttpResponse(
            stream_generator(), content_type="text/event-stream"
        )
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"
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
