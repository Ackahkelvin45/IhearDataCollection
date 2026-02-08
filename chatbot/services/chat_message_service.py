"""
ChatMessageService: unified orchestration for sending messages (streaming and non-streaming).

Single source of truth for:
- Chat history extraction (robust by-role iteration)
- Context building
- Message processing via ChatbotService
- Message persistence
"""

import logging
import time
from typing import Dict, Any, List, Tuple, Generator, Optional

from django.db import transaction

logger = logging.getLogger(__name__)

ERROR_MESSAGE = (
    "I encountered an error while processing your question: {error}. Please try again."
)
EMPTY_RESPONSE_FALLBACK = (
    "I apologize, but I couldn't generate a response. Please try again or rephrase your question."
)


def get_chat_history(session, exclude_last: bool = False) -> List[Tuple[str, str]]:
    """
    Extract chat history as list of (user_content, assistant_content) tuples.
    Iterates by role instead of index pairing—handles failed saves, out-of-order messages.
    """
    messages = list(session.messages.all().order_by("created_at"))

    if exclude_last and messages:
        messages = messages[:-1]

    chat_history = []
    pending_user = None

    for msg in messages:
        if msg.role == "user":
            pending_user = msg.content
        elif msg.role == "assistant" and pending_user is not None:
            chat_history.append((pending_user, msg.content))
            pending_user = None

    return chat_history


def build_context(session, request, chat_history: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Build comprehensive context for the chatbot."""
    context = {
        "session_id": str(session.id),
        "user_id": request.user.id if request.user.is_authenticated else None,
        "chat_history": chat_history,
        "session_created": session.created_at.isoformat(),
        "message_count": session.messages.count(),
        "user_authenticated": request.user.is_authenticated,
    }
    if request.user.is_authenticated:
        context.update({
            "username": request.user.username,
            "user_email": request.user.email,
            "is_staff": request.user.is_staff,
            "date_joined": request.user.date_joined.isoformat(),
        })
    return context


def _approx_tokens(text: str) -> int:
    """Approximate token count (word split). Replace with real tokenizer later if needed."""
    return len((text or "").split())


def _save_error_message(session, error: Exception, response_time: float, streaming: bool = False):
    """Persist error as assistant message so chat history stays consistent."""
    from .models import Message

    error_content = ERROR_MESSAGE.format(error=str(error))
    Message.objects.create(
        session=session,
        role="assistant",
        content=error_content,
        sources=[],
        tokens_used=_approx_tokens(error_content),
        response_time=response_time,
        metadata={"intent": "error", "method_used": "error", "error": str(error), "streaming": streaming},
    )


class ChatMessageService:
    """
    Orchestrates message processing and persistence.
    Both streaming and non-streaming flow through ChatbotService for consistent behavior.
    """

    def process_non_streaming(
        self, session, message_text: str, request
    ) -> Tuple[Any, Any]:
        """
        Process a message (non-streaming). Returns (user_message, assistant_message).
        Wrapped in transaction.atomic() for consistency.
        """
        from .models import Message
        from .chatbot_service import ChatbotService

        with transaction.atomic():
            user_message = Message.objects.create(
                session=session, role="user", content=message_text
            )

            chat_history = get_chat_history(session, exclude_last=True)
            context = build_context(session, request, chat_history)

            start_time = time.time()
            try:
                chatbot_service = ChatbotService()
                result = chatbot_service.process_question(message_text, context)
            except Exception as e:
                logger.error(f"Error in process_non_streaming: {e}", exc_info=True)
                response_time = time.time() - start_time
                _save_error_message(session, e, response_time, streaming=False)
                session.save()
                assistant_message = Message.objects.filter(
                    session=session, role="assistant"
                ).order_by("-created_at").first()
                return user_message, assistant_message

            response_time = time.time() - start_time
            answer = result.get("answer") or "I couldn't generate a response. Please try again."
            tokens_used = result.get("tokens_used", 0) or _approx_tokens(answer)
            sources = result.get("sources", [])

            assistant_message = Message.objects.create(
                session=session,
                role="assistant",
                content=answer,
                sources=sources,
                tokens_used=tokens_used,
                response_time=response_time,
                metadata={
                    "intent": result.get("intent", "EXPLANATORY"),
                    "method_used": result.get("method_used", "rag"),
                    "table": result.get("table"),
                    "pagination": result.get("pagination"),
                },
            )
            session.save()
            return user_message, assistant_message

    def process_streaming(
        self, session, message_text: str, request
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process a message with streaming. Yields structured events (dicts).
        Caller formats to SSE at the HTTP boundary.
        """
        from .models import Message
        from .chatbot_service import ChatbotService
        from .streaming_service import StreamingService

        user_message = Message.objects.create(
            session=session, role="user", content=message_text
        )

        chat_history = get_chat_history(session, exclude_last=True)
        context = build_context(session, request, chat_history)

        start_time = time.time()
        assistant_message_saved = False

        try:
            chatbot_service = ChatbotService()
            result = chatbot_service.process_question(message_text, context)
            intent = result.get("intent", "EXPLANATORY")
            response_time = time.time() - start_time

            if intent == "NUMERIC":
                # Only show "Querying database..." for numeric/database queries
                yield {"type": "querying"}
                answer = result.get("answer") or "I processed your numeric query."
                tokens_used = result.get("tokens_used", 0) or _approx_tokens(answer)
                sources = result.get("sources", [])

                Message.objects.create(
                    session=session,
                    role="assistant",
                    content=answer,
                    sources=sources,
                    tokens_used=tokens_used,
                    response_time=response_time,
                    metadata={
                        "intent": intent,
                        "method_used": result.get("method_used", "database_query"),
                        "conversation_context": result.get("conversation_context"),
                        "follow_up_suggestions": result.get("follow_up_suggestions"),
                        "processing_time": result.get("processing_time", response_time),
                        "table": result.get("table"),
                        "pagination": result.get("pagination"),
                    },
                )
                assistant_message_saved = True
                session.save()

                yield {"type": "start"}
                yield {"type": "token", "content": answer}
                yield {
                    "type": "source",
                    "sources": sources,
                    "intent": intent,
                    "method": result.get("method_used", "database"),
                    "follow_up_suggestions": result.get("follow_up_suggestions", []),
                }
                if result.get("table"):
                    yield {
                        "type": "table",
                        "table": result.get("table"),
                        "pagination": result.get("pagination"),
                    }
                yield {
                    "type": "complete",
                    "tokens_used": tokens_used,
                    "response_time": response_time,
                }
                return

            # EXPLANATORY or mixed: stream RAG response
            streaming_service = StreamingService()
            accumulated_content = ""
            accumulated_sources = []
            tokens_used = 0

            for event in streaming_service.stream_events(
                chatbot_service.rag_service, message_text, chat_history
            ):
                yield event

                if event.get("type") == "token":
                    accumulated_content += event.get("content", "")
                elif event.get("type") == "source":
                    if "sources" in event:
                        accumulated_sources = event.get("sources", [])
                    else:
                        accumulated_sources.append(event)
                elif event.get("type") == "complete":
                    tokens_used = event.get("tokens_used", 0) or _approx_tokens(accumulated_content)

            response_time = time.time() - start_time

            if not accumulated_content.strip():
                accumulated_content = EMPTY_RESPONSE_FALLBACK
                logger.warning(f"Empty response for question: {message_text[:50]}")

            if not tokens_used:
                tokens_used = _approx_tokens(accumulated_content)

            Message.objects.create(
                session=session,
                role="assistant",
                content=accumulated_content,
                sources=accumulated_sources if isinstance(accumulated_sources, list) else [],
                tokens_used=tokens_used,
                response_time=response_time,
                metadata={
                    "intent": "EXPLANATORY",
                    "method_used": "rag_streaming",
                    "streaming": True,
                },
            )
            assistant_message_saved = True
            session.save()

        except Exception as e:
            logger.error(f"Error in process_streaming: {e}", exc_info=True)
            response_time = time.time() - start_time
            if not assistant_message_saved:
                _save_error_message(session, e, response_time, streaming=True)
                session.save()
            yield {"type": "error", "message": str(e)}
