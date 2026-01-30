import logging
import time
from typing import Dict, Any, List
from .intent_classifier import IntentClassifier, ALLOWED_INTENTS, DEFAULT_INTENT
from .rag_service import RAGService
from .dataset_service import DatasetService
from .memory import get_conversation_memory, save_conversation_memory

logger = logging.getLogger(__name__)


def _safe_confidence(value: Any) -> float:
    """Parse confidence from routing (may be str) and clamp to [0, 1]."""
    try:
        c = float(value)
    except (TypeError, ValueError):
        c = 0.8
    return max(0.0, min(c, 1.0))


def _dedupe_sources(sources: List[Any]) -> List[Any]:
    """Deduplicate sources by (title, content snippet), preserving order."""
    if not sources:
        return []
    seen = set()
    out = []
    for s in sources:
        if isinstance(s, dict):
            key = (str(s.get("title", "")), str(s.get("content", ""))[:100])
        else:
            key = (id(s),)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


class ChatbotService:
    """
    Main chatbot service: routes questions by intent, uses RAG and DatasetService,
    and keeps conversation memory for context-aware replies.
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.rag_service = RAGService()
        self.dataset_service = DatasetService()

    def process_question(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        start_time = time.time()
        context = context or {}
        session_id = str(context.get("session_id", "default"))

        try:
            enriched_context = self._enrich_context(question, context)
            conversation_memory = get_conversation_memory(session_id)

            routing_info = self.intent_classifier.classify(
                question,
                {
                    "conversation_length": enriched_context["conversation_length"],
                    "recent_topics": enriched_context["recent_topics"],
                    "last_intent": conversation_memory.get("last_intent"),
                },
            )
            intent = routing_info.get("intent", DEFAULT_INTENT)
            if intent not in ALLOWED_INTENTS:
                intent = DEFAULT_INTENT

            if intent == "NUMERIC":
                result = self._handle_numeric_question(question, enriched_context)
                method = "database_query"
            elif intent == "EXPLANATORY":
                result = self._handle_explanatory_question(question, enriched_context)
                method = "document_analysis"
            else:
                result = self._handle_mixed_question(question, enriched_context)
                method = "hybrid_analysis"

            answer = result.get("answer", "")
            enhanced_answer = self._enhance_with_context(
                answer, enriched_context, conversation_memory
            )
            follow_up_suggestions = self._generate_follow_up_suggestions(
                question, intent, result, conversation_memory
            )

            self._update_conversation_memory(
                conversation_memory, question, enhanced_answer, intent
            )
            save_conversation_memory(session_id, conversation_memory)

            confidence = _safe_confidence(routing_info.get("confidence", 0.8))
            result.update({
                "answer": enhanced_answer,
                "intent": intent,
                "confidence": confidence,
                "processing_time": time.time() - start_time,
                "method_used": method,
                "routing_reasoning": routing_info.get("reasoning", ""),
                "conversation_context": {
                    "session_id": session_id,
                    "message_count": conversation_memory.get("message_count", 0),
                    "topics_discussed": conversation_memory.get("topics", []),
                    "last_intent": conversation_memory.get("last_intent"),
                },
                "follow_up_suggestions": follow_up_suggestions,
            })
            return result

        except Exception as e:
            logger.exception("Chatbot error")
            return {
                "answer": "An error occurred while processing your request. Please try again.",
                "intent": "error",
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "conversation_context": {"session_id": session_id},
                "sources": [],
                "follow_up_suggestions": [],
                "error": str(e),
            }

    def _enrich_context(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Build enriched context from request context (chat_history, etc.)."""
        chat_history = context.get("chat_history") or []
        conversation_length = len(chat_history)
        recent_topics = []
        if chat_history:
            for user_msg, _ in chat_history[-3:]:
                recent_topics.append((user_msg or "")[:80])
        return {
            "conversation_length": conversation_length,
            "recent_topics": recent_topics,
            "chat_history": chat_history,
            "session_id": context.get("session_id"),
        }

    def _handle_numeric_question(self, question: str, enriched_context: Dict[str, Any]) -> Dict[str, Any]:
        """Route to dataset/database queries."""
        ctx = {"chat_history": enriched_context.get("chat_history")}
        return self.dataset_service.query_dataset(question, ctx)

    def _handle_explanatory_question(self, question: str, enriched_context: Dict[str, Any]) -> Dict[str, Any]:
        """Route to RAG over uploaded documents."""
        chat_history = enriched_context.get("chat_history") or []
        return self.rag_service.query(question, chat_history=chat_history)

    def _handle_mixed_question(self, question: str, enriched_context: Dict[str, Any]) -> Dict[str, Any]:
        """Try numeric first; if answer is short or suggests docs, also consider RAG. Default: explanatory."""
        numeric_result = self._handle_numeric_question(question, enriched_context)
        answer = numeric_result.get("answer", "")
        # If numeric already gave a substantial answer, use it; else also try RAG and combine
        if answer and len(answer.strip()) > 100:
            return numeric_result
        rag_result = self._handle_explanatory_question(question, enriched_context)
        rag_answer = rag_result.get("answer", "")
        combined = f"{answer.strip()}\n\n{rag_answer.strip()}".strip() if answer.strip() else rag_answer
        numeric_sources = numeric_result.get("sources", []) or []
        rag_sources = rag_result.get("sources", []) or []
        sources = _dedupe_sources(numeric_sources + rag_sources)
        return {
            "answer": combined or answer or rag_answer,
            "sources": sources,
            "tokens_used": numeric_result.get("tokens_used", 0) + rag_result.get("tokens_used", 0),
        }

    def _enhance_with_context(
        self, answer: str, enriched_context: Dict[str, Any], conversation_memory: Dict[str, Any]
    ) -> str:
        """Optionally add context to the answer; for now return as-is."""
        return answer or ""

    def _generate_follow_up_suggestions(
        self,
        question: str,
        intent: str,
        result: Dict[str, Any],
        conversation_memory: Dict[str, Any],
    ) -> List[str]:
        """Suggest short follow-up questions."""
        suggestions = []
        if intent == "NUMERIC":
            suggestions = [
                "Show me a breakdown by category",
                "What is the average per region?",
                "How many documents do I have?",
            ]
        else:
            suggestions = [
                "Tell me more about the I Hear project",
                "What datasets are available?",
                "Summarize the main points",
            ]
        return suggestions[:3]

    def _update_conversation_memory(
        self, memory: Dict[str, Any], question: str, answer: str, intent: str
    ) -> None:
        """Update in-place: message_count, topics, last_intent, last_question, last_answer."""
        memory["message_count"] = memory.get("message_count", 0) + 1
        memory["last_intent"] = intent
        memory["last_question"] = (question or "")[:500]
        memory["last_answer"] = (answer or "")[:500]
        topics = memory.get("topics", [])
        if intent not in topics:
            topics = (topics + [intent])[-10:]
        memory["topics"] = topics


def process_chatbot_question(question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Convenience: one-off process without keeping a service instance."""
    service = ChatbotService()
    return service.process_question(question, context)
