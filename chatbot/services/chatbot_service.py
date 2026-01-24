import logging
import time
from typing import Dict, Any, Optional, List
from django.conf import settings

from .intent_classifier import IntentClassifier, get_question_routing
from .rag_service import RAGService
from .dataset_service import DatasetService

logger = logging.getLogger(__name__)


class ChatbotService:
    """
    Main chatbot service that intelligently routes questions to appropriate handlers.

    Hybrid Architecture:
    - NUMERIC questions → DatasetService (SQL/Database queries)
    - EXPLANATORY questions → RAGService (Document analysis)
    - MIXED questions → Combined approach (SQL + RAG)

    Features:
    - Conversation context awareness
    - Session state management
    - Personalized responses
    - Intelligent follow-up suggestions
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.rag_service = None  # Initialize lazily to avoid import issues
        self.dataset_service = None  # Initialize lazily
        self.conversation_memory = {}  # Store conversation state per session

    def process_question(
        self, question: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user questions with enhanced context awareness.

        Args:
            question: User's natural language question
            context: Additional context (chat_history, user_info, session_id, etc.)

        Returns:
            {
                'answer': str,
                'intent': str,
                'confidence': float,
                'sources': list,
                'processing_time': float,
                'method_used': str,
                'conversation_context': dict,
                'follow_up_suggestions': list
            }
        """
        start_time = time.time()
        session_id = context.get("session_id", "default") if context else "default"

        try:
            # Enrich context with conversation history and session state
            enriched_context = self._enrich_context(question, context or {})

            # Get conversation memory for this session
            conversation_memory = self._get_conversation_memory(session_id)

            # Classify the question intent with context awareness
            routing_info = self._classify_with_context(question, enriched_context)
            intent = routing_info["intent"]

            logger.info(
                f"Question classified as {intent}: {question} (Session: {session_id})"
            )

            # Route to appropriate handler with enhanced context
            if intent == "NUMERIC":
                result = self._handle_numeric_question(question, enriched_context)
                method = "database_query"
            elif intent == "EXPLANATORY":
                result = self._handle_explanatory_question(question, enriched_context)
                method = "document_analysis"
            elif intent == "MIXED":
                result = self._handle_mixed_question(question, enriched_context)
                method = "hybrid_analysis"
            else:
                # Default to explanatory
                result = self._handle_explanatory_question(question, enriched_context)
                method = "document_analysis"

            # Enhance response with conversation context
            enhanced_answer = self._enhance_with_context(
                result["answer"], enriched_context, conversation_memory
            )

            # Generate follow-up suggestions
            follow_up_suggestions = self._generate_follow_up_suggestions(
                question, intent, result, conversation_memory
            )

            # Update conversation memory
            self._update_conversation_memory(
                session_id, question, enhanced_answer, intent
            )

            # Add comprehensive metadata
            result.update(
                {
                    "answer": enhanced_answer,
                    "intent": intent,
                    "confidence": routing_info["confidence"],
                    "processing_time": time.time() - start_time,
                    "method_used": method,
                    "routing_reasoning": routing_info["reasoning"],
                    "suggested_tools": routing_info["suggested_tools"],
                    "conversation_context": {
                        "session_id": session_id,
                        "message_count": conversation_memory.get("message_count", 0),
                        "topics_discussed": conversation_memory.get("topics", []),
                        "last_intent": conversation_memory.get("last_intent"),
                    },
                    "follow_up_suggestions": follow_up_suggestions,
                }
            )

            return result

        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question. If this continues, please try starting a new conversation or contact support.",
                "intent": "error",
                "confidence": 0.0,
                "sources": [],
                "processing_time": time.time() - start_time,
                "method_used": "error",
                "conversation_context": {
                    "session_id": session_id,
                    "error_occurred": True,
                },
                "follow_up_suggestions": [
                    "Try rephrasing your question",
                    "Start a new conversation",
                ],
                "error": str(e),
            }

    def _enrich_context(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich the context with additional information for better responses.
        """
        enriched = dict(context)  # Copy original context

        # Add question analysis
        enriched["question_length"] = len(question)
        enriched["question_words"] = len(question.split())

        # Add user information if available
        if "user_id" in context and context["user_id"]:
            enriched["has_user"] = True
            # Could add user preferences, history, etc. here

        # Add session information
        session_id = context.get("session_id", "default")
        enriched["session_id"] = session_id

        # Add conversation flow information
        chat_history = context.get("chat_history", [])
        enriched["conversation_length"] = len(chat_history)
        enriched["is_first_message"] = len(chat_history) == 0

        # Extract recent topics from chat history
        recent_topics = []
        if chat_history:
            # Analyze last few exchanges for context
            recent_messages = (
                chat_history[-3:] if len(chat_history) > 3 else chat_history
            )
            for user_msg, assistant_msg in recent_messages:
                # Simple topic extraction (could be enhanced with NLP)
                words = user_msg.lower().split()
                if any(
                    word in words for word in ["dataset", "data", "recording", "audio"]
                ):
                    recent_topics.append("data_analysis")
                if any(
                    word in words for word in ["quality", "improve", "better", "issue"]
                ):
                    recent_topics.append("quality_improvement")
                if any(word in words for word in ["how", "what", "why", "explain"]):
                    recent_topics.append("explanation")

        enriched["recent_topics"] = list(set(recent_topics))

        return enriched

    def _get_conversation_memory(self, session_id: str) -> Dict[str, Any]:
        """
        Get conversation memory for a session.
        """
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = {
                "message_count": 0,
                "topics": [],
                "last_intent": None,
                "last_question": None,
                "last_answer": None,
                "preferences": {},
                "context_flags": set(),
            }
        return self.conversation_memory[session_id]

    def _classify_with_context(
        self, question: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Classify question intent with context awareness.
        """
        base_routing = get_question_routing(question)

        # Adjust classification based on context
        conversation_length = context.get("conversation_length", 0)
        recent_topics = context.get("recent_topics", [])

        # If this is a follow-up question in a data analysis conversation
        if conversation_length > 2 and "data_analysis" in recent_topics:
            # Boost confidence for numeric questions in data contexts
            if base_routing["intent"] == "NUMERIC":
                base_routing["confidence"] = min(base_routing["confidence"] + 0.1, 1.0)
            # Or for explanatory questions about data
            elif "explain" in question.lower() or "why" in question.lower():
                base_routing["intent"] = "EXPLANATORY"
                base_routing["confidence"] = 0.9

        return base_routing

    def _enhance_with_context(
        self, answer: str, context: Dict[str, Any], memory: Dict[str, Any]
    ) -> str:
        """
        Enhance the answer with contextual information for better user experience.
        """
        enhanced_answer = answer

        # Add personalized touches based on conversation history
        message_count = memory.get("message_count", 0)

        # For first-time users
        if message_count == 0:
            enhanced_answer += "\n\nWelcome! I'm here to help you with your data collection project. Feel free to ask me anything about your datasets, recordings, or analysis!"

        # For returning conversations
        elif message_count > 5:
            enhanced_answer += "\n\nGreat to continue our conversation! I remember we've been discussing your project."

        # Reference recent topics if relevant
        recent_topics = memory.get("topics", [])
        if recent_topics and len(recent_topics) > 0:
            last_topic = recent_topics[-1]
            if "data_analysis" in recent_topics and "dataset" in answer.lower():
                enhanced_answer += (
                    " This builds on our previous discussion about your data analysis."
                )

        # Add helpful context about capabilities
        if "I detected this as a numeric query" in answer:
            enhanced_answer += "\n\n💡 **Tip:** For specific dataset questions, try asking about your recordings, categories, or statistical analysis!"

        return enhanced_answer

    def _generate_follow_up_suggestions(
        self, question: str, intent: str, result: Dict[str, Any], memory: Dict[str, Any]
    ) -> List[str]:
        """
        Generate intelligent follow-up suggestions based on context.
        """
        suggestions = []

        # Base suggestions based on intent
        if intent == "NUMERIC":
            suggestions.extend(
                [
                    "Would you like me to show you the breakdown by categories?",
                    "Want to see statistics about your recordings?",
                    "Should I analyze trends in your data?",
                ]
            )
        elif intent == "EXPLANATORY":
            suggestions.extend(
                [
                    "Would you like more details about this topic?",
                    "Can I help you understand the data better?",
                    "Want to explore related concepts?",
                ]
            )

        # Context-aware suggestions
        recent_topics = memory.get("topics", [])
        if "data_analysis" in recent_topics:
            suggestions.append("Want to dive deeper into your dataset analysis?")
        if "quality_improvement" in recent_topics:
            suggestions.append("Need help with improving your data quality?")

        # Limit to 3 most relevant suggestions
        return suggestions[:3]

    def _update_conversation_memory(
        self, session_id: str, question: str, answer: str, intent: str
    ):
        """
        Update conversation memory with new interaction.
        """
        memory = self._get_conversation_memory(session_id)

        memory["message_count"] += 1
        memory["last_intent"] = intent
        memory["last_question"] = question
        memory["last_answer"] = answer

        # Extract topics from question
        question_lower = question.lower()
        if any(
            word in question_lower
            for word in ["dataset", "data", "recording", "audio", "analysis"]
        ):
            if "data_analysis" not in memory["topics"]:
                memory["topics"].append("data_analysis")
        if any(
            word in question_lower
            for word in ["quality", "improve", "better", "issue", "problem"]
        ):
            if "quality_improvement" not in memory["topics"]:
                memory["topics"].append("quality_improvement")
        if any(word in question_lower for word in ["explain", "why", "how", "what"]):
            if "explanation" not in memory["topics"]:
                memory["topics"].append("explanation")

        # Clean up old topics if too many
        if len(memory["topics"]) > 5:
            memory["topics"] = memory["topics"][-5:]

    def _handle_numeric_question(
        self, question: str, context: Dict = None
    ) -> Dict[str, Any]:
        """
        Handle numeric questions that require counting, filtering, aggregation, etc.
        Routes to DatasetService for database queries.
        """
        if not self.dataset_service:
            self.dataset_service = DatasetService()

        try:
            # Actually call the dataset service to get real data
            result = self.dataset_service.query_dataset(question, context)

            return {
                "answer": result.get(
                    "answer", "I couldn't retrieve the requested data."
                ),
                "sources": result.get("sources", []),
                "data_used": result.get("data_used", {}),
                "data_type": "numeric",
            }
        except Exception as e:
            logger.error(f"Error in numeric query handling: {e}")
            return {
                "answer": f"I encountered an error while processing your numeric query: {str(e)}. Please try again.",
                "sources": [],
                "data_type": "error",
            }

    def _handle_explanatory_question(
        self, question: str, context: Dict = None
    ) -> Dict[str, Any]:
        """
        Handle explanatory questions that require analysis, patterns, or insights.
        Routes to RAGService for document analysis with enhanced context.
        """
        if not self.rag_service:
            try:
                self.rag_service = RAGService()
            except Exception as e:
                logger.error(f"Failed to initialize RAG service: {e}")
                return {
                    "answer": "I apologize, but I'm having trouble accessing the document analysis system. Please try again in a moment, or ask me about your datasets instead!",
                    "sources": [],
                    "data_type": "error",
                }

        # Get chat history and enhance with context
        chat_history = context.get("chat_history", []) if context else []

        # Add contextual information to the question if relevant
        enhanced_question = question
        recent_topics = context.get("recent_topics", [])

        if "data_analysis" in recent_topics and len(chat_history) > 0:
            # Add context from recent conversation
            enhanced_question = (
                f"{question} (Continuing our discussion about data analysis)"
            )

        # Query using RAG with enhanced context
        result = self.rag_service.query(enhanced_question, chat_history=chat_history)

        # Enhance the answer with context awareness
        answer = result.get(
            "answer",
            "I found some relevant information but couldn't generate a complete answer.",
        )

        # Add contextual enhancements
        if context.get("conversation_length", 0) > 2:
            answer += "\n\n💡 Building on our previous conversation, this information should help clarify things further."

        if not result.get("sources"):
            answer += "\n\n📝 If you have specific documents uploaded, I can provide more detailed analysis based on their content."

        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "tokens_used": result.get("tokens_used", 0),
            "data_type": "document_analysis",
            "context_used": bool(chat_history),
        }

    def _handle_mixed_question(
        self, question: str, context: Dict = None
    ) -> Dict[str, Any]:
        """
        Handle mixed questions that need both numeric data and explanatory analysis.
        Combines results from both DatasetService and RAGService.
        """
        # First, try to get numeric data
        numeric_result = self._handle_numeric_question(question, context)

        # Then, get explanatory analysis using the numeric result as context
        enhanced_question = (
            f"{question}. Additional context: {numeric_result.get('answer', '')}"
        )
        explanatory_result = self._handle_explanatory_question(
            enhanced_question, context
        )

        # Combine the results
        combined_answer = (
            f"{numeric_result['answer']}\n\n{explanatory_result['answer']}"
        )

        return {
            "answer": combined_answer,
            "sources": explanatory_result.get("sources", []),
            "tokens_used": explanatory_result.get("tokens_used", 0),
            "data_type": "hybrid_analysis",
            "numeric_part": numeric_result,
            "explanatory_part": explanatory_result,
        }

    def get_available_capabilities(self) -> Dict[str, Any]:
        """
        Get information about what the chatbot can do
        """
        return {
            "document_analysis": {
                "description": "Analyze uploaded documents and answer questions about their content",
                "supported_formats": ["PDF", "DOCX", "TXT", "MD"],
                "method": "RAG (Retrieval-Augmented Generation)",
            },
            "numeric_queries": {
                "description": "Handle counting, filtering, and aggregation questions",
                "method": "Database/SQL queries",
                "status": "Framework ready, needs database integration",
            },
            "mixed_analysis": {
                "description": "Combine specific data with explanatory analysis",
                "method": "Hybrid database + document analysis",
                "status": "Framework ready, needs database integration",
            },
            "intent_classification": {
                "description": "Automatically classify question types for optimal routing",
                "accuracy": "~85%",
                "method": "Keyword-based classification",
            },
        }


# Convenience function for easy use
def process_chatbot_question(
    question: str, context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Quick function to process a chatbot question
    """
    service = ChatbotService()
    return service.process_question(question, context)
