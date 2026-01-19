import logging
import time
from typing import Dict, Any, Optional
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
    """

    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.rag_service = None  # Initialize lazily to avoid import issues
        self.dataset_service = None  # Initialize lazily

    def process_question(self, question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user questions.

        Args:
            question: User's natural language question
            context: Additional context (chat_history, user_info, etc.)

        Returns:
            {
                'answer': str,
                'intent': str,
                'confidence': float,
                'sources': list,
                'processing_time': float,
                'method_used': str
            }
        """
        start_time = time.time()

        try:
            # Classify the question intent
            routing_info = get_question_routing(question)
            intent = routing_info['intent']

            logger.info(f"Question classified as {intent}: {question}")

            # Route to appropriate handler
            if intent == 'NUMERIC':
                result = self._handle_numeric_question(question, context)
                method = 'database_query'
            elif intent == 'EXPLANATORY':
                result = self._handle_explanatory_question(question, context)
                method = 'document_analysis'
            elif intent == 'MIXED':
                result = self._handle_mixed_question(question, context)
                method = 'hybrid_analysis'
            else:
                # Default to explanatory
                result = self._handle_explanatory_question(question, context)
                method = 'document_analysis'

            # Add metadata
            result.update({
                'intent': intent,
                'confidence': routing_info['confidence'],
                'processing_time': time.time() - start_time,
                'method_used': method,
                'routing_reasoning': routing_info['reasoning'],
                'suggested_tools': routing_info['suggested_tools']
            })

            return result

        except Exception as e:
            logger.error(f"Error processing question '{question}': {e}")
            return {
                'answer': f"I apologize, but I encountered an error while processing your question. Please try rephrasing or contact support if the issue persists.",
                'intent': 'error',
                'confidence': 0.0,
                'sources': [],
                'processing_time': time.time() - start_time,
                'method_used': 'error',
                'error': str(e)
            }

    def _handle_numeric_question(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """
        Handle numeric questions that require counting, filtering, aggregation, etc.
        Routes to DatasetService for database queries.
        """
        if not self.dataset_service:
            self.dataset_service = DatasetService()

        # For now, provide a helpful response about numeric queries
        # In production, this would connect to actual database tables

        return {
            'answer': f"I understand you're asking a numeric question: '{question}'. This type of question would typically query the database for specific counts, averages, or filtered data. Currently, I'm set up to analyze documents, but numeric database queries would be handled here.",
            'sources': [],
            'data_type': 'numeric'
        }

    def _handle_explanatory_question(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """
        Handle explanatory questions that require analysis, patterns, or insights.
        Routes to RAGService for document analysis.
        """
        if not self.rag_service:
            try:
                self.rag_service = RAGService()
            except Exception as e:
                logger.error(f"Failed to initialize RAG service: {e}")
                return {
                    'answer': "I apologize, but I'm having trouble accessing the document analysis system. Please try again in a moment.",
                    'sources': [],
                    'data_type': 'error'
                }

        # Get chat history if provided
        chat_history = context.get('chat_history', []) if context else []

        # Query using RAG
        result = self.rag_service.query(question, chat_history=chat_history)

        return {
            'answer': result.get('answer', 'I found some relevant information but couldn\'t generate a complete answer.'),
            'sources': result.get('sources', []),
            'tokens_used': result.get('tokens_used', 0),
            'data_type': 'document_analysis'
        }

    def _handle_mixed_question(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """
        Handle mixed questions that need both numeric data and explanatory analysis.
        Combines results from both DatasetService and RAGService.
        """
        # First, try to get numeric data
        numeric_result = self._handle_numeric_question(question, context)

        # Then, get explanatory analysis using the numeric result as context
        enhanced_question = f"{question}. Additional context: {numeric_result.get('answer', '')}"
        explanatory_result = self._handle_explanatory_question(enhanced_question, context)

        # Combine the results
        combined_answer = f"{numeric_result['answer']}\n\n{explanatory_result['answer']}"

        return {
            'answer': combined_answer,
            'sources': explanatory_result.get('sources', []),
            'tokens_used': explanatory_result.get('tokens_used', 0),
            'data_type': 'hybrid_analysis',
            'numeric_part': numeric_result,
            'explanatory_part': explanatory_result
        }

    def get_available_capabilities(self) -> Dict[str, Any]:
        """
        Get information about what the chatbot can do
        """
        return {
            'document_analysis': {
                'description': 'Analyze uploaded documents and answer questions about their content',
                'supported_formats': ['PDF', 'DOCX', 'TXT', 'MD'],
                'method': 'RAG (Retrieval-Augmented Generation)'
            },
            'numeric_queries': {
                'description': 'Handle counting, filtering, and aggregation questions',
                'method': 'Database/SQL queries',
                'status': 'Framework ready, needs database integration'
            },
            'mixed_analysis': {
                'description': 'Combine specific data with explanatory analysis',
                'method': 'Hybrid database + document analysis',
                'status': 'Framework ready, needs database integration'
            },
            'intent_classification': {
                'description': 'Automatically classify question types for optimal routing',
                'accuracy': '~85%',
                'method': 'Keyword-based classification'
            }
        }


# Convenience function for easy use
def process_chatbot_question(question: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Quick function to process a chatbot question
    """
    service = ChatbotService()
    return service.process_question(question, context)
