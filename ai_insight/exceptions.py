"""
Custom exceptions for AI Insight functionality
"""

from rest_framework.views import exception_handler
from rest_framework.response import Response
from rest_framework import status
from loguru import logger


class AIInsightException(Exception):
    default_message = "An error occurred in AI Insight"
    default_code = "ai_insight_error"

    def __init__(self, message=None, code=None, details=None):
        self.message = message or self.default_message
        self.code = code or self.default_code
        self.details = details or {}
        super().__init__(self.message)


class AgentInitializationError(AIInsightException):
    default_message = "Failed to initialize AI agent"
    default_code = "agent_initialization_error"


class MessageProcessingError(AIInsightException):
    default_message = "Failed to process message"
    default_code = "message_processing_error"


class SessionLimitExceededError(AIInsightException):
    default_message = "Session limit exceeded"
    default_code = "session_limit_exceeded"


class InvalidQueryError(AIInsightException):
    default_message = "Query contains invalid content"
    default_code = "invalid_query"


class DatabaseConnectionError(AIInsightException):
    default_message = "Database connection error"
    default_code = "database_connection_error"


class AIModelTimeoutError(AIInsightException):
    default_message = "AI model response timed out"
    default_code = "ai_model_timeout"


def ai_insight_exception_handler(exc, context):
    response = exception_handler(exc, context)

    if isinstance(exc, AIInsightException):
        logger.error(
            f"AI Insight Exception: {exc.code} - {exc.message}",
            extra={"details": exc.details, "context": context},
        )

        custom_response_data = {
            "error": {"code": exc.code, "message": exc.message, "details": exc.details}
        }

        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if isinstance(exc, (SessionLimitExceededError, InvalidQueryError)):
            status_code = status.HTTP_400_BAD_REQUEST
        elif isinstance(exc, DatabaseConnectionError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif isinstance(exc, AIModelTimeoutError):
            status_code = status.HTTP_408_REQUEST_TIMEOUT

        return Response(custom_response_data, status=status_code)

    if response is not None:
        logger.warning(
            f"API Exception: {type(exc).__name__} - {str(exc)}",
            extra={"context": context},
        )

        request = context.get("request")
        if request and hasattr(request, "META"):
            response.data["request_id"] = request.META.get("HTTP_X_REQUEST_ID")  # type: ignore

    return response
