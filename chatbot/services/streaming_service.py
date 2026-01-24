import json
import time
import asyncio
from typing import Generator, AsyncGenerator, Dict, Any
import logging

logger = logging.getLogger(__name__)


class StreamingService:
    """Service for handling Server-Sent Events (SSE) streaming"""

    def __init__(self):
        self.heartbeat_interval = 15  # seconds

    def format_sse(self, data: Dict[str, Any], event: str = "message") -> str:
        """
        Format data as Server-Sent Event

        Args:
            data: Data to send
            event: Event type

        Returns:
            Formatted SSE string
        """
        msg = f"event: {event}\n"
        msg += f"data: {json.dumps(data)}\n\n"
        return msg

    def stream_response(
        self, rag_service, question: str, chat_history: list = None
    ) -> Generator[str, None, None]:
        """
        Stream RAG response as SSE with optimized performance

        Args:
            rag_service: RAGService instance
            question: User's question
            chat_history: Conversation history

        Yields:
            SSE formatted strings immediately
        """
        try:
            # Get streaming generator from RAG service and yield immediately
            sources = []

            for chunk in rag_service.query_stream(question, chat_history):
                # Yield each chunk immediately for fastest streaming
                if chunk["type"] in ["start", "token", "source", "complete", "error"]:
                    yield self.format_sse(chunk, event=f"stream_{chunk['type']}")

                    # Handle source accumulation for completion
                    if chunk["type"] == "source":
                        sources.append(chunk)

        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield self.format_sse({"type": "error", "message": str(e)}, event="stream_error")

    async def astream_response(
        self, rag_service, question: str, chat_history: list = None
    ) -> AsyncGenerator[str, None]:
        """
        Async stream RAG response as SSE for ASGI compatibility

        Args:
            rag_service: RAGService instance
            question: User's question
            chat_history: Conversation history

        Yields:
            SSE formatted strings immediately
        """
        try:
            # Get streaming generator from RAG service and yield immediately
            sources = []

            # Convert sync generator to async yielding
            loop = asyncio.get_event_loop()
            for chunk in rag_service.query_stream(question, chat_history):
                # Yield each chunk immediately for fastest streaming
                if chunk["type"] in ["start", "token", "source", "complete", "error"]:
                    yield self.format_sse(chunk, event=f"stream_{chunk['type']}")

                    # Handle source accumulation for completion
                    if chunk["type"] == "source":
                        sources.append(chunk)

                    # Small yield to prevent blocking
                    await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in astream_response: {e}")
            yield self.format_sse({"type": "error", "message": str(e)}, event="astream_error")

    def heartbeat(self) -> str:
        """Generate heartbeat message to keep connection alive"""
        return ": heartbeat\n\n"

    def create_sse_response(self, generator: Generator) -> Generator[str, None, None]:
        """
        Wrap a generator with SSE formatting and heartbeat

        Args:
            generator: Data generator

        Yields:
            SSE formatted data with heartbeats
        """
        last_heartbeat = time.time()

        try:
            for data in generator:
                yield data

                # Send heartbeat if needed
                if time.time() - last_heartbeat > self.heartbeat_interval:
                    yield self.heartbeat()
                    last_heartbeat = time.time()

        except GeneratorExit:
            logger.info("Client disconnected from stream")
        except Exception as e:
            logger.error(f"Error in SSE stream: {e}")
            yield self.format_sse({"type": "error", "message": str(e)}, event="error")
