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
        Optimized for speed - minimal string operations

        Args:
            data: Data to send
            event: Event type

        Returns:
            Formatted SSE string
        """
        # Use faster JSON encoding with separators for compact output
        json_data = json.dumps(data, separators=(',', ':'))
        return f"event: {event}\ndata: {json_data}\n\n"

    def stream_events(
        self, rag_service, question: str, chat_history: list = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream RAG response as structured events (dicts).
        Caller formats to SSE at the HTTP boundary—no parse/format round-trip.

        Args:
            rag_service: RAGService instance
            question: User's question
            chat_history: Conversation history

        Yields:
            Dict events: {"type": "start"}, {"type": "token", "content": "..."}, etc.
        """
        try:
            for chunk in rag_service.query_stream(question, chat_history):
                chunk_type = chunk.get("type")
                if chunk_type in ["start", "token", "source", "complete", "error"]:
                    yield chunk
        except Exception as e:
            logger.error(f"Error in stream_events: {e}")
            yield {"type": "error", "message": str(e)}

    def stream_response(
        self, rag_service, question: str, chat_history: list = None
    ) -> Generator[str, None, None]:
        """
        Stream RAG response as SSE strings (legacy). Prefer stream_events + format_sse at edge.
        """
        for event in self.stream_events(rag_service, question, chat_history):
            event_type = event.get("type", "message")
            yield self.format_sse(event, event=f"stream_{event_type}")

    async def astream_response(
        self, rag_service, question: str, chat_history: list = None
    ) -> AsyncGenerator[str, None]:
        """
        Async stream RAG response as SSE for ASGI compatibility
        Optimized for maximum speed - no artificial delays

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

            # Convert sync generator to async yielding - optimized for speed
            for chunk in rag_service.query_stream(question, chat_history):
                # Yield each chunk immediately for fastest streaming
                if chunk["type"] in ["start", "token", "source", "complete", "error"]:
                    yield self.format_sse(chunk, event=f"stream_{chunk['type']}")

                    # Handle source accumulation for completion
                    if chunk["type"] == "source":
                        sources.append(chunk)

                    # No artificial delay - yield immediately for fastest UX

        except Exception as e:
            logger.error(f"Error in astream_response: {e}")
            yield self.format_sse(
                {"type": "error", "message": str(e)}, event="astream_error"
            )

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
