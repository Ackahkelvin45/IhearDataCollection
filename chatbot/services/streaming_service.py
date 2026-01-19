import json
import time
from typing import Generator, Dict, Any
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
        Stream RAG response as SSE
        
        Args:
            rag_service: RAGService instance
            question: User's question
            chat_history: Conversation history
            
        Yields:
            SSE formatted strings
        """
        try:
            # Send start event
            yield self.format_sse({"type": "start"}, event="stream_start")

            # Get streaming generator from RAG service
            accumulated_response = ""
            sources = []
            
            for chunk in rag_service.query_stream(question, chat_history):
                if chunk["type"] == "token":
                    # Stream token
                    accumulated_response += chunk["content"]
                    yield self.format_sse(chunk, event="token")
                    
                elif chunk["type"] == "source":
                    # Accumulate sources
                    sources.append(chunk)
                    
                elif chunk["type"] == "complete":
                    # Send sources
                    for source in sources:
                        yield self.format_sse(source, event="source")
                    
                    # Send completion
                    yield self.format_sse(
                        {
                            "type": "complete",
                            "full_response": accumulated_response,
                            "tokens_used": chunk.get("tokens_used", 0),
                        },
                        event="stream_complete",
                    )
                    
                elif chunk["type"] == "error":
                    yield self.format_sse(chunk, event="error")
                    break

        except Exception as e:
            logger.error(f"Error in stream_response: {e}")
            yield self.format_sse(
                {"type": "error", "message": str(e)}, event="error"
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
            yield self.format_sse(
                {"type": "error", "message": str(e)}, event="error"
            )

