"""
LangChain tool for fetching live web content at query time.

Works with both:
- chatbot RAG pipeline (used directly by RAGService)
- data_insights agent workflow (added to AGENT_TOOLS list)
"""

import logging
from typing import Optional, Dict, Any, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .web_fetcher import fetch_page_text, DEFAULT_MAX_CHARS

logger = logging.getLogger(__name__)


class WebFetchInput(BaseModel):
    """Input schema for the web fetch tool."""

    url: str = Field(
        description="The full URL to fetch content from (e.g. https://rail.knust.edu.gh/about)"
    )
    query_context: Optional[str] = Field(
        default=None,
        description="What the user is asking about, so the tool can describe the fetched content in context",
    )


class WebFetchTool(BaseTool):
    name: str = "fetch_webpage"
    description: str = (
        "Fetch and extract text content from a live webpage. "
        "Use this when the user asks a question about a website, a URL, or needs "
        "information that is not in the uploaded documents or the database. "
        "Returns the extracted text content (truncated to ~8000 characters). "
        "Important: only use this for URLs the user has explicitly asked about "
        "or URLs that are clearly relevant to the conversation."
    )
    args_schema: Type[BaseModel] = WebFetchInput

    max_chars: int = Field(
        default=DEFAULT_MAX_CHARS, description="Max characters to return from a page"
    )
    timeout: int = Field(default=15, description="HTTP request timeout in seconds")

    def _run(self, url: str, query_context: Optional[str] = None) -> str:
        """Fetch a webpage and return its extracted text content.

        Returns a plain string so the LLM can read it directly as context.
        A JSON wrapper would add an extra LLM call; plain text is consumed
        inline by the model that invoked the tool.
        """
        logger.info(
            f"WebFetchTool fetching URL: {url} (context: {query_context or 'none'})"
        )

        text, error = fetch_page_text(
            url, timeout=self.timeout, max_chars=self.max_chars
        )

        if error:
            logger.warning(f"WebFetchTool error for {url}: {error}")
            return f"[Web fetch error: {error}]"

        if not text:
            return "[Web fetch returned empty content]"

        logger.info(f"WebFetchTool successfully fetched {len(text)} chars from {url}")

        # Return plain text — the LLM reads it as retrieved context
        header = f"Content from {url}:"
        if query_context:
            header += f" (fetched because user asked: {query_context})"
        return f"{header}\n\n{text}"
