import os
import re
import time
import hashlib
import logging
from typing import List, Dict, Any, Optional, Generator

import faiss
from django.conf import settings
from django.core.cache import cache

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

logger = logging.getLogger(__name__)

# Prompt template as plain string to avoid LangChain PromptTemplate format quirks
RAG_PROMPT_TEMPLATE = """
You are the assistant for The I Hear Project. You help users navigate the dashboard, answer questions about datasets, and explain project features.

You are a helpful and friendly AI assistant for **The I Hear Project** - a data collection and analysis platform focused on noise datasets and audio recordings.

You have access to:
- **Uploaded documents**: Context from documents users have uploaded (provided below).
- **Knowledge Base URLs**: Pre-configured websites that are regularly indexed. Their content appears in the Context section when relevant.
- **Web fetch capability**: You can fetch live content from any website. If the user asks about a specific website or URL, or you need information that isn't in the uploaded documents or knowledge base, use the `fetch_webpage` tool to retrieve live content.

{knowledge_base_info}

Your role:
- Be conversational and helpful for general greetings and casual conversation
- When relevant information is provided in the Context section, use it to give accurate, grounded answers
- When the user asks about a website or URL, use the fetch_webpage tool to get live content
- When no relevant context is available and the question requires external knowledge, explain that you can try fetching the relevant website if the user provides a URL
- Maintain conversation context and remember what was discussed earlier

Context (if available):
{context}

Previous conversation:
{chat_history}

User's question:
{question}

Instructions:
1. **For conversational questions** (greetings like "hi", "how are you", "hello", "thanks", etc.): Respond naturally and warmly, then offer to help with The I Hear Project or their datasets.

2. **If Context contains relevant information**: Use it to provide accurate, grounded answers.

3. **If the user asks about a website or mentions a URL**: Use the fetch_webpage tool to retrieve the live content, then answer based on what you fetched.

4. **If Context is empty or not relevant and no URL is mentioned**: Politely explain that you don't have enough information yet and ask the user to either provide a URL to check or upload relevant documents.

Answer clearly, helpfully, and in a friendly conversational tone.
"""

# URL regex pattern for detecting URLs in user questions
_URL_PATTERN = re.compile(r'https?://[^\s<>"\')\]]+', re.IGNORECASE)

EMBEDDING_DIM = 1536  # text-embedding-3-small

# OpenAI embeddings API allows max 300k tokens per request. Batch to stay safely under.
# ~70 tokens/chunk typical; 200 chunks ≈ 14k tokens per batch.
EMBEDDING_BATCH_SIZE = 200
# Delay between batches (seconds) to avoid 429 rate limits (tokens per minute).
EMBEDDING_BATCH_DELAY_SECONDS = 2.0


class FastRAGService:
    """High-performance RAG service using FAISS (core implementation)."""

    def __init__(self):
        self.config = settings.CHATBOT

        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.config["EMBEDDINGS"]["MODEL"],
            openai_api_key=settings.OPENAI_API_KEY,
        )

        # LLM
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7,
        )

        # Vector store: same path as document processing (Celery) for add_documents
        self.vector_store_path = settings.FAISS_VECTOR_STORE_PATH
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vectorstore = self._load_or_create_vectorstore()
        self._vectorstore_mtime = self._get_vectorstore_mtime()

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["RAG"]["CHUNK_SIZE"],
            chunk_overlap=self.config["RAG"]["CHUNK_OVERLAP"],
        )

        logger.info("FastRAGService initialized")

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------

    def _load_or_create_vectorstore(self) -> FAISS:
        if os.path.exists(self.vector_store_path):
            try:
                return FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                logger.warning("Failed to load FAISS index, rebuilding: %s", e)

        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        # If index failed to load but we have chunks in DB, rebuild from DB
        self._rebuild_vectorstore_from_db()
        return self.vectorstore

    def save_vectorstore(self):
        self.vectorstore.save_local(self.vector_store_path)

    def _rebuild_vectorstore_from_db(self) -> None:
        """
        Rebuild the FAISS index from DocumentChunk table when the on-disk index
        is missing or failed to load. Ensures RAG works when documents exist in DB.
        """
        from django.apps import apps

        DocumentChunk = apps.get_model("chatbot", "DocumentChunk")

        chunks = DocumentChunk.objects.select_related("document").order_by(
            "document_id", "chunk_index"
        )
        if not chunks.exists():
            logger.info("No DocumentChunk rows in DB - keeping empty vector store")
            return

        texts = []
        metadatas = []
        for ch in chunks:
            texts.append(ch.content)
            metadatas.append(
                {
                    "doc_id": str(ch.document_id),
                    "title": ch.document.title,
                    "chunk_index": ch.chunk_index,
                    "total_chunks": ch.document.total_chunks,
                    **(ch.metadata or {}),
                }
            )

        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self._add_texts_batched(texts, metadatas)
        self.save_vectorstore()
        self._vectorstore_mtime = self._get_vectorstore_mtime()
        logger.info(
            "Rebuilt FAISS vector store from DB: %s chunks from DocumentChunk",
            len(texts),
        )

    def _get_vectorstore_mtime(self) -> float:
        """Return mtime of index file on disk (0 if missing). Used to detect Celery updates."""
        path = os.path.join(self.vector_store_path, "index.faiss")
        try:
            return os.path.getmtime(path)
        except OSError:
            return 0.0

    def _reload_if_updated(self) -> None:
        """If Celery (or another process) wrote a new index to shared volume, reload it."""
        mtime = self._get_vectorstore_mtime()
        if mtime > 0 and mtime > getattr(self, "_vectorstore_mtime", 0):
            try:
                self.vectorstore = FAISS.load_local(
                    self.vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                self._vectorstore_mtime = mtime
                logger.info("Reloaded RAG vector store from disk (index updated)")
            except Exception as e:
                logger.warning("Could not reload vector store: %s", e)

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def _add_texts_batched(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add texts in batches to avoid OpenAI 300k-token-per-request and 429 rate limits."""
        total = len(texts)
        for i in range(0, total, EMBEDDING_BATCH_SIZE):
            batch_texts = texts[i : i + EMBEDDING_BATCH_SIZE]
            batch_metadatas = metadatas[i : i + EMBEDDING_BATCH_SIZE]
            self.vectorstore.add_texts(batch_texts, batch_metadatas)
            logger.debug(
                "Added batch %d–%d (%d chunks) to vector store",
                i,
                i + len(batch_texts),
                len(batch_texts),
            )
            # Delay between batches to avoid 429 (tokens per minute) rate limit
            if i + EMBEDDING_BATCH_SIZE < total and EMBEDDING_BATCH_DELAY_SECONDS > 0:
                time.sleep(EMBEDDING_BATCH_DELAY_SECONDS)

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        self._add_texts_batched(texts, metadatas)
        self.save_vectorstore()

    def delete_document(self, document_id: str) -> None:
        """
        Remove a document's chunks from the FAISS index by rebuilding the index
        without that document. FAISS does not support per-vector delete.
        """
        from django.apps import apps

        DocumentChunk = apps.get_model("chatbot", "DocumentChunk")

        # Build texts/metadatas for all chunks that are NOT from this document
        chunks = (
            DocumentChunk.objects.exclude(document_id=document_id)
            .select_related("document")
            .order_by("document_id", "chunk_index")
        )
        if not chunks.exists():
            # No other documents: save empty index
            index = faiss.IndexFlatL2(EMBEDDING_DIM)
            self.vectorstore = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            self.save_vectorstore()
            self._vectorstore_mtime = self._get_vectorstore_mtime()
            logger.info(
                f"Removed document {document_id} from vector store; index is now empty"
            )
            return

        texts = []
        metadatas = []
        for ch in chunks:
            texts.append(ch.content)
            metadatas.append(
                {
                    "doc_id": str(ch.document_id),
                    "title": ch.document.title,
                    "chunk_index": ch.chunk_index,
                    "total_chunks": ch.document.total_chunks,
                    **(ch.metadata or {}),
                }
            )

        # Rebuild index from remaining chunks
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self._add_texts_batched(texts, metadatas)
        self.save_vectorstore()
        self._vectorstore_mtime = self._get_vectorstore_mtime()
        logger.info(
            f"Removed document {document_id} from vector store; rebuilt index with {len(texts)} chunks"
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, question: str) -> List[Document]:
        # Reload from disk if Celery (or another worker) updated the index on shared volume
        self._reload_if_updated()

        cache_key = f"retrieval:{hashlib.md5(question.encode()).hexdigest()}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        # Check if vector store has any documents; if empty, try rebuilding from DB
        total_vectors = self.vectorstore.index.ntotal
        if total_vectors == 0:
            self._rebuild_vectorstore_from_db()
            total_vectors = self.vectorstore.index.ntotal
        if total_vectors == 0:
            logger.warning("Vector store is empty - no documents available for RAG")
            return []

        try:
            retriever = self.vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.config["RAG"]["TOP_K_RESULTS"],
                    "score_threshold": 0.25,
                },
            )

            docs = retriever.invoke(question)

            if not docs:
                logger.info(
                    f"No relevant documents found for question: {question[:50]}..."
                )

            cache.set(cache_key, docs, 300)
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    # ------------------------------------------------------------------
    # Query (FAST)
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> Dict[str, Any]:

        start = time.time()
        docs = self._retrieve(question)

        # Build context from retrieved documents
        if docs:
            context = "\n\n".join(d.page_content for d in docs)
            logger.info(f"Retrieved {len(docs)} documents for RAG query")
        else:
            # Empty context - tell LLM to say it doesn't have enough information
            context = "[No relevant documents found in uploaded documents. For specific questions, politely explain that you don't have enough information yet and suggest uploading relevant documents. For conversational questions like greetings, respond naturally.]"
            logger.info(
                "No documents retrieved - will suggest uploading documents for specific questions"
            )

        history = self._format_chat_history(chat_history)
        kb_info = self._get_kb_url_context()

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            chat_history=history,
            question=question,
            knowledge_base_info=kb_info,
        )

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": [d.metadata for d in docs] if docs else [],
            "response_time": time.time() - start,
            "tokens_used": (
                len(response.content.split()) if hasattr(response, "content") else 0
            ),
        }

    # ------------------------------------------------------------------
    # Knowledge Base URL context
    # ------------------------------------------------------------------

    @staticmethod
    def _get_kb_url_context() -> str:
        """
        Build a summary of available Knowledge Base URLs for the prompt.
        Lets the LLM know what external websites are available in its knowledge base.
        """
        from django.apps import apps

        try:
            KnowledgeBaseURL = apps.get_model("chatbot", "KnowledgeBaseURL")
        except LookupError:
            return ""

        active_urls = KnowledgeBaseURL.objects.filter(
            is_active=True, total_chunks__gt=0
        )
        if not active_urls.exists():
            return ""

        lines = ["**Available Knowledge Base URLs (pre-indexed, searchable):**"]
        for u in active_urls:
            desc = f" — {u.description}" if u.description else ""
            lines.append(
                f"- {u.title}: {u.url}{desc} ({u.total_chunks} chunks, refreshed {u.refresh_frequency})"
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Live-data / Web-fetch queries
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_urls(text: str) -> List[str]:
        """Extract URLs from user question text."""
        return _URL_PATTERN.findall(text or "")

    def _add_fetched_content_to_kb(self, url: str, text: str) -> int:
        """
        Chunk fetched web content and add it to the FAISS vector store so it
        becomes part of the chatbot's knowledge base for future queries.
        Returns the number of chunks added.
        """
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            return 0

        texts = []
        metadatas = []
        for i, chunk_text in enumerate(chunks):
            texts.append(chunk_text)
            metadatas.append(
                {
                    "doc_id": f"web_{hashlib.md5(url.encode()).hexdigest()[:12]}",
                    "title": url,
                    "url": url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source_type": "web_fetched",
                    "fetched_from": url,
                }
            )

        self._add_texts_batched(texts, metadatas)
        self.save_vectorstore()
        self._vectorstore_mtime = self._get_vectorstore_mtime()
        logger.info("Added %d chunks from %s to FAISS knowledge base", len(chunks), url)
        return len(chunks)

    def query_with_live_data(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        persist_to_kb: bool = False,
    ) -> Dict[str, Any]:
        """
        Query the RAG system with optional live web fetching.

        Flow:
        1. Extract any URLs from the user question → pre-fetch them
        2. If persist_to_kb=True, chunk fetched content and add to FAISS
        3. Retrieve relevant docs from FAISS (now includes previously fetched URLs)
        4. Give the LLM the combined context AND access to the fetch_webpage tool,
           so it can fetch additional URLs it deems relevant.
        5. If the LLM calls the tool, execute it, feed the result back, and get final answer.
        """
        from .web_fetch_tool import WebFetchTool
        from .web_fetcher import fetch_page_text

        start = time.time()

        # -- Pre-fetch any URLs the user explicitly mentioned --
        urls_in_question = self._extract_urls(question)
        live_content_parts = []
        web_sources = []
        kb_chunks_added = 0

        for url in urls_in_question[:3]:  # max 3 URLs to avoid abuse
            text, error = fetch_page_text(url)
            if text and not error:
                live_content_parts.append(f"--- Content from {url} ---\n{text}")
                web_sources.append(
                    {"title": url, "url": url, "source_type": "live_web"}
                )
                logger.info(f"Pre-fetched URL: {url} ({len(text)} chars)")

                # Persist fetched content to FAISS knowledge base for future queries
                if persist_to_kb:
                    kb_chunks_added += self._add_fetched_content_to_kb(url, text)
            else:
                logger.warning(f"Could not pre-fetch URL {url}: {error}")

        # -- Retrieve from FAISS (may now include previously-fetched web chunks) --
        docs = self._retrieve(question)

        if docs:
            faiss_context = "\n\n".join(d.page_content for d in docs)
            logger.info(f"Retrieved {len(docs)} documents for RAG query")
        else:
            faiss_context = "[No relevant documents found in uploaded documents.]"

        # Merge FAISS context with live content
        combined_context = faiss_context
        if live_content_parts:
            combined_context += "\n\n=== LIVE WEB CONTENT ===\n\n" + "\n\n".join(
                live_content_parts
            )

        history = self._format_chat_history(chat_history)

        # -- Tool-bound LLM for optional additional fetches --
        web_fetch_tool = WebFetchTool()
        llm_with_tools = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7,
        ).bind_tools([web_fetch_tool])

        kb_info = self._get_kb_url_context()
        prompt = RAG_PROMPT_TEMPLATE.format(
            context=combined_context,
            chat_history=history,
            question=question,
            knowledge_base_info=kb_info,
        )

        # First LLM call: may return tool_calls or a direct answer
        response = llm_with_tools.invoke(prompt)

        # If the LLM wants to fetch additional URLs, do it and re-prompt
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_messages = []
            for tool_call in response.tool_calls:
                if tool_call.get("name") == "fetch_webpage":
                    url = tool_call.get("args", {}).get("url", "")
                    query_ctx = tool_call.get("args", {}).get("query_context", "")
                    if url:
                        text, error = fetch_page_text(url)
                        result = text if text else f"[Error: {error}]"
                        tool_messages.append(
                            ToolMessage(
                                content=result,
                                tool_call_id=tool_call.get("id", ""),
                            )
                        )
                        if text:
                            web_sources.append(
                                {"title": url, "url": url, "source_type": "live_web"}
                            )
                            if persist_to_kb:
                                kb_chunks_added += self._add_fetched_content_to_kb(
                                    url, text
                                )

            if tool_messages:
                messages = [HumanMessage(content=prompt), response] + tool_messages
                final_response = self.llm.invoke(messages)
                answer = (
                    final_response.content
                    if hasattr(final_response, "content")
                    else str(final_response)
                )
            else:
                answer = (
                    response.content if hasattr(response, "content") else str(response)
                )
        else:
            answer = response.content if hasattr(response, "content") else str(response)

        all_sources = [d.metadata for d in docs] if docs else []
        all_sources.extend(web_sources)

        result = {
            "answer": answer,
            "sources": all_sources,
            "response_time": time.time() - start,
            "tokens_used": len(answer.split()) if answer else 0,
            "live_data_used": len(web_sources) > 0,
        }
        if persist_to_kb and kb_chunks_added > 0:
            result["kb_chunks_added"] = kb_chunks_added
        return result

    def query_stream_with_live_data(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        persist_to_kb: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming variant of query_with_live_data.
        Pre-fetches URLs, chunks + persists to FAISS if persist_to_kb=True,
        then streams the LLM response.
        """
        from .web_fetcher import fetch_page_text

        yield {"type": "start"}

        # Pre-fetch URLs in the question first (before FAISS retrieval,
        # so newly fetched content is available for retrieval)
        urls_in_question = self._extract_urls(question)
        live_content_parts = []
        web_sources = []
        kb_chunks_added = 0

        for url in urls_in_question[:3]:
            text, error = fetch_page_text(url)
            if text and not error:
                live_content_parts.append(f"--- Content from {url} ---\n{text}")
                web_sources.append(
                    {"title": url, "url": url, "source_type": "live_web"}
                )
                if persist_to_kb:
                    kb_chunks_added += self._add_fetched_content_to_kb(url, text)

        # Now retrieve from FAISS — may include chunks from previously fetched URLs
        docs = self._retrieve(question)
        sources = []

        # Yield FAISS sources
        if docs:
            for doc in docs[:3]:
                sources.append(
                    {
                        "title": doc.metadata.get("title", "Document"),
                        "content": (
                            doc.page_content[:200] + "..."
                            if len(doc.page_content) > 200
                            else doc.page_content
                        ),
                        "metadata": doc.metadata,
                    }
                )

        # Add web sources
        for j, url in enumerate(urls_in_question[:3]):
            preview = ""
            if j < len(live_content_parts):
                content = live_content_parts[j]
                preview = content[:300] + "..." if len(content) > 300 else content
            sources.append(
                {
                    "title": f"Live: {url}",
                    "content": preview,
                    "url": url,
                    "source_type": "live_web",
                }
            )

        if sources:
            yield {"type": "source", "sources": sources}

        # Build combined context
        if docs:
            faiss_context = "\n\n".join(d.page_content for d in docs)
        else:
            faiss_context = "[No relevant documents found in uploaded documents.]"

        combined_context = faiss_context
        if live_content_parts:
            combined_context += "\n\n=== LIVE WEB CONTENT ===\n\n" + "\n\n".join(
                live_content_parts
            )

        history = self._format_chat_history(chat_history)
        kb_info = self._get_kb_url_context()

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=combined_context,
            chat_history=history,
            question=question,
            knowledge_base_info=kb_info,
        )

        streaming_llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=True,
            temperature=0.7,
            max_tokens=800,
        )

        full_response = ""
        try:
            for chunk in streaming_llm.stream(prompt):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield {"type": "token", "content": chunk.content}

            tokens_used = len(full_response.split()) if full_response else 0

            all_sources = sources + ([d.metadata for d in docs] if docs else [])
            complete_event = {
                "type": "complete",
                "full_response": full_response,
                "tokens_used": tokens_used,
                "sources": all_sources,
                "live_data_used": len(web_sources) > 0,
            }
            if persist_to_kb and kb_chunks_added > 0:
                complete_event["kb_chunks_added"] = kb_chunks_added
            yield complete_event
        except Exception as stream_error:
            logger.error(f"Error during live-data streaming: {stream_error}")
            if full_response:
                yield {
                    "type": "complete",
                    "full_response": full_response,
                    "tokens_used": len(full_response.split()),
                    "sources": sources,
                }
            yield {"type": "error", "message": str(stream_error)}

    # ------------------------------------------------------------------
    # Streaming (FAST)
    # ------------------------------------------------------------------

    def query_stream(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
    ) -> Generator[Dict[str, Any], None, None]:

        yield {"type": "start"}

        docs = self._retrieve(question)

        # Build context from retrieved documents and prepare sources
        sources = []
        if docs:
            context = "\n\n".join(d.page_content for d in docs)
            logger.info(f"Retrieved {len(docs)} documents for streaming RAG query")

            # Yield sources immediately for fast feedback
            for doc in docs[:3]:  # Limit to 3 sources
                sources.append(
                    {
                        "title": doc.metadata.get("title", "Document"),
                        "content": (
                            doc.page_content[:200] + "..."
                            if len(doc.page_content) > 200
                            else doc.page_content
                        ),
                        "metadata": doc.metadata,
                    }
                )
            yield {"type": "source", "sources": sources}
        else:
            # Empty context - tell LLM to say it doesn't have enough information for specific questions
            context = "[No relevant documents found in uploaded documents. For specific questions, politely explain that you don't have enough information yet and suggest uploading relevant documents. For conversational questions like greetings, respond naturally.]"
            logger.info(
                "No documents retrieved for streaming - will suggest uploading documents for specific questions"
            )

        history = self._format_chat_history(chat_history)
        kb_info = self._get_kb_url_context()

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            chat_history=history,
            question=question,
            knowledge_base_info=kb_info,
        )

        streaming_llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            streaming=True,
            temperature=0.7,
            max_tokens=800,
        )

        full_response = ""
        tokens_used = 0

        try:
            for chunk in streaming_llm.stream(prompt):
                if hasattr(chunk, "content") and chunk.content:
                    full_response += chunk.content
                    yield {"type": "token", "content": chunk.content}

            # Calculate tokens used (approximate word count)
            tokens_used = len(full_response.split()) if full_response else 0

            # Yield completion with all metadata
            yield {
                "type": "complete",
                "full_response": full_response,
                "tokens_used": tokens_used,
                "sources": sources,
            }
        except Exception as stream_error:
            logger.error(f"Error during LLM streaming: {stream_error}")
            # Yield error but also yield what we have so far
            if full_response:
                yield {
                    "type": "complete",
                    "full_response": full_response,
                    "tokens_used": len(full_response.split()),
                    "sources": sources,
                }
            yield {
                "type": "error",
                "message": str(stream_error),
            }

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _format_chat_history(self, history: Optional[List[tuple]]) -> str:
        """Format chat history for the prompt - include recent conversation for context"""
        if not history:
            return "This is the start of the conversation."

        # Include last 4 exchanges (8 messages) for better context
        recent_history = history[-4:] if len(history) > 4 else history

        formatted = []
        for user_msg, assistant_msg in recent_history:
            formatted.append(f"User: {user_msg}")
            formatted.append(f"Assistant: {assistant_msg}")

        return "\n".join(formatted)

    def get_stats(self):
        """Get statistics about the vector store"""
        total_vectors = self.vectorstore.index.ntotal
        return {
            "total_vectors": total_vectors,
            "embedding_model": self.config["EMBEDDINGS"]["MODEL"],
            "has_documents": total_vectors > 0,
            "vector_store_path": self.vector_store_path,
        }

    def check_vector_store_health(self) -> Dict[str, Any]:
        """Check if vector store is properly initialized and has documents"""
        try:
            total_vectors = self.vectorstore.index.ntotal
            return {
                "healthy": True,
                "total_vectors": total_vectors,
                "has_documents": total_vectors > 0,
                "message": (
                    f"Vector store has {total_vectors} documents"
                    if total_vectors > 0
                    else "Vector store is empty - no documents indexed"
                ),
            }
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                "healthy": False,
                "error": str(e),
                "message": "Vector store health check failed",
            }


class RAGService(FastRAGService):
    """
    Backwards-compatible wrapper for the fast RAG implementation.

    Older parts of the codebase import `RAGService` from `.rag_service`
    or via `chatbot.services`. This thin subclass preserves that API
    while delegating all behavior to `FastRAGService`.
    """

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Adapter to match the older `get_collection_stats()` name used in views.
        """
        return self.get_stats()
