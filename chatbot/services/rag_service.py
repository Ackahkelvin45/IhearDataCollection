import os
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
logger = logging.getLogger(__name__)

# Prompt template as plain string to avoid LangChain PromptTemplate format quirks
RAG_PROMPT_TEMPLATE = """
You are the assistant for The I Hear Project. You help users find where to click, answer questions from uploaded documents, and answer basic dataset counts; for deeper analysis you suggest Data Insights.

You are a helpful and friendly AI assistant for **The I Hear Project** - a data collection and analysis platform focused on noise datasets and audio recordings.

Your role:
- Be conversational and helpful for general greetings and casual conversation
- When relevant documents are provided in the Context section, use them to give accurate, document-based answers
- When no relevant documents are available for specific questions, politely explain that you don't have enough information yet and suggest uploading relevant documents
- Maintain conversation context and remember what was discussed earlier

Context from uploaded documents (if available):
{context}

Previous conversation:
{chat_history}

User's question:
{question}

Instructions:
1. **For conversational questions** (greetings like "hi", "how are you", "hello", "thanks", etc.): Respond naturally and warmly, then offer to help with The I Hear Project or their datasets.

2. **If Context contains relevant information**: Use it to provide accurate, document-based answers.

3. **If Context is empty or not relevant**: Politely explain that you don't have enough information yet and suggest uploading relevant documents.

Answer clearly, helpfully, and in a friendly conversational tone.
"""

EMBEDDING_DIM = 1536  # text-embedding-3-small


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

        chunks = (
            DocumentChunk.objects.select_related("document")
            .order_by("document_id", "chunk_index")
        )
        if not chunks.exists():
            logger.info("No DocumentChunk rows in DB - keeping empty vector store")
            return

        texts = []
        metadatas = []
        for ch in chunks:
            texts.append(ch.content)
            metadatas.append({
                "doc_id": str(ch.document_id),
                "title": ch.document.title,
                "chunk_index": ch.chunk_index,
                "total_chunks": ch.document.total_chunks,
                **(ch.metadata or {}),
            })

        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.vectorstore.add_texts(texts, metadatas)
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

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        self.vectorstore.add_texts(texts, metadatas)
        self.save_vectorstore()

    def delete_document(self, document_id: str) -> None:
        """
        Remove a document's chunks from the FAISS index by rebuilding the index
        without that document. FAISS does not support per-vector delete.
        """
        from django.apps import apps
        DocumentChunk = apps.get_model("chatbot", "DocumentChunk")

        # Build texts/metadatas for all chunks that are NOT from this document
        chunks = DocumentChunk.objects.exclude(document_id=document_id).select_related("document").order_by("document_id", "chunk_index")
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
            logger.info(f"Removed document {document_id} from vector store; index is now empty")
            return

        texts = []
        metadatas = []
        for ch in chunks:
            texts.append(ch.content)
            metadatas.append({
                "doc_id": str(ch.document_id),
                "title": ch.document.title,
                "chunk_index": ch.chunk_index,
                "total_chunks": ch.document.total_chunks,
                **(ch.metadata or {}),
            })

        # Rebuild index from remaining chunks
        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.vectorstore.add_texts(texts, metadatas)
        self.save_vectorstore()
        self._vectorstore_mtime = self._get_vectorstore_mtime()
        logger.info(f"Removed document {document_id} from vector store; rebuilt index with {len(texts)} chunks")

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
                logger.info(f"No relevant documents found for question: {question[:50]}...")
            
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
            logger.info("No documents retrieved - will suggest uploading documents for specific questions")

        history = self._format_chat_history(chat_history)

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            chat_history=history,
            question=question,
        )

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": [d.metadata for d in docs] if docs else [],
            "response_time": time.time() - start,
            "tokens_used": len(response.content.split()) if hasattr(response, 'content') else 0,
        }

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
                sources.append({
                    "title": doc.metadata.get("title", "Document"),
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata,
                })
            yield {"type": "source", "sources": sources}
        else:
            # Empty context - tell LLM to say it doesn't have enough information for specific questions
            context = "[No relevant documents found in uploaded documents. For specific questions, politely explain that you don't have enough information yet and suggest uploading relevant documents. For conversational questions like greetings, respond naturally.]"
            logger.info("No documents retrieved for streaming - will suggest uploading documents for specific questions")
        
        history = self._format_chat_history(chat_history)

        prompt = RAG_PROMPT_TEMPLATE.format(
            context=context,
            chat_history=history,
            question=question,
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
                "message": f"Vector store has {total_vectors} documents" if total_vectors > 0 else "Vector store is empty - no documents indexed",
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
