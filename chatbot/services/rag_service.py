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
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

logger = logging.getLogger(__name__)

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

        # Vector store
        self.vector_store_path = os.path.join(settings.BASE_DIR, "vector_store")
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vectorstore = self._load_or_create_vectorstore()

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
            except Exception:
                logger.warning("Failed to load FAISS index, rebuilding")

        index = faiss.IndexFlatL2(EMBEDDING_DIM)
        return FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore={},
            index_to_docstore_id={},
        )

    def save_vectorstore(self):
        self.vectorstore.save_local(self.vector_store_path)

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]]):
        self.vectorstore.add_texts(texts, metadatas)
        self.save_vectorstore()

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def _prompt(self) -> PromptTemplate:
        template = """You are an expert AI assistant for **The I Hear Project**.

IMPORTANT: Use ONLY the information provided in the Context section below. If the Context is empty or doesn't contain relevant information, you MUST say that you don't have specific information about that topic in the uploaded documents, and suggest that the user upload relevant documents or ask about their datasets instead.

Context from uploaded documents:
{context}

Previous conversation:
{chat_history}

User's question:
{question}

Instructions:
- If the Context contains relevant information, use it to answer the question accurately
- If the Context is empty or doesn't have relevant information, politely explain that you don't have specific information about this topic in the uploaded documents
- Always be honest about what information you have access to
- If you don't have relevant context, suggest the user upload documents or ask about their datasets

Answer clearly, practically, and concisely."""
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"],
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, question: str) -> List[Document]:
        cache_key = f"retrieval:{hashlib.md5(question.encode()).hexdigest()}"
        cached = cache.get(cache_key)
        if cached:
            return cached

        # Check if vector store has any documents
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
            context = "[No relevant documents found in the uploaded documents. The vector store may be empty or no documents match this query.]"
            logger.warning("No documents retrieved - RAG will work without context")

        history = self._format_chat_history(chat_history)

        prompt = self._prompt().format(
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
            context = "[No relevant documents found in the uploaded documents. The vector store may be empty or no documents match this query.]"
            logger.warning("No documents retrieved for streaming - RAG will work without context")
        
        history = self._format_chat_history(chat_history)

        prompt = self._prompt().format(
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
        if not history:
            return "No previous conversation."
        return "\n".join(
            f"User: {h}\nAssistant: {a}" for h, a in history[-2:]
        )

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
