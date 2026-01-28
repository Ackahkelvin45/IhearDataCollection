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
        template = """
You are an expert AI assistant for **The I Hear Project**.

Context:
{context}

Chat History:
{chat_history}

Question:
{question}

Answer clearly, practically, and concisely.
"""
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

        retriever = self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.config["RAG"]["TOP_K_RESULTS"],
                "score_threshold": 0.25,
            },
        )

        docs = retriever.invoke(question)
        cache.set(cache_key, docs, 300)
        return docs

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

        context = "\n\n".join(d.page_content for d in docs)
        history = self._format_chat_history(chat_history)

        prompt = self._prompt().format(
            context=context,
            chat_history=history,
            question=question,
        )

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": [d.metadata for d in docs],
            "response_time": time.time() - start,
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
        context = "\n\n".join(d.page_content for d in docs)
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

        for chunk in streaming_llm.stream(prompt):
            if chunk.content:
                full_response += chunk.content
                yield {"type": "token", "content": chunk.content}

        yield {
            "type": "complete",
            "full_response": full_response,
            "tokens_used": len(full_response.split()),
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
        return {
            "total_vectors": self.vectorstore.index.ntotal,
            "embedding_model": self.config["EMBEDDINGS"]["MODEL"],
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
