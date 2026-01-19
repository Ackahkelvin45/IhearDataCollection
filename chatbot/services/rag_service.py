import os
import hashlib
import time
import pickle
import logging

from typing import List, Dict, Any, Optional, Generator
from pathlib import Path

from django.conf import settings
from django.core.cache import cache

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.docstore import InMemoryDocstore

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RAGService:
    """RAG Service using FAISS for retrieval"""

    def __init__(self):
        self.config = settings.CHATBOT

        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.OPENAI_API_KEY,
            model=self.config["EMBEDDINGS"]["MODEL"],
        )

        # LLM
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.7,
            streaming=self.config["RAG"]["STREAMING_ENABLED"],
        )

        # Vector store
        self.vector_store_path = os.path.join(settings.BASE_DIR, "vector_store")
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vectorstore = self._initialize_vectorstore()

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config["RAG"]["CHUNK_SIZE"],
            chunk_overlap=self.config["RAG"]["CHUNK_OVERLAP"],
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        logger.info("RAG Service initialized with FAISS")

    # ------------------------------------------------------------------
    # Vector store handling
    # ------------------------------------------------------------------

    def _initialize_vectorstore(self) -> FAISS:
        if not FAISS_AVAILABLE:
            raise RuntimeError("FAISS is not installed. Install faiss-cpu.")

        index_path = os.path.join(self.vector_store_path, "faiss_index.pkl")

        if os.path.exists(index_path):
            try:
                with open(index_path, "rb") as f:
                    logger.info("Loaded existing FAISS index")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")

        # Create empty FAISS index
        dim = len(self.embeddings.embed_query("init"))
        index = faiss.IndexFlatL2(dim)

        vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore({}),
            index_to_docstore_id={},
        )

        logger.info("Created new FAISS index")
        return vectorstore

    def save_vectorstore(self) -> None:
        path = os.path.join(self.vector_store_path, "faiss_index.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(self.vectorstore, f)
            logger.info("FAISS index saved")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    def add_documents(
        self, texts: List[str], metadatas: List[Dict[str, Any]]
    ) -> List[str]:
        try:
            ids = self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
            self.save_vectorstore()
            logger.info(f"Added {len(ids)} chunks to FAISS")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise

    def delete_document(self, doc_id: str) -> None:
        """
        FAISS does not support filtered deletes.
        We rebuild the index excluding the document.
        """
        try:
            remaining_docs: List[Document] = []

            for _id, doc in self.vectorstore.docstore._dict.items():
                if doc.metadata.get("doc_id") != doc_id:
                    remaining_docs.append(doc)

            # Rebuild index
            dim = len(self.embeddings.embed_query("rebuild"))
            index = faiss.IndexFlatL2(dim)

            self.vectorstore = FAISS.from_documents(
                documents=remaining_docs,
                embedding=self.embeddings,
                index=index,
            )

            self.save_vectorstore()
            logger.info(f"Deleted document {doc_id}")

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            raise

    # ------------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------------

    def build_prompt_template(self) -> PromptTemplate:
        template = """You are an intelligent assistant helping users with a data collection project. You have access to project documentation and can provide helpful information.

**Guidelines:**
1. For greetings, casual conversation, and general questions: Respond naturally and helpfully.

2. For project-specific questions: Use the provided documentation context first and foremost.

3. If a question is about the data collection project and you have relevant context: Provide a detailed answer based on the documentation.

4. Only if a technical question about the project cannot be answered from the context: Say "I don't have enough information in the documentation to answer that question."

5. Be conversational, friendly, and helpful while staying focused on the data collection project.

Context (use this for project-specific questions):
{context}

Chat History:
{chat_history}

Question:
{question}

Helpful Answer:
"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "chat_history", "question"],
        )

    # ------------------------------------------------------------------
    # Chain
    # ------------------------------------------------------------------

    def get_conversational_chain(
        self,
        chat_history: Optional[List[tuple]] = None,
        streaming_callback=None,
    ):
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        if chat_history:
            for human, ai in chat_history:
                memory.chat_memory.add_user_message(human)
                memory.chat_memory.add_ai_message(ai)

        llm = self.llm
        if streaming_callback:
            llm = ChatOpenAI(
                model_name=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.7,
                streaming=True,
                callbacks=[streaming_callback],
            )

        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": self.config["RAG"]["TOP_K_RESULTS"]}
            ),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.build_prompt_template()},
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        chat_history: Optional[List[tuple]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:

        start_time = time.time()

        if use_cache:
            cache_key = self._get_cache_key(question, chat_history)
            cached = cache.get(cache_key)
            if cached:
                return cached

        chain = self.get_conversational_chain(chat_history)
        result = chain({"question": question})

        sources = [
            {
                "content": d.page_content,
                "metadata": d.metadata,
                "doc_id": d.metadata.get("doc_id"),
                "title": d.metadata.get("title"),
                "chunk_index": d.metadata.get("chunk_index"),
            }
            for d in result.get("source_documents", [])
        ]

        response = {
            "answer": result["answer"],
            "sources": sources,
            "response_time": time.time() - start_time,
            "tokens_used": self._estimate_tokens(question, result["answer"]),
        }

        if use_cache:
            cache.set(cache_key, response, timeout=3600)

        return response

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def query_stream(
        self, question: str, chat_history: Optional[List[tuple]] = None
    ) -> Generator[Dict[str, Any], None, None]:

        from langchain.callbacks.base import BaseCallbackHandler

        class StreamingCallback(BaseCallbackHandler):
            def __init__(self):
                self.tokens = []

            def on_llm_new_token(self, token: str, **kwargs):
                self.tokens.append(token)

        callback = StreamingCallback()

        try:
            chain = self.get_conversational_chain(chat_history, callback)
            result = chain({"question": question})

            for token in callback.tokens:
                yield {"type": "token", "content": token}

            for doc in result["source_documents"]:
                yield {
                    "type": "source",
                    "doc_id": doc.metadata.get("doc_id"),
                    "title": doc.metadata.get("title"),
                    "excerpt": doc.page_content[:200],
                    "metadata": doc.metadata,
                }

            yield {
                "type": "complete",
                "tokens_used": self._estimate_tokens(question, result["answer"]),
            }

        except Exception as e:
            yield {"type": "error", "message": str(e)}

    # ------------------------------------------------------------------
    # Utils
    # ------------------------------------------------------------------

    def _get_cache_key(self, question: str, chat_history: Optional[List[tuple]]) -> str:
        base = f"{question}:{chat_history}"
        return "rag:" + hashlib.md5(base.encode()).hexdigest()

    def _estimate_tokens(self, question: str, answer: str) -> int:
        return (len(question) + len(answer)) // 4

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            return {
                "total_chunks": self.vectorstore.index.ntotal,
                "embedding_model": self.config["EMBEDDINGS"]["MODEL"],
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
