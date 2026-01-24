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
from langchain.memory import ConversationBufferWindowMemory
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
        template = """You are an expert AI assistant specialized in data collection projects, audio processing, and data analysis. You help users with their data collection project by providing comprehensive support, insights, and guidance.

**Your Expertise Areas:**
- Data collection project management and workflows
- Audio recording and noise data analysis
- Dataset management and statistics
- Technical documentation and best practices
- Data visualization and insights
- Troubleshooting and problem-solving

**Response Guidelines:**
1. **Be Comprehensive**: Provide detailed, helpful answers that go beyond basic information. Include relevant context, examples, and actionable insights.

2. **Use Documentation Wisely**: When relevant project documentation is provided in the context, incorporate it naturally into your response. Don't force it if it's not directly relevant.

3. **Be Proactive**: Offer suggestions, best practices, and next steps. Anticipate what the user might need next.

4. **Handle All Question Types**:
   - For technical questions: Provide step-by-step guidance
   - For analytical questions: Offer data-driven insights and interpretations
   - For procedural questions: Give clear, actionable instructions
   - For troubleshooting: Provide systematic solutions and alternatives

5. **Be Conversational & Helpful**: Respond naturally, use appropriate formatting, and encourage further interaction. Be encouraging and supportive.

6. **Data Analysis Focus**: When discussing datasets, recordings, or analysis:
   - Provide statistical insights and interpretations
   - Suggest visualization approaches
   - Explain data patterns and trends
   - Recommend data quality improvements

7. **Project Context**: This is an audio data collection project focused on capturing and analyzing noise recordings from different environments, regions, and categories.

Context from project documentation (use when directly relevant):
{context}

Chat History:
{chat_history}

Question: {question}

Expert Answer:
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
        memory = ConversationBufferWindowMemory(
            k=5,  # Keep last 5 exchanges to avoid token limits
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
        result = chain.invoke({"question": question})

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

        try:
            # Send start event immediately
            yield {"type": "start"}

            # Retrieve relevant documents quickly
            retriever = self.vectorstore.as_retriever(
                search_kwargs={"k": self.config["RAG"]["TOP_K_RESULTS"]}
            )
            docs = retriever.invoke(question)

            # Yield sources immediately for fast feedback
            if docs:
                sources = []
                for doc in docs[:3]:  # Limit to 3 sources for speed
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

            # Create context from retrieved documents
            context = "\n\n".join([doc.page_content for doc in docs])

            # Build prompt
            prompt = self.build_prompt_template()
            formatted_prompt = prompt.format(
                context=context,
                chat_history=self._format_chat_history(chat_history or []),
                question=question,
            )

            # Create streaming LLM with optimized settings for speed
            streaming_llm = ChatOpenAI(
                model_name=settings.OPENAI_MODEL,
                openai_api_key=settings.OPENAI_API_KEY,
                temperature=0.7,
                streaming=True,
                max_tokens=800,  # Reduced for faster responses
                request_timeout=30,  # Reasonable timeout
            )

            # Use streaming iteration to get tokens immediately
            accumulated_response = ""

            try:
                # Stream the response using async iteration
                for chunk in streaming_llm.stream(formatted_prompt):
                    if hasattr(chunk, "content") and chunk.content:
                        token = chunk.content
                        accumulated_response += token

                        # Yield each token immediately for real-time streaming
                        yield {"type": "token", "content": token}

            except Exception as stream_error:
                logger.warning(
                    f"Streaming failed, falling back to regular call: {stream_error}"
                )

                # Fallback to regular call if streaming fails
                response = streaming_llm.invoke(formatted_prompt)
                full_content = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # Yield the full content as tokens for compatibility
                words = full_content.split()
                for word in words:
                    yield {"type": "token", "content": word + " "}
                    # Small delay to simulate streaming effect
                    import time

                    time.sleep(0.01)

                accumulated_response = full_content

            # Yield completion with final data
            yield {
                "type": "complete",
                "tokens_used": len(
                    accumulated_response.split()
                ),  # Approximate word count
                "full_response": accumulated_response,
                "response_time": 0.1,  # Placeholder
            }

        except Exception as e:
            logger.error(f"Error in streaming query: {e}")
            yield {"type": "error", "message": str(e)}

    def _format_chat_history(self, chat_history: List[tuple]) -> str:
        """Format chat history for the prompt"""
        if not chat_history:
            return "No previous conversation."

        formatted = []
        for human, ai in chat_history[-2:]:  # Last 2 exchanges for brevity
            formatted.append(f"User: {human}")
            formatted.append(f"Assistant: {ai}")

        return "\n".join(formatted)

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
